"""Mesh bootstrap — initialize all mesh components on node startup."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from homie_core.config import HomieConfig
from homie_core.mesh.identity import NodeIdentity, load_identity, save_identity
from homie_core.mesh.capabilities import detect_capabilities, NodeCapabilities
from homie_core.mesh.registry import MeshNodeRegistry
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.node_context import NodeContext, collect_local_context
from homie_core.mesh.unified_user_model import UnifiedUserModel
from homie_core.mesh.cross_device_perceiver import CrossDevicePerceiver
from homie_core.mesh.context_handoff import ContextHandoff
from homie_core.mesh.feedback_collector import FeedbackCollector
from homie_core.mesh.feedback_store import FeedbackStore
from homie_core.mesh.training_trigger import TrainingTrigger
from homie_core.mesh.task_executor import MeshTaskExecutor
from homie_core.mesh.task_dispatcher import TaskDispatcher
from homie_core.mesh.auth import AuthStore, Role

logger = logging.getLogger(__name__)


class MeshContext:
    """Container for all initialized mesh components."""

    def __init__(
        self,
        identity: NodeIdentity,
        capabilities: NodeCapabilities,
        mesh_manager: MeshManager,
        registry: MeshNodeRegistry,
        user_model: UnifiedUserModel,
        perceiver: CrossDevicePerceiver,
        handoff: ContextHandoff,
        feedback_collector: FeedbackCollector,
        feedback_store: FeedbackStore,
        training_trigger: TrainingTrigger,
        task_executor: MeshTaskExecutor,
        task_dispatcher: TaskDispatcher,
        auth_store: AuthStore,
        enabled: bool = True,
    ):
        self.identity = identity
        self.capabilities = capabilities
        self.mesh_manager = mesh_manager
        self.registry = registry
        self.user_model = user_model
        self.perceiver = perceiver
        self.handoff = handoff
        self.feedback_collector = feedback_collector
        self.feedback_store = feedback_store
        self.training_trigger = training_trigger
        self.task_executor = task_executor
        self.task_dispatcher = task_dispatcher
        self.auth_store = auth_store
        self.enabled = enabled

    @property
    def node_id(self) -> str:
        return self.identity.node_id

    @property
    def node_name(self) -> str:
        return self.identity.node_name

    def collect_context(self) -> NodeContext:
        """Collect current node context and update unified model."""
        ctx = collect_local_context(
            node_id=self.identity.node_id,
            node_name=self.identity.node_name,
        )
        self.user_model.update_node(ctx)
        return ctx

    def get_cross_device_block(self) -> str:
        """Get the cross-device context block for Brain injection."""
        return self.perceiver.get_context_block()


def bootstrap_mesh(config: HomieConfig, data_dir: Optional[Path] = None) -> MeshContext:
    """Initialize all mesh components. Called once at node startup.

    Returns a MeshContext with references to all components.
    If mesh is disabled in config, returns a minimal context with enabled=False.
    """
    storage_path = Path(config.storage.path)
    mesh_dir = Path(data_dir) if data_dir else storage_path / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # 1. Identity
    node_path = storage_path / "node.json"
    identity = load_identity(node_path)
    if identity is None:
        identity = NodeIdentity.generate()
        save_identity(identity, node_path)
        logger.info("Created new node identity: %s (%s)", identity.node_id, identity.node_name)
    else:
        logger.info("Loaded node identity: %s (%s)", identity.node_id, identity.node_name)

    # 2. Capabilities
    capabilities = detect_capabilities()
    logger.info("Node capabilities: score=%.0f, gpu=%s", capabilities.capability_score(),
                capabilities.gpu["name"] if capabilities.gpu else "none")

    # 3. Registry
    registry = MeshNodeRegistry(mesh_dir / "nodes.db")
    registry.initialize()

    # 4. Mesh manager (events + sync)
    mesh_manager = MeshManager(identity=identity, data_dir=mesh_dir)

    # 5. Context systems
    user_model = UnifiedUserModel()
    handoff = ContextHandoff()
    perceiver = CrossDevicePerceiver(unified_model=user_model)

    # 6. Feedback + learning
    feedback_collector = FeedbackCollector(node_id=identity.node_id)
    feedback_store = FeedbackStore(mesh_dir / "feedback.db")
    feedback_store.initialize()
    training_trigger = TrainingTrigger(feedback_store=feedback_store)

    # 7. Task execution
    task_executor = MeshTaskExecutor()
    task_dispatcher = TaskDispatcher(
        local_node_id=identity.node_id,
        executor=task_executor,
    )

    # 8. Auth
    auth_store = AuthStore(mesh_dir / "auth.db")
    auth_store.initialize()

    # Create default admin user if none exists
    if not auth_store.list_users():
        user, api_key = auth_store.create_user(
            username=config.user.name or "admin",
            role=Role.ADMIN,
            node_id=identity.node_id,
        )
        logger.info("Created default admin user: %s (API key saved to vault)", user.username)

    # Emit startup event
    mesh_manager.emit("system", "node_started", {
        "node_id": identity.node_id,
        "node_name": identity.node_name,
        "capability_score": capabilities.capability_score(),
        "role": identity.role,
    })

    logger.info("Mesh bootstrap complete: node=%s, role=%s, score=%.0f",
                identity.node_name, identity.role, capabilities.capability_score())

    return MeshContext(
        identity=identity,
        capabilities=capabilities,
        mesh_manager=mesh_manager,
        registry=registry,
        user_model=user_model,
        perceiver=perceiver,
        handoff=handoff,
        feedback_collector=feedback_collector,
        feedback_store=feedback_store,
        training_trigger=training_trigger,
        task_executor=task_executor,
        task_dispatcher=task_dispatcher,
        auth_store=auth_store,
        enabled=config.mesh.enabled,
    )
