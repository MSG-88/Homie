"""Mesh REST API — FastAPI router for mesh operations.

Provides HTTP endpoints for mesh management, node status, inference,
task dispatch, and context queries. Authenticated via API key.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Header
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None


# --- Request/Response models ---

class NodeStatusResponse(BaseModel):
    node_id: str
    node_name: str
    role: str
    mesh_id: Optional[str] = None
    capability_score: float
    gpu: Optional[dict] = None
    cpu_cores: int
    ram_gb: float
    os: str

class MeshStatusResponse(BaseModel):
    node_id: str
    node_name: str
    role: str
    mesh_id: Optional[str] = None
    nodes: list[dict] = []
    event_count: int

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    content: str
    source: str

class TaskRequest(BaseModel):
    target_node: str
    command: str
    reason: str = ""
    dry_run: bool = False

class TaskResponse(BaseModel):
    task_id: str
    state: str
    stdout: str = ""
    error: Optional[str] = None

class ContextResponse(BaseModel):
    context_block: str
    active_nodes: list[str]
    primary_node: Optional[str] = None

class FeedbackRequest(BaseModel):
    signal_type: str
    query: str
    response_preview: str

class TrainingSummaryResponse(BaseModel):
    total_signals: int
    sft_pairs: int
    dpo_pairs: int
    ready: bool


def create_mesh_router(mesh_context) -> Optional[APIRouter]:
    """Create the mesh API router with all endpoints.

    Returns None if FastAPI is not installed.
    """
    if not HAS_FASTAPI:
        logger.warning("FastAPI not installed — mesh API disabled")
        return None

    router = APIRouter(prefix="/api/mesh", tags=["mesh"])
    ctx = mesh_context

    async def verify_api_key(authorization: str = Header(default="")) -> None:
        """Verify API key from Authorization header."""
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing API key")
        key = authorization.replace("Bearer ", "")
        user = ctx.auth_store.authenticate(key)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return user

    @router.get("/status", response_model=MeshStatusResponse)
    async def mesh_status():
        nodes = ctx.registry.list_all()
        return MeshStatusResponse(
            node_id=ctx.identity.node_id,
            node_name=ctx.identity.node_name,
            role=ctx.identity.role,
            mesh_id=ctx.identity.mesh_id,
            nodes=[n.__dict__ for n in nodes],
            event_count=ctx.mesh_manager.event_count(),
        )

    @router.get("/node", response_model=NodeStatusResponse)
    async def node_status():
        caps = ctx.capabilities
        return NodeStatusResponse(
            node_id=ctx.identity.node_id,
            node_name=ctx.identity.node_name,
            role=ctx.identity.role,
            mesh_id=ctx.identity.mesh_id,
            capability_score=caps.capability_score(),
            gpu=caps.gpu,
            cpu_cores=caps.cpu_cores,
            ram_gb=caps.ram_gb,
            os=caps.os,
        )

    @router.get("/context", response_model=ContextResponse)
    async def get_context():
        ctx.collect_context()
        return ContextResponse(
            context_block=ctx.get_cross_device_block(),
            active_nodes=ctx.user_model.active_nodes,
            primary_node=ctx.user_model.primary_node,
        )

    @router.post("/task", response_model=TaskResponse)
    async def dispatch_task(req: TaskRequest, user=Depends(verify_api_key)):
        if not user.can_execute_tasks():
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        from homie_core.mesh.task_model import MeshTask
        task = MeshTask(
            source_node=ctx.identity.node_id,
            target_node=req.target_node,
            command=req.command,
            reason=req.reason,
            dry_run=req.dry_run,
        )
        result = ctx.task_dispatcher.dispatch(task)
        ctx.auth_store.log_action(
            user.user_id, "task_dispatch",
            target=req.target_node,
            details={"command": req.command, "state": result.state},
            source_node=ctx.identity.node_id,
            target_node=req.target_node,
        )
        return TaskResponse(
            task_id=result.task_id,
            state=result.state,
            stdout=result.stdout,
            error=result.error,
        )

    @router.post("/feedback")
    async def record_feedback(req: FeedbackRequest):
        from homie_core.mesh.feedback_collector import FeedbackSignal, SignalType
        sig = FeedbackSignal(
            signal_type=req.signal_type,
            query=req.query,
            response_preview=req.response_preview,
            node_id=ctx.identity.node_id,
        )
        ctx.feedback_store.save(sig)
        ctx.mesh_manager.emit("preference", "feedback_signal", sig.to_dict())
        return {"status": "recorded", "signal_id": sig.signal_id}

    @router.get("/training", response_model=TrainingSummaryResponse)
    async def training_status():
        summary = ctx.training_trigger.get_summary()
        return TrainingSummaryResponse(
            total_signals=summary["total_signals"],
            sft_pairs=summary["sft_pairs"],
            dpo_pairs=summary["dpo_pairs"],
            ready=summary["ready"],
        )

    @router.get("/events")
    async def list_events(limit: int = 50, after: Optional[str] = None):
        events = ctx.mesh_manager.events_since(after, limit=limit)
        return [e.to_dict() for e in events]

    @router.get("/health")
    async def health():
        return {"status": "ok", "node_id": ctx.identity.node_id, "events": ctx.mesh_manager.event_count()}

    return router
