"""ModelDistributor — announce model updates across the Homie mesh."""
from __future__ import annotations
from homie_core.mesh.events import HomieEvent
from homie_core.mesh.mesh_manager import MeshManager


class ModelDistributor:
    """Publishes model lifecycle events to the mesh via MeshManager."""

    def __init__(self, mesh_manager: MeshManager) -> None:
        self._mgr = mesh_manager

    def announce_update(
        self,
        model_name: str,
        model_path: str,
        score_improvement: float,
        cycle: int,
    ) -> HomieEvent:
        """Emit a model_updated event so all mesh nodes know about the new model."""
        return self._mgr.emit(
            "learning",
            "model_updated",
            {
                "model_name": model_name,
                "model_path": model_path,
                "score_improvement": score_improvement,
                "cycle": cycle,
            },
        )

    def announce_training_started(
        self,
        cycle: int,
        sft_pairs: int,
        dpo_pairs: int,
    ) -> HomieEvent:
        """Emit a training_started event so nodes know a new training cycle began."""
        return self._mgr.emit(
            "learning",
            "training_started",
            {"cycle": cycle, "sft_pairs": sft_pairs, "dpo_pairs": dpo_pairs},
        )

    def announce_training_completed(
        self,
        cycle: int,
        score: float,
        promoted: bool,
    ) -> HomieEvent:
        """Emit a training_completed event with the outcome of the training cycle."""
        return self._mgr.emit(
            "learning",
            "training_completed",
            {"cycle": cycle, "score": score, "promoted": promoted},
        )

    def get_model_history(self) -> list[HomieEvent]:
        """Return all model_updated events recorded on this node."""
        return [
            e
            for e in self._mgr._event_store.events_by_category("learning")
            if e.event_type == "model_updated"
        ]
