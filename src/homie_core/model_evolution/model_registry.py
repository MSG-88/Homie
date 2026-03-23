"""Model registry — tracks all Homie model versions."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelVersion:
    """A versioned Homie model."""

    version_id: str
    base_model: str
    ollama_name: str
    modelfile_hash: str
    status: str = "created"  # created, shadow_testing, active, archived, rolled_back
    metrics: dict = field(default_factory=dict)
    changelog: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "base_model": self.base_model,
            "ollama_name": self.ollama_name,
            "modelfile_hash": self.modelfile_hash,
            "status": self.status,
            "metrics": json.dumps(self.metrics) if isinstance(self.metrics, dict) else self.metrics,
            "changelog": self.changelog,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        metrics = data.get("metrics", "{}")
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        return cls(
            version_id=data["version_id"],
            base_model=data["base_model"],
            ollama_name=data["ollama_name"],
            modelfile_hash=data["modelfile_hash"],
            status=data.get("status", "created"),
            metrics=metrics,
            changelog=data.get("changelog", ""),
            created_at=data.get("created_at", time.time()),
        )


class ModelRegistry:
    """Manages model version lifecycle."""

    def __init__(self, storage) -> None:
        self._storage = storage
        self._version_counter = 0

    def register(self, base_model: str, ollama_name: str, modelfile_hash: str, changelog: str = "") -> ModelVersion:
        """Register a new model version."""
        self._version_counter += 1
        version = ModelVersion(
            version_id=f"homie-v{self._version_counter}",
            base_model=base_model,
            ollama_name=ollama_name,
            modelfile_hash=modelfile_hash,
            changelog=changelog,
        )
        self._storage.save_model_version(version.version_id, version.to_dict())
        return version

    def get_active(self) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        data = self._storage.get_active_model_version()
        return ModelVersion.from_dict(data) if data else None

    def promote(self, version_id: str) -> None:
        """Promote a version to active, archiving the current active."""
        current = self.get_active()
        if current and current.version_id != version_id:
            self._storage.update_model_version_status(current.version_id, "archived")
        self._storage.update_model_version_status(version_id, "active")

    def rollback(self, current_version_id: str) -> Optional[ModelVersion]:
        """Rollback current version and restore previous."""
        self._storage.update_model_version_status(current_version_id, "rolled_back")
        prev_data = self._storage.get_previous_model_version()
        if prev_data:
            prev = ModelVersion.from_dict(prev_data)
            self._storage.update_model_version_status(prev.version_id, "active")
            return prev
        return None

    def list_versions(self) -> list[ModelVersion]:
        """List all versions."""
        rows = self._storage.list_model_versions()
        return [ModelVersion.from_dict(r) for r in rows]
