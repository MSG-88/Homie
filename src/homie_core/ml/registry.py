"""Model registry — tracks all local ML models (trained, deployed, archived)."""

from __future__ import annotations

import time
from typing import Any, Optional

from homie_core.ml.base import LocalModel


class MLModelRegistry:
    """Central registry for locally-trained ML models.

    Keeps track of model instances, their metrics, deployment status, and
    archive state.
    """

    def __init__(self) -> None:
        self._models: dict[str, LocalModel] = {}
        self._metrics: dict[str, dict[str, Any]] = {}
        self._status: dict[str, str] = {}  # "active", "archived", "deployed"
        self._timestamps: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, model: LocalModel) -> str:
        """Register *model* in the registry.

        Returns the model's name as a confirmation key.
        """
        name = model.name
        self._models[name] = model
        self._status[name] = "active"
        self._timestamps[name] = time.time()
        self._metrics[name] = model.metrics
        return name

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[LocalModel]:
        """Return the model instance with *name*, or ``None`` if not found."""
        return self._models.get(name)

    def has(self, name: str) -> bool:
        """Return ``True`` if *name* is registered."""
        return name in self._models

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_active(self) -> list[str]:
        """Return names of all non-archived models."""
        return [n for n, s in self._status.items() if s != "archived"]

    def list_all(self) -> list[str]:
        """Return names of all models (including archived)."""
        return list(self._models.keys())

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self, name: str) -> dict:
        """Return the last-known metrics for *name*.

        If metrics were manually set via :meth:`update_metrics` they take
        precedence.  Otherwise the model's own metrics are returned.
        """
        if name not in self._models:
            raise KeyError(f"Model {name!r} is not registered.")
        stored = self._metrics.get(name)
        if stored:
            return dict(stored)
        # Fall back to the model's own metrics
        return dict(self._models[name].metrics)

    def update_metrics(self, name: str, metrics: dict) -> None:
        """Manually override stored metrics for *name*."""
        if name not in self._models:
            raise KeyError(f"Model {name!r} is not registered.")
        self._metrics[name] = dict(metrics)

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def set_status(self, name: str, status: str) -> None:
        """Set the status for *name* (``'active'``, ``'deployed'``, ``'archived'``)."""
        if name not in self._models:
            raise KeyError(f"Model {name!r} is not registered.")
        if status not in ("active", "deployed", "archived"):
            raise ValueError(f"Invalid status {status!r}.")
        self._status[name] = status

    def get_status(self, name: str) -> str:
        """Return the current status of *name*."""
        if name not in self._models:
            raise KeyError(f"Model {name!r} is not registered.")
        return self._status[name]

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def archive(self, name: str) -> None:
        """Archive a model (soft-delete)."""
        self.set_status(name, "archived")

    def remove(self, name: str) -> bool:
        """Permanently remove a model from the registry."""
        if name not in self._models:
            return False
        del self._models[name]
        self._metrics.pop(name, None)
        self._status.pop(name, None)
        self._timestamps.pop(name, None)
        return True

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> list[dict[str, Any]]:
        """Return a list of summary dicts for every registered model."""
        results = []
        for name, model in self._models.items():
            results.append({
                "name": name,
                "type": model.model_type,
                "trained": model.is_trained,
                "status": self._status.get(name, "unknown"),
                "metrics": self._metrics.get(name, {}),
            })
        return results
