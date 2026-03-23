"""Base model interface for all local ML models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class LocalModel(ABC):
    """Abstract base class for all locally-trained ML models.

    Every model must support training, prediction, and persistence (save/load).
    The ``model_type`` field indicates the kind of model: ``"classifier"``,
    ``"embedder"``, or ``"regressor"``.
    """

    name: str
    model_type: str  # "classifier", "embedder", "regressor"

    def __init__(self, name: str, model_type: str) -> None:
        self.name = name
        self.model_type = model_type
        self._trained = False
        self._metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, X: list, y: list) -> dict:
        """Train the model on feature list *X* and labels *y*.

        Returns a dict of training metrics (accuracy, loss, etc.).
        """

    @abstractmethod
    def predict(self, X: list) -> list:
        """Return predictions for the given inputs."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model state to *path*."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model state from *path*."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Return ``True`` if the model has been trained at least once."""
        return self._trained

    @property
    def metrics(self) -> dict[str, Any]:
        """Return the most-recent training metrics."""
        return dict(self._metrics)

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return f"<{self.__class__.__name__} name={self.name!r} type={self.model_type!r} {status}>"
