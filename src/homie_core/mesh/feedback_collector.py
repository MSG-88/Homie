"""FeedbackCollector — Capture implicit learning signals from user interactions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from homie_core.mesh.events import generate_ulid
from homie_core.utils import utc_now


class SignalType:
    ACCEPTED = "accepted"
    REGENERATED = "regenerated"
    CORRECTED = "corrected"
    IGNORED = "ignored"
    RATED = "rated"


@dataclass
class FeedbackSignal:
    signal_type: str
    query: str
    response_preview: str
    node_id: str
    activity_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    signal_id: str = field(default_factory=generate_ulid)
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "query": self.query,
            "response_preview": self.response_preview,
            "node_id": self.node_id,
            "activity_context": self.activity_context,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackSignal":
        sig = cls(
            signal_type=data["signal_type"],
            query=data["query"],
            response_preview=data["response_preview"],
            node_id=data["node_id"],
            activity_context=data.get("activity_context", ""),
            metadata=data.get("metadata", {}),
        )
        sig.signal_id = data["signal_id"]
        sig.timestamp = data["timestamp"]
        return sig


class FeedbackCollector:
    def __init__(self, node_id: str, activity_context: str = "") -> None:
        self.node_id = node_id
        self.activity_context = activity_context
        self.signals: list[FeedbackSignal] = []

    def _record(
        self,
        signal_type: str,
        query: str,
        response_preview: str,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackSignal:
        sig = FeedbackSignal(
            signal_type=signal_type,
            query=query,
            response_preview=response_preview,
            node_id=self.node_id,
            activity_context=self.activity_context,
            metadata=metadata or {},
        )
        self.signals.append(sig)
        return sig

    def record_accepted(self, query: str, response: str) -> FeedbackSignal:
        """User accepted / used the response without modification."""
        return self._record(SignalType.ACCEPTED, query, response)

    def record_regenerated(
        self, query: str, original: str, regenerated: str
    ) -> FeedbackSignal:
        """User asked for a new response (thumbs-down / regenerate)."""
        return self._record(
            SignalType.REGENERATED,
            query,
            regenerated,
            metadata={"original": original},
        )

    def record_corrected(
        self, query: str, original: str, correction: str
    ) -> FeedbackSignal:
        """User manually corrected the response."""
        return self._record(
            SignalType.CORRECTED,
            query,
            original,
            metadata={"correction": correction},
        )

    def record_ignored(self, query: str, response: str) -> FeedbackSignal:
        """User ignored / dismissed the response."""
        return self._record(SignalType.IGNORED, query, response)

    def record_rated(
        self, query: str, response: str, rating: int
    ) -> FeedbackSignal:
        """User gave an explicit numeric rating."""
        return self._record(
            SignalType.RATED,
            query,
            response,
            metadata={"rating": rating},
        )

    def flush(self) -> list[FeedbackSignal]:
        """Return all buffered signals and clear the buffer."""
        batch, self.signals = self.signals, []
        return batch
