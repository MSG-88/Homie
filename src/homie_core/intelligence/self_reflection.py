"""Reflective self-correction with Platt-scaled confidence calibration.

Implements a self-reflection layer that scores proposed actions across
multiple dimensions (relevance, helpfulness, urgency), then calibrates
the raw confidence using Platt scaling so the output probability is
well-calibrated against real user feedback.

Reference: Platt (1999) "Probabilistic Outputs for SVMs"
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ReflectionResult:
    """Outcome of scoring a proposed action."""

    action: str
    raw_score: float
    confidence: float
    calibrated_confidence: float
    reasoning: Dict[str, Any] = field(default_factory=dict)


class SelfReflection:
    """Score and calibrate proposed assistant actions.

    Parameters
    ----------
    dimension_weights : dict | None
        Custom weights for scoring dimensions.  Defaults to
        ``{"relevance": 0.4, "helpfulness": 0.35, "urgency": 0.25}``.
    action_threshold : float
        Minimum *calibrated* confidence required for ``should_act`` to
        return ``True``.
    learning_rate : float
        Step size for online gradient-descent updates to Platt parameters.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "relevance": 0.4,
        "helpfulness": 0.35,
        "urgency": 0.25,
    }

    def __init__(
        self,
        dimension_weights: Optional[Dict[str, float]] = None,
        action_threshold: float = 0.5,
        learning_rate: float = 0.01,
    ) -> None:
        self.dimension_weights = dict(dimension_weights or self.DEFAULT_WEIGHTS)
        self.action_threshold = action_threshold
        self.learning_rate = learning_rate

        # Platt scaling parameters: P(y=1|f) = 1 / (1 + exp(A*f + B))
        self._platt_a: float = -1.0
        self._platt_b: float = 0.0

        # Feedback history: (predicted_score, was_correct)
        self._feedback_history: List[Tuple[float, bool]] = []

    # ------------------------------------------------------------------
    # Platt scaling
    # ------------------------------------------------------------------

    def _platt_scale(self, raw_score: float) -> float:
        """Map a raw score through the Platt sigmoid.

        Clamps the logit to [-20, 20] for numerical stability.
        """
        logit = self._platt_a * raw_score + self._platt_b
        logit = max(-20.0, min(20.0, logit))
        return 1.0 / (1.0 + math.exp(logit))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_action(
        self,
        action: str,
        context: Dict[str, Any],
        features: Dict[str, float],
    ) -> ReflectionResult:
        """Score a proposed *action* given *context* and feature scores.

        The raw score is the weighted average of the supplied feature
        dimensions.  Confidence is derived via a steep sigmoid so that
        scores near 0.5 map to ~0.5 and extremes saturate.  The
        calibrated confidence then passes through Platt scaling.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        reasoning: Dict[str, Any] = {}

        for dim, weight in self.dimension_weights.items():
            if dim in features:
                val = features[dim]
                weighted_sum += weight * val
                total_weight += weight
                reasoning[dim] = val

        raw_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Steep sigmoid centred at 0.5
        conf_logit = -5.0 * (raw_score - 0.5)
        conf_logit = max(-20.0, min(20.0, conf_logit))
        confidence = 1.0 / (1.0 + math.exp(conf_logit))

        calibrated_confidence = self._platt_scale(confidence)

        return ReflectionResult(
            action=action,
            raw_score=raw_score,
            confidence=confidence,
            calibrated_confidence=calibrated_confidence,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def should_act(self, result: ReflectionResult) -> bool:
        """Return ``True`` if the calibrated confidence meets the threshold."""
        return result.calibrated_confidence >= self.action_threshold

    # ------------------------------------------------------------------
    # Online calibration
    # ------------------------------------------------------------------

    def record_feedback(
        self, predicted_score: float, was_correct: bool
    ) -> None:
        """Update Platt parameters via online gradient descent.

        Uses logistic loss:
            error = p - target
            A -= lr * error * f
            B -= lr * error

        where *p* is the current Platt-scaled probability and *f* is the
        predicted score used as the input feature.
        """
        self._feedback_history.append((predicted_score, was_correct))

        target = 1.0 if was_correct else 0.0
        p = self._platt_scale(predicted_score)
        error = p - target

        self._platt_a -= self.learning_rate * error * predicted_score
        self._platt_b -= self.learning_rate * error

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Return a summary of calibration state."""
        total = len(self._feedback_history)
        correct = sum(1 for _, c in self._feedback_history if c)
        accuracy = correct / total if total > 0 else 0.0

        return {
            "total_feedback": total,
            "accuracy": accuracy,
            "platt_a": self._platt_a,
            "platt_b": self._platt_b,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> str:
        """Serialize the reflection state to a JSON string."""
        return json.dumps({
            "dimension_weights": self.dimension_weights,
            "action_threshold": self.action_threshold,
            "learning_rate": self.learning_rate,
            "platt_a": self._platt_a,
            "platt_b": self._platt_b,
            "feedback_history": [
                {"score": s, "correct": c}
                for s, c in self._feedback_history
            ],
        })

    @classmethod
    def deserialize(cls, data: str) -> "SelfReflection":
        """Reconstruct a ``SelfReflection`` instance from a JSON string."""
        obj = json.loads(data)
        sr = cls(
            dimension_weights=obj["dimension_weights"],
            action_threshold=obj["action_threshold"],
            learning_rate=obj["learning_rate"],
        )
        sr._platt_a = obj["platt_a"]
        sr._platt_b = obj["platt_b"]
        sr._feedback_history = [
            (entry["score"], entry["correct"])
            for entry in obj["feedback_history"]
        ]
        return sr
