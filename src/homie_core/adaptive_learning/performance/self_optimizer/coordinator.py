"""SelfOptimizer — coordinates prompt optimization, model tuning, and pipeline gating."""

import logging
from typing import Any, Optional

from .model_tuner import ModelTuner
from .pipeline_gate import PipelineGate
from .profiler import OptimizationProfiler
from .prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


class SelfOptimizer:
    """Coordinates all performance self-optimization."""

    def __init__(
        self,
        storage,
        hardware_fingerprint: str,
        promotion_threshold: int = 3,
    ) -> None:
        self.profiler = OptimizationProfiler(storage=storage, hardware_fingerprint=hardware_fingerprint)
        self.prompt_optimizer = PromptOptimizer()
        self.model_tuner = ModelTuner(profiler=self.profiler)
        self.pipeline_gate = PipelineGate(promotion_threshold=promotion_threshold)

    def optimize_query(
        self,
        complexity: str,
        query_hint: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get optimized parameters for a query."""
        # Apply pipeline gating
        effective = self.pipeline_gate.apply(complexity)

        # Set prompt optimizer's complexity
        self.prompt_optimizer.set_complexity(effective)

        # Get model parameters
        params = self.model_tuner.select_parameters(effective, query_hint=query_hint)

        return {
            "effective_complexity": effective,
            "original_complexity": complexity,
            **params,
        }

    def record_result(
        self,
        query_type: str,
        complexity: str,
        temperature: float,
        max_tokens: int,
        response_tokens: float,
        latency_ms: float,
    ) -> None:
        """Record an inference result for profile learning."""
        self.model_tuner.record_result(
            query_type=query_type,
            temperature=temperature,
            max_tokens=max_tokens,
            response_tokens=response_tokens,
            latency_ms=latency_ms,
        )

    def record_clarification(self, tier: str) -> None:
        """Record that a gated query needed clarification — may promote tier."""
        self.pipeline_gate.record_clarification(tier)

    def get_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        return {
            "gate": self.pipeline_gate.get_stats(),
        }
