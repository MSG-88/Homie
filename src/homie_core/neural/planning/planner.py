"""Planner — decomposes goals into ThoughtChains using LLM inference."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Callable, Optional

from .goal import ThoughtChain, ThoughtStep

logger = logging.getLogger(__name__)

# Prompt templates --------------------------------------------------------

CLASSIFY_PROMPT = """\
Classify the complexity of this goal. Reply with exactly one word: trivial, simple, moderate, or complex.

Goal: {goal}
Context keys: {context_keys}

Rules:
- trivial: can be answered directly with no steps (greetings, simple facts)
- simple: 1-2 sequential steps, single agent
- moderate: 3-5 steps, may involve multiple agents or light parallelism
- complex: 6+ steps, multiple agents, dependencies, hierarchical decomposition needed

Complexity:"""

PLAN_PROMPT = """\
You are Homie's planning engine. Decompose the following goal into a step-by-step plan.

Goal: {goal}
Strategy: {strategy}
Context: {context}

Return a JSON object with this exact structure:
{{
  "steps": [
    {{
      "id": "step-1",
      "reasoning": "why this step is needed",
      "action": "what to do",
      "expected_outcome": "what success looks like",
      "agent": "reasoning|research|action|validation",
      "dependencies": []
    }}
  ]
}}

Rules:
- For "direct" strategy: return exactly 1 step.
- For "linear" strategy: steps are sequential; each step depends on the previous.
- For "parallel" strategy: identify independent steps that can run concurrently (empty dependencies). Add a final aggregation step that depends on all parallel ones.
- For "hierarchical" strategy: decompose into phases, each phase may have parallel sub-steps. Use dependency IDs to express the DAG.
- Every step must have a unique id like "step-1", "step-2", etc.
- Valid agents: reasoning, research, action, validation.
- Keep it concise. No more than 10 steps.

JSON:"""


# Strategy selection ------------------------------------------------------

COMPLEXITY_TO_STRATEGY = {
    "trivial": "direct",
    "simple": "linear",
    "moderate": "parallel",
    "complex": "hierarchical",
}


class Planner:
    """Decomposes a goal string into a ThoughtChain via LLM inference."""

    def __init__(
        self,
        inference_fn: Callable[[str], str],
        default_strategy: str = "hierarchical",
    ):
        self._infer = inference_fn
        self._default_strategy = default_strategy

    # -- public API -------------------------------------------------------

    def plan(self, goal: str, context: Optional[dict] = None) -> ThoughtChain:
        """Create a ThoughtChain for *goal* using the appropriate strategy."""
        ctx = context or {}
        complexity = self.classify_goal_complexity(goal, ctx)
        strategy = COMPLEXITY_TO_STRATEGY.get(complexity, self._default_strategy)
        logger.info("Goal complexity=%s  strategy=%s", complexity, strategy)
        return self._build_chain(goal, strategy, ctx)

    def classify_goal_complexity(
        self, goal: str, context: Optional[dict] = None
    ) -> str:
        """Ask the LLM to classify goal complexity."""
        ctx = context or {}
        prompt = CLASSIFY_PROMPT.format(
            goal=goal, context_keys=", ".join(ctx.keys()) if ctx else "none"
        )
        raw = self._infer(prompt).strip().lower()
        # Robustly extract the classification word
        for level in ("trivial", "simple", "moderate", "complex"):
            if level in raw:
                return level
        logger.warning("Could not parse complexity from LLM: %r, defaulting to moderate", raw)
        return "moderate"

    # -- internals --------------------------------------------------------

    def _build_chain(self, goal: str, strategy: str, context: dict) -> ThoughtChain:
        prompt = PLAN_PROMPT.format(
            goal=goal,
            strategy=strategy,
            context=json.dumps(context, default=str) if context else "{}",
        )
        raw = self._infer(prompt)
        steps = self._parse_steps(raw)
        return ThoughtChain(goal=goal, steps=steps, current_step=0, status="thinking")

    @staticmethod
    def _parse_steps(raw: str) -> list[ThoughtStep]:
        """Parse the LLM JSON response into ThoughtStep objects."""
        # Try to find JSON in the response
        text = raw.strip()
        # Handle cases where LLM wraps JSON in markdown fences
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Last-ditch: find the first { ... } block
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                raise ValueError(f"Could not parse plan JSON from LLM output: {raw[:200]}")

        raw_steps = data.get("steps", [])
        steps: list[ThoughtStep] = []
        for s in raw_steps:
            steps.append(
                ThoughtStep(
                    id=s.get("id", f"step-{uuid.uuid4().hex[:6]}"),
                    reasoning=s.get("reasoning", ""),
                    action=s.get("action", ""),
                    expected_outcome=s.get("expected_outcome", ""),
                    agent=s.get("agent", "reasoning"),
                    dependencies=s.get("dependencies", []),
                )
            )
        if not steps:
            raise ValueError("LLM returned a plan with zero steps")
        return steps
