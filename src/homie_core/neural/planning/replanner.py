"""Replanner — generates alternative plans when a step fails."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Callable

from .goal import ThoughtChain, ThoughtStep

logger = logging.getLogger(__name__)

REPLAN_PROMPT = """\
You are Homie's re-planning engine. A step in the current plan has failed.

Original goal: {goal}
Failed step: {failed_step}
Error: {error}

Completed steps so far:
{completed_summary}

Remaining steps (not yet attempted):
{remaining_summary}

Generate an alternative plan that achieves the same goal, starting from the current state.
Do NOT repeat already-completed steps. Build on their results.

Return a JSON object:
{{
  "steps": [
    {{
      "id": "step-1",
      "reasoning": "why this step",
      "action": "what to do",
      "expected_outcome": "success criteria",
      "agent": "reasoning|research|action|validation",
      "dependencies": []
    }}
  ]
}}

JSON:"""


class Replanner:
    """Generates alternative ThoughtChains when a step fails."""

    def __init__(
        self,
        inference_fn: Callable[[str], str],
        max_replans: int = 3,
    ):
        self._infer = inference_fn
        self._max_replans = max_replans
        self._replan_count: dict[str, int] = {}  # goal -> count

    @property
    def max_replans(self) -> int:
        return self._max_replans

    def can_replan(self, chain: ThoughtChain) -> bool:
        """Check whether we haven't exceeded the replan budget for this goal."""
        count = self._replan_count.get(chain.goal, 0)
        return count < self._max_replans

    def replan(
        self,
        chain: ThoughtChain,
        failed_step: ThoughtStep,
        error: str,
    ) -> ThoughtChain:
        """Generate a new ThoughtChain that works around the failure.

        Raises RuntimeError if the replan budget is exhausted.
        """
        if not self.can_replan(chain):
            raise RuntimeError(
                f"Replan budget exhausted for goal: {chain.goal!r} "
                f"(max {self._max_replans})"
            )

        self._replan_count[chain.goal] = self._replan_count.get(chain.goal, 0) + 1
        logger.info(
            "Re-planning goal=%r  attempt=%d/%d  failed_step=%s  error=%s",
            chain.goal,
            self._replan_count[chain.goal],
            self._max_replans,
            failed_step.id,
            error[:120],
        )

        completed = [s for s in chain.steps if s.status == "complete"]
        remaining = [
            s for s in chain.steps
            if s.status == "pending" and s.id != failed_step.id
        ]

        prompt = REPLAN_PROMPT.format(
            goal=chain.goal,
            failed_step=json.dumps(failed_step.to_dict()),
            error=error,
            completed_summary=self._summarize_steps(completed),
            remaining_summary=self._summarize_steps(remaining),
        )

        raw = self._infer(prompt)
        steps = self._parse_steps(raw, completed)
        new_chain = ThoughtChain(
            goal=chain.goal,
            steps=steps,
            current_step=0,
            status="replanning",
        )
        return new_chain

    def reset(self, goal: str) -> None:
        """Reset the replan counter for a goal (e.g., on full success)."""
        self._replan_count.pop(goal, None)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _summarize_steps(steps: list[ThoughtStep]) -> str:
        if not steps:
            return "(none)"
        lines = []
        for s in steps:
            result_str = json.dumps(s.result) if s.result else "n/a"
            lines.append(
                f"- [{s.id}] {s.action} (status={s.status}, result={result_str})"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_steps(
        raw: str, completed: list[ThoughtStep]
    ) -> list[ThoughtStep]:
        """Parse LLM output into ThoughtStep list, prepending completed steps."""
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                raise ValueError(
                    f"Could not parse replan JSON from LLM: {raw[:200]}"
                )

        # Keep completed steps as-is, then add new ones
        new_steps: list[ThoughtStep] = list(completed)
        for s in data.get("steps", []):
            new_steps.append(
                ThoughtStep(
                    id=s.get("id", f"step-{uuid.uuid4().hex[:6]}"),
                    reasoning=s.get("reasoning", ""),
                    action=s.get("action", ""),
                    expected_outcome=s.get("expected_outcome", ""),
                    agent=s.get("agent", "reasoning"),
                    dependencies=s.get("dependencies", []),
                )
            )
        if len(new_steps) == len(completed):
            raise ValueError("Re-plan produced zero new steps")
        return new_steps
