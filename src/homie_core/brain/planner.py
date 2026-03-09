from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field


class PlannedAction(BaseModel):
    action: str  # "respond", "query_memory", "run_plugin", "suggest", "teach", "execute"
    target: str = ""  # plugin name, memory query, etc.
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.5


class ActionPlanner:
    SUPPORTED_ACTIONS = ["respond", "query_memory", "run_plugin", "suggest", "teach", "execute"]

    def __init__(self, model_engine):
        self._engine = model_engine

    def plan(self, user_input: str, context: dict | None = None) -> PlannedAction:
        ctx_str = json.dumps(context or {}, default=str)[:500]
        prompt = f"""Analyze the user's input and decide what action to take. Return a JSON object with: action, target, params, reason, confidence.

Supported actions: {', '.join(self.SUPPORTED_ACTIONS)}
- "respond": Just reply to the user conversationally
- "query_memory": Look up something in memory (target = query string)
- "run_plugin": Execute a plugin action (target = plugin name, params = action params)
- "suggest": Proactively suggest something
- "teach": User is teaching you something (params.fact = the fact)
- "execute": Run a system command or automation

Context: {ctx_str}
User input: {user_input}

Return ONLY valid JSON, nothing else."""

        response = self._engine.generate(prompt, max_tokens=300, temperature=0.2)
        return self._parse_action(response, user_input)

    def _parse_action(self, response: str, fallback_input: str) -> PlannedAction:
        try:
            # Try to find JSON in response
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            action = data.get("action", "respond")
            if action not in self.SUPPORTED_ACTIONS:
                action = "respond"
            return PlannedAction(
                action=action,
                target=data.get("target", ""),
                params=data.get("params", {}),
                reason=data.get("reason", ""),
                confidence=float(data.get("confidence", 0.5)),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return PlannedAction(action="respond", reason="Failed to parse action plan")
