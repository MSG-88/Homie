from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from homie_core.memory.working import WorkingMemory


class MemoryConsolidator:
    def __init__(self, model_engine):
        self._engine = model_engine

    def create_session_digest(self, working: WorkingMemory) -> dict[str, Any]:
        conversation = working.get_conversation()
        context = working.snapshot()
        if not conversation:
            return {"summary": "", "mood": None, "key_events": []}
        conv_text = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
        ctx_text = ", ".join(f"{k}={v}" for k, v in context.items())
        prompt = f"""Summarize this interaction session in 1-2 sentences. Include what the user was doing, their apparent mood, and the outcome.

Context: {ctx_text}

Conversation:
{conv_text}

Respond with ONLY the summary, nothing else."""
        summary = self._engine.generate(prompt, max_tokens=200, temperature=0.3)
        return {"summary": summary.strip(), "mood": None, "key_events": [], "context": context}

    def extract_facts(self, episode_summary: str) -> list[str]:
        prompt = f"""From this session summary, extract any new facts learned about the user. Return a JSON array of strings. If no facts, return [].

Summary: {episode_summary}

Respond with ONLY the JSON array."""
        response = self._engine.generate(prompt, max_tokens=300, temperature=0.2)
        try:
            facts = json.loads(response.strip())
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, str)]
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse fact extraction JSON from model response: %s", e)
        return []
