from __future__ import annotations

import json
from typing import Any, Optional

from homie_core.memory.working import WorkingMemory
from homie_core.memory.episodic import EpisodicMemory
from homie_core.memory.semantic import SemanticMemory
from homie_core.utils import utc_now


class BrainOrchestrator:
    def __init__(
        self,
        model_engine,
        working_memory: WorkingMemory,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
    ):
        self._engine = model_engine
        self._wm = working_memory
        self._em = episodic_memory
        self._sm = semantic_memory
        self._system_prompt = "You are Homie, a helpful local AI assistant. Be concise and direct."

    def process(self, user_input: str) -> str:
        self._wm.add_message("user", user_input)
        context = self._build_context(user_input)
        prompt = self._build_prompt(user_input, context)
        response = self._engine.generate(prompt, max_tokens=1024, temperature=0.7)
        self._wm.add_message("assistant", response)
        return response

    def _build_context(self, query: str) -> dict[str, Any]:
        context = {"working": self._wm.snapshot()}

        if self._em:
            try:
                episodes = self._em.recall(query, n=3)
                context["relevant_episodes"] = [e["summary"] for e in episodes]
            except Exception:
                context["relevant_episodes"] = []

        if self._sm:
            try:
                facts = self._sm.get_facts(min_confidence=0.5)
                context["known_facts"] = [f["fact"] for f in facts[:10]]
                profiles = self._sm.get_all_profiles()
                context["user_profile"] = profiles
            except Exception:
                context["known_facts"] = []
                context["user_profile"] = {}

        return context

    def _build_prompt(self, user_input: str, context: dict) -> str:
        parts = [self._system_prompt]

        if context.get("known_facts"):
            parts.append(f"\nKnown facts about the user:\n- " + "\n- ".join(context["known_facts"]))

        if context.get("relevant_episodes"):
            parts.append(f"\nRelevant past interactions:\n- " + "\n- ".join(context["relevant_episodes"]))

        if context.get("working", {}).get("active_window"):
            parts.append(f"\nCurrent context: User is in {context['working'].get('active_window', 'unknown')}")

        conversation = self._wm.get_conversation()
        if len(conversation) > 1:
            recent = conversation[-6:]  # last 3 exchanges
            conv_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent[:-1])
            parts.append(f"\nRecent conversation:\n{conv_text}")

        parts.append(f"\nUser: {user_input}\nAssistant:")
        return "\n".join(parts)

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt
