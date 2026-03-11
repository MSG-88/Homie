"""Brain orchestrator — delegates to the Cognitive Architecture for intelligent responses.

The orchestrator is the public API used by cli.py, daemon.py, and overlay.py.
It delegates all reasoning to CognitiveArchitecture, which runs the full
6-stage pipeline: Perceive → Classify → Retrieve → Reason → Reflect → Adapt.

For backward compatibility, _build_optimized_prompt() is retained as a fallback
when cognitive architecture components are not fully wired.
"""
from __future__ import annotations

from typing import Any, Iterator, Optional

from homie_core.memory.working import WorkingMemory
from homie_core.memory.episodic import EpisodicMemory
from homie_core.memory.semantic import SemanticMemory
from homie_core.brain.cognitive_arch import CognitiveArchitecture


# Rough token estimate: ~4 chars per token
_CHARS_PER_TOKEN = 4

# Maximum prompt budget in characters (fallback mode)
_MAX_PROMPT_CHARS = 3000


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

        # Wire up the cognitive architecture
        self._cognitive = CognitiveArchitecture(
            model_engine=model_engine,
            working_memory=working_memory,
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
            system_prompt=self._system_prompt,
        )

    def process(self, user_input: str) -> str:
        """Full cognitive pipeline — blocking generate."""
        return self._cognitive.process(user_input)

    def process_stream(self, user_input: str) -> Iterator[str]:
        """Full cognitive pipeline — streaming for instant first-token."""
        return self._cognitive.process_stream(user_input)

    def _build_optimized_prompt(self, user_input: str) -> str:
        """Fallback prompt builder — used when cognitive arch isn't available.

        Priority order (highest first):
        1. System prompt + user query (always included)
        2. Current context (active window)
        3. Last 2 conversation turns
        4. Top 3 relevant facts
        5. Top 1 relevant episode
        """
        parts = [self._system_prompt]
        budget = _MAX_PROMPT_CHARS - len(self._system_prompt) - len(user_input) - 50

        active = self._wm.get("active_window")
        if active and budget > 100:
            ctx_line = f"\nContext: User is in {active}"
            parts.append(ctx_line)
            budget -= len(ctx_line)

        conversation = self._wm.get_conversation()
        if len(conversation) > 1 and budget > 200:
            recent = conversation[-4:]
            conv_lines = []
            for m in recent[:-1]:
                line = f"{m['role']}: {m['content'][:150]}"
                conv_lines.append(line)
            conv_text = "\n".join(conv_lines)
            if len(conv_text) <= budget:
                parts.append(f"\nRecent:\n{conv_text}")
                budget -= len(conv_text) + 10

        if self._sm and budget > 100:
            try:
                facts = self._sm.get_facts(min_confidence=0.6)
                for f in facts[:3]:
                    fact_text = f["fact"][:100]
                    if len(fact_text) + 5 <= budget:
                        parts.append(f"- {fact_text}")
                        budget -= len(fact_text) + 5
            except Exception:
                pass

        if self._em and budget > 100:
            try:
                episodes = self._em.recall(user_input, n=1)
                if episodes:
                    ep = episodes[0]["summary"][:150]
                    parts.append(f"\nRelated: {ep}")
                    budget -= len(ep) + 12
            except Exception:
                pass

        parts.append(f"\nUser: {user_input}\nAssistant:")
        return "\n".join(parts)

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt
        self._cognitive.set_system_prompt(prompt)
