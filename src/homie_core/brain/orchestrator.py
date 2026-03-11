from __future__ import annotations

from typing import Any, Iterator, Optional

from homie_core.memory.working import WorkingMemory
from homie_core.memory.episodic import EpisodicMemory
from homie_core.memory.semantic import SemanticMemory


# Rough token estimate: ~4 chars per token
_CHARS_PER_TOKEN = 4

# Maximum prompt budget in characters (leaves room for response)
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

    def process(self, user_input: str) -> str:
        """Blocking generate — use process_stream() for real-time output."""
        self._wm.add_message("user", user_input)
        prompt = self._build_optimized_prompt(user_input)
        response = self._engine.generate(prompt, max_tokens=512, temperature=0.7)
        self._wm.add_message("assistant", response)
        return response

    def process_stream(self, user_input: str) -> Iterator[str]:
        """Stream tokens as they're generated — first token arrives fast."""
        self._wm.add_message("user", user_input)
        prompt = self._build_optimized_prompt(user_input)
        chunks = []
        for token in self._engine.stream(prompt, max_tokens=512, temperature=0.7):
            chunks.append(token)
            yield token
        full_response = "".join(chunks)
        self._wm.add_message("assistant", full_response)

    def _build_optimized_prompt(self, user_input: str) -> str:
        """Build a compact prompt that fits within token budget.

        Priority order (highest first):
        1. System prompt + user query (always included)
        2. Current context (active window)
        3. Last 2 conversation turns
        4. Top 3 relevant facts
        5. Top 1 relevant episode
        """
        # Start with mandatory parts
        parts = [self._system_prompt]
        budget = _MAX_PROMPT_CHARS - len(self._system_prompt) - len(user_input) - 50

        # Priority 1: Current context (cheap, very useful)
        active = self._wm.get("active_window")
        if active and budget > 100:
            ctx_line = f"\nContext: User is in {active}"
            parts.append(ctx_line)
            budget -= len(ctx_line)

        # Priority 2: Recent conversation (2 turns max, truncated)
        conversation = self._wm.get_conversation()
        if len(conversation) > 1 and budget > 200:
            recent = conversation[-4:]  # last 2 exchanges
            conv_lines = []
            for m in recent[:-1]:  # exclude current message
                line = f"{m['role']}: {m['content'][:150]}"
                conv_lines.append(line)
            conv_text = "\n".join(conv_lines)
            if len(conv_text) <= budget:
                parts.append(f"\nRecent:\n{conv_text}")
                budget -= len(conv_text) + 10

        # Priority 3: Relevant facts (top 3, short)
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

        # Priority 4: Relevant episode (1 only, truncated)
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
