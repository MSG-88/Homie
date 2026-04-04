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
from homie_core.brain.tool_registry import ToolRegistry
from homie_core.rag.pipeline import RagPipeline
from homie_core.middleware import MiddlewareStack, HookRegistry


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
        tool_registry: Optional[ToolRegistry] = None,
        rag_pipeline: Optional[RagPipeline] = None,
        middleware_stack: Optional[MiddlewareStack] = None,
        knowledge_graph=None,
    ):
        self._engine = model_engine
        self._wm = working_memory
        self._em = episodic_memory
        self._sm = semantic_memory
        self._system_prompt = "You are Homie, a helpful local AI assistant. Be concise and direct."
        self._middleware = middleware_stack or MiddlewareStack()
        self._hooks = HookRegistry()

        # Wire up the cognitive architecture with tools + learning + RAG + knowledge graph
        self._cognitive = CognitiveArchitecture(
            model_engine=model_engine,
            working_memory=working_memory,
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
            system_prompt=self._system_prompt,
            tool_registry=tool_registry,
            rag_pipeline=rag_pipeline,
            hooks=self._hooks,
            knowledge_graph=knowledge_graph,
        )

    @property
    def hooks(self) -> HookRegistry:
        return self._hooks

    # ------------------------------------------------------------------
    # Context-overflow detection
    # ------------------------------------------------------------------

    def _is_context_overflow(self, error: Exception) -> bool:
        """Return True if *error* looks like a model context-length exceeded error."""
        msg = str(error).lower()
        if "too long" in msg:
            return True
        return "context" in msg and ("overflow" in msg or "length" in msg)

    # ------------------------------------------------------------------
    # Process methods (with overflow recovery)
    # ------------------------------------------------------------------

    def process(self, user_input: str, *, state: dict | None = None) -> str:
        """Full cognitive pipeline — blocking generate."""
        state = state or {}
        message = self._middleware.run_before_turn(user_input, state)
        if message is None:
            return ""
        try:
            response = self._cognitive.process(message)
        except Exception as e:
            if self._is_context_overflow(e):
                state["context_overflow"] = True
                message = self._middleware.run_before_turn(user_input, state)
                if message is None:
                    return ""
                response = self._cognitive.process(message)
            else:
                raise
        response = self._middleware.run_after_turn(response, state)
        return response

    def process_stream(self, user_input: str, *, state: dict | None = None) -> Iterator[str]:
        """Full cognitive pipeline — streaming for instant first-token."""
        state = state or {}
        message = self._middleware.run_before_turn(user_input, state)
        if message is None:
            return
        try:
            tokens = list(self._cognitive.process_stream(message))
        except Exception as e:
            if self._is_context_overflow(e):
                state["context_overflow"] = True
                message = self._middleware.run_before_turn(user_input, state)
                if message is None:
                    return
                tokens = list(self._cognitive.process_stream(message))
            else:
                raise
        response = "".join(tokens)
        response = self._middleware.run_after_turn(response, state)
        # Yield final response as single token to preserve streaming interface
        # (middleware may have modified it, so we re-yield the processed result)
        # But to preserve original streaming behaviour when no middleware modifies,
        # yield original tokens if response matches original join
        if response == "".join(tokens):
            yield from tokens
        else:
            yield response

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

    def consolidate_session(self, mood: Optional[str] = None) -> Optional[str]:
        """Consolidate session into episodic memory. Call at session end."""
        return self._cognitive.consolidate_session(mood=mood)
