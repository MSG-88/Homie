"""SummarizationMiddleware — auto-compresses conversation history at 85% context fill.

When the total character count of the conversation exceeds 85% of the model's
context window (approximated as 4 chars per token), this middleware:

1. Offloads the full conversation to the backend as a Markdown archive.
2. Compresses it in-place using ContextCompressor, keeping the most recent
   messages verbatim and replacing the middle with an extractive summary.

It also injects a ``compact_conversation`` tool so the assistant can trigger
manual compression on demand.
"""
from __future__ import annotations

import datetime

from homie_core.middleware.base import HomieMiddleware
from homie_core.brain.context_compressor import ContextCompressor
from homie_core.backend.protocol import BackendProtocol
from homie_core.memory.working import WorkingMemory
from homie_core.config import HomieConfig


class SummarizationMiddleware(HomieMiddleware):
    """Auto-compress conversation history when approaching context window limit."""

    name = "summarization"
    order = 5

    def __init__(
        self,
        config: HomieConfig,
        backend: BackendProtocol,
        working_memory: WorkingMemory,
    ) -> None:
        self._config = config
        self._backend = backend
        self._wm = working_memory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _trigger_chars(self) -> int:
        """Character count that triggers compression (≈ 85% of context window)."""
        return int(
            self._config.llm.context_length
            * self._config.context.summarize_trigger_pct
            * 4  # ~4 chars per token
        )

    def _make_compressor(self, conversation_length: int) -> ContextCompressor:
        """Build a ContextCompressor tuned to the current conversation size."""
        keep_count = max(3, int(conversation_length * self._config.context.summarize_keep_pct))
        return ContextCompressor(
            threshold_chars=0,        # threshold handled externally
            protect_first_n=2,
            protect_last_n=keep_count,
            summary_target_chars=800,
        )

    def _offload_history(self, conversation: list[dict]) -> None:
        """Write the current conversation to the backend as a Markdown archive."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        lines = []
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"## {role}\n\n{content}\n")
        self._backend.write(f"/conversation_history/{timestamp}.md", "\n".join(lines))

    # ------------------------------------------------------------------
    # HomieMiddleware overrides
    # ------------------------------------------------------------------

    def before_turn(self, message: str, state: dict) -> str:
        """Compress conversation if total chars exceed the trigger threshold."""
        conversation = self._wm.get_conversation()
        total_chars = sum(len(m.get("content", "")) for m in conversation)
        if total_chars > self._trigger_chars:
            self._offload_history(conversation)
            compressor = self._make_compressor(len(conversation))
            compressed = compressor.compress(conversation)
            self._wm._conversation = type(self._wm._conversation)(
                compressed, maxlen=self._wm._conversation.maxlen
            )
        return message

    def modify_tools(self, tools: list[dict]) -> list[dict]:
        """Inject the compact_conversation tool into every tool list."""
        compact_tool = {
            "name": "compact_conversation",
            "description": (
                "Manually compact conversation history to save context space. "
                "Only works when conversation is large enough."
            ),
        }
        return tools + [compact_tool]
