"""ContextOverflowRecoveryMiddleware — emergency compressor for context-exceeded errors.

When the orchestrator catches a context-overflow error from the model engine it
sets ``state["context_overflow"] = True`` before re-running the middleware
pipeline.  This middleware detects that flag, compresses the conversation
aggressively (protect only 1 head + 3 tail messages), clears the flag, and
lets processing continue with the now-smaller context.
"""
from __future__ import annotations

from homie_core.middleware.base import HomieMiddleware
from homie_core.brain.context_compressor import ContextCompressor
from homie_core.memory.working import WorkingMemory


class ContextOverflowRecoveryMiddleware(HomieMiddleware):
    """Emergency conversation compressor triggered by context-exceeded errors."""

    name = "context_overflow_recovery"
    order = 1

    def __init__(self, working_memory: WorkingMemory) -> None:
        self._wm = working_memory

    def before_turn(self, message: str, state: dict) -> str:
        """If the context_overflow flag is set, compress and clear the flag."""
        if state.pop("context_overflow", False):
            conversation = self._wm.get_conversation()
            compressor = ContextCompressor(
                threshold_chars=0,      # always compress when called
                protect_first_n=1,
                protect_last_n=3,
                summary_target_chars=400,
            )
            compressed = compressor.compress(conversation)
            self._wm._conversation = type(self._wm._conversation)(
                compressed, maxlen=self._wm._conversation.maxlen
            )
        return message
