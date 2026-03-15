"""Tests for ContextOverflowRecoveryMiddleware."""
from __future__ import annotations

import pytest

from homie_core.memory.working import WorkingMemory
from homie_core.middleware.context_overflow import ContextOverflowRecoveryMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_middleware():
    wm = WorkingMemory()
    return ContextOverflowRecoveryMiddleware(working_memory=wm), wm


def _fill_conversation(wm: WorkingMemory, n: int = 20, chars_each: int = 200):
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        wm.add_message(role, f"msg {i}: " + "y" * chars_each)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_name_and_order(self):
        mw, _ = _make_middleware()
        assert mw.name == "context_overflow_recovery"
        assert mw.order == 1


class TestNoFlag:
    def test_message_passthrough_without_flag(self):
        """No context_overflow flag → message returned unchanged."""
        mw, wm = _make_middleware()
        _fill_conversation(wm, 10)
        state = {}
        result = mw.before_turn("hello", state)
        assert result == "hello"

    def test_conversation_unchanged_without_flag(self):
        """Conversation is not touched when flag is absent."""
        mw, wm = _make_middleware()
        _fill_conversation(wm, 10)
        original_len = len(wm.get_conversation())
        mw.before_turn("hello", {})
        assert len(wm.get_conversation()) == original_len

    def test_false_flag_treated_as_no_flag(self):
        """state['context_overflow'] = False → no compression."""
        mw, wm = _make_middleware()
        _fill_conversation(wm, 10)
        original_len = len(wm.get_conversation())
        state = {"context_overflow": False}
        mw.before_turn("hello", state)
        assert len(wm.get_conversation()) == original_len


class TestWithFlag:
    def test_compression_triggered_by_flag(self):
        """state['context_overflow'] = True → conversation is compressed."""
        mw, wm = _make_middleware()
        _fill_conversation(wm, 20, chars_each=200)
        original_len = len(wm.get_conversation())
        state = {"context_overflow": True}
        mw.before_turn("retry", state)
        compressed_len = len(wm.get_conversation())
        assert compressed_len < original_len

    def test_flag_consumed_after_processing(self):
        """Flag is removed from state dict after overflow handling."""
        mw, wm = _make_middleware()
        _fill_conversation(wm, 20)
        state = {"context_overflow": True}
        mw.before_turn("retry", state)
        assert "context_overflow" not in state

    def test_flag_consumed_even_when_conversation_short(self):
        """Flag is popped from state regardless of conversation length."""
        mw, wm = _make_middleware()
        wm.add_message("user", "hello")
        state = {"context_overflow": True}
        mw.before_turn("retry", state)
        assert "context_overflow" not in state

    def test_message_returned_after_compression(self):
        """before_turn returns the message string after compressing."""
        mw, wm = _make_middleware()
        _fill_conversation(wm, 20)
        state = {"context_overflow": True}
        result = mw.before_turn("retry message", state)
        assert result == "retry message"

    def test_after_compression_conversation_shorter_in_chars(self):
        """Compressed conversation has fewer total chars than original."""
        mw, wm = _make_middleware()
        _fill_conversation(wm, 20, chars_each=200)
        original_chars = sum(len(m.get("content", "")) for m in wm.get_conversation())
        state = {"context_overflow": True}
        mw.before_turn("retry", state)
        compressed_chars = sum(len(m.get("content", "")) for m in wm.get_conversation())
        assert compressed_chars < original_chars

    def test_other_state_keys_preserved(self):
        """Other keys in state dict are not removed by overflow middleware."""
        mw, wm = _make_middleware()
        _fill_conversation(wm, 20)
        state = {"context_overflow": True, "user_id": "abc", "session": 42}
        mw.before_turn("retry", state)
        assert state["user_id"] == "abc"
        assert state["session"] == 42
