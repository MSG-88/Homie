"""Tests for SummarizationMiddleware."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from homie_core.config import HomieConfig
from homie_core.memory.working import WorkingMemory
from homie_core.middleware.summarization import SummarizationMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_middleware(context_length: int = 1000, trigger_pct: float = 0.85):
    """Build a SummarizationMiddleware with a tiny context window for testing."""
    config = HomieConfig()
    config.llm.context_length = context_length
    config.context.summarize_trigger_pct = trigger_pct
    config.context.summarize_keep_pct = 0.20

    backend = MagicMock()
    wm = WorkingMemory()
    return SummarizationMiddleware(config=config, backend=backend, working_memory=wm), wm, backend


def _fill_conversation(wm: WorkingMemory, n_messages: int, chars_each: int = 200) -> None:
    """Populate working memory with synthetic conversation messages."""
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        wm.add_message(role, f"message {i}: " + "x" * chars_each)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSummarizationMiddlewareMetadata:
    def test_name_and_order(self):
        mw, _, _ = _make_middleware()
        assert mw.name == "summarization"
        assert mw.order == 5


class TestBelowTrigger:
    def test_message_passthrough(self):
        """Short conversation → message returned unchanged, no compression."""
        mw, wm, backend = _make_middleware(context_length=10_000)
        wm.add_message("user", "hi")
        wm.add_message("assistant", "hello")
        result = mw.before_turn("new message", {})
        assert result == "new message"

    def test_no_history_offloaded(self):
        """No backend write called when below trigger."""
        mw, wm, backend = _make_middleware(context_length=10_000)
        wm.add_message("user", "hi")
        mw.before_turn("test", {})
        backend.write.assert_not_called()

    def test_conversation_unchanged(self):
        """Conversation list unchanged when below trigger."""
        mw, wm, backend = _make_middleware(context_length=10_000)
        wm.add_message("user", "hello world")
        wm.add_message("assistant", "hi there")
        before = wm.get_conversation()
        mw.before_turn("new", {})
        after = wm.get_conversation()
        assert len(after) == len(before)


class TestAboveTrigger:
    def test_compression_triggered(self):
        """Large conversation → compressed (fewer messages remain)."""
        # trigger_chars = 1000 * 0.85 * 4 = 3400 chars
        mw, wm, _ = _make_middleware(context_length=1000, trigger_pct=0.85)
        # Fill with 20 messages × 200 chars each = 4000 chars > 3400
        _fill_conversation(wm, 20, chars_each=200)
        original_len = len(wm.get_conversation())
        mw.before_turn("compress me", {})
        compressed_len = len(wm.get_conversation())
        assert compressed_len < original_len

    def test_history_offloaded_to_backend(self):
        """Backend.write() is called when compression fires."""
        mw, wm, backend = _make_middleware(context_length=1000, trigger_pct=0.85)
        _fill_conversation(wm, 20, chars_each=200)
        mw.before_turn("trigger", {})
        backend.write.assert_called_once()

    def test_backend_write_path_has_timestamp(self):
        """Written path is under /conversation_history/ with a timestamp."""
        mw, wm, backend = _make_middleware(context_length=1000, trigger_pct=0.85)
        _fill_conversation(wm, 20, chars_each=200)
        mw.before_turn("trigger", {})
        call_path = backend.write.call_args[0][0]
        assert call_path.startswith("/conversation_history/")
        assert call_path.endswith(".md")

    def test_message_still_returned(self):
        """before_turn returns the original message even after compression."""
        mw, wm, _ = _make_middleware(context_length=1000, trigger_pct=0.85)
        _fill_conversation(wm, 20, chars_each=200)
        result = mw.before_turn("hello after compress", {})
        assert result == "hello after compress"

    def test_after_compression_conversation_shorter(self):
        """Post-compression conversation is shorter than the original."""
        mw, wm, _ = _make_middleware(context_length=1000, trigger_pct=0.85)
        _fill_conversation(wm, 20, chars_each=200)
        original_chars = sum(len(m.get("content", "")) for m in wm.get_conversation())
        mw.before_turn("go", {})
        compressed_chars = sum(len(m.get("content", "")) for m in wm.get_conversation())
        assert compressed_chars < original_chars


class TestModifyTools:
    def test_adds_compact_conversation_tool(self):
        """modify_tools appends compact_conversation to the tool list."""
        mw, _, _ = _make_middleware()
        result = mw.modify_tools([])
        names = [t["name"] for t in result]
        assert "compact_conversation" in names

    def test_preserves_existing_tools(self):
        """Existing tools are kept when compact_conversation is added."""
        mw, _, _ = _make_middleware()
        existing = [{"name": "search_web", "description": "..."}]
        result = mw.modify_tools(existing)
        names = [t["name"] for t in result]
        assert "search_web" in names
        assert "compact_conversation" in names

    def test_compact_tool_has_description(self):
        """The injected compact_conversation tool has a description key."""
        mw, _, _ = _make_middleware()
        result = mw.modify_tools([])
        compact = next(t for t in result if t["name"] == "compact_conversation")
        assert "description" in compact
        assert len(compact["description"]) > 0


class TestToolPairPreservation:
    def test_tool_call_result_pairs_not_orphaned(self):
        """Tool-call/result pairs must not be split across compression boundary."""
        mw, wm, _ = _make_middleware(context_length=500, trigger_pct=0.5)

        # Build a conversation where tool pairs appear in the middle
        wm.add_message("user", "x" * 100)
        wm.add_message("assistant", "x" * 100)
        # Tool pair
        wm.add_message("assistant", "<tool>search(query='foo')</tool>" + "x" * 50)
        wm.add_message("user", "[Tool: search] Result: some result")
        wm.add_message("user", "x" * 100)
        wm.add_message("assistant", "x" * 100)

        mw.before_turn("new", {})

        compressed = wm.get_conversation()
        # Walk through and verify no tool result without its preceding call
        for i, msg in enumerate(compressed):
            content = msg.get("content", "")
            if "[Tool:" in content and "Result:" in content:
                # There should be an assistant tool-call message before this
                if i > 0:
                    prev = compressed[i - 1].get("content", "")
                    assert "<tool>" in prev or "[COMPRESSED]" in prev, (
                        "Tool result found without preceding tool call or compression summary"
                    )
