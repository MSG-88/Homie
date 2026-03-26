"""Tests for conversation memory persistence."""
import pytest
import time
from homie_core.memory.conversation_store import ConversationStore, ConversationSummary


class TestConversationStore:
    def test_save_and_get_turns(self, tmp_path):
        store = ConversationStore(db_path=tmp_path / "conv.db")
        store.save_turn("s1", "user", "Hello Homie")
        store.save_turn("s1", "assistant", "Hi there!")
        store.save_turn("s1", "user", "What time is it?")
        turns = store.get_session_turns("s1")
        assert len(turns) == 3
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Hello Homie"
        store.close()

    def test_save_and_get_summary(self, tmp_path):
        store = ConversationStore(db_path=tmp_path / "conv.db")
        summary = ConversationSummary(
            session_id="s1", timestamp=time.time(), duration_minutes=5.0,
            turn_count=10, summary="Discussed project status and emails",
            key_topics=["email", "project"], facts_learned=["prefers bullet points"],
            action_items=["review PR"], user_mood="positive",
        )
        store.save_summary(summary)
        results = store.get_recent_summaries(5)
        assert len(results) == 1
        assert results[0].summary == "Discussed project status and emails"
        assert "email" in results[0].key_topics
        store.close()

    def test_search_conversations(self, tmp_path):
        store = ConversationStore(db_path=tmp_path / "conv.db")
        store.save_summary(ConversationSummary("s1", time.time(), 5, 10, "Talked about emails and git", ["email", "git"], [], [], ""))
        store.save_summary(ConversationSummary("s2", time.time(), 3, 6, "Discussed LinkedIn strategy", ["linkedin"], [], [], ""))
        results = store.search_conversations("email")
        assert len(results) == 1
        assert "email" in results[0].summary
        store.close()

    def test_context_for_prompt(self, tmp_path):
        store = ConversationStore(db_path=tmp_path / "conv.db")
        store.save_summary(ConversationSummary("s1", time.time(), 5, 10, "Reviewed inbox and drafted standup", ["email"], [], ["review PR"], ""))
        ctx = store.get_context_for_prompt()
        assert "Recent conversation" in ctx
        assert "Reviewed inbox" in ctx
        assert "review PR" in ctx
        store.close()

    def test_empty_context(self, tmp_path):
        store = ConversationStore(db_path=tmp_path / "conv.db")
        ctx = store.get_context_for_prompt()
        assert ctx == ""
        store.close()

    def test_generate_summary_heuristic(self, tmp_path):
        store = ConversationStore(db_path=tmp_path / "conv.db")
        store.save_turn("s1", "user", "Check my emails")
        store.save_turn("s1", "assistant", "You have 10 unread")
        store.save_turn("s1", "user", "Thanks, that was helpful!")
        summary = store.generate_summary_from_turns("s1")
        assert summary.turn_count == 3
        assert "email" in summary.key_topics
        assert summary.user_mood == "grateful"
        store.close()

    def test_generate_summary_with_llm(self, tmp_path):
        store = ConversationStore(db_path=tmp_path / "conv.db")
        store.save_turn("s1", "user", "What am I working on?")
        store.save_turn("s1", "assistant", "You are working on the Homie project")

        def mock_inference(**kwargs):
            return "Summary: Discussed current project status.\nFacts: user works on Homie\nActions: continue development"

        summary = store.generate_summary_from_turns("s1", inference_fn=mock_inference)
        assert "project" in summary.summary.lower()
        assert len(summary.facts_learned) > 0
        store.close()

    def test_stats(self, tmp_path):
        store = ConversationStore(db_path=tmp_path / "conv.db")
        store.save_turn("s1", "user", "Hello")
        store.save_turn("s1", "assistant", "Hi")
        store.save_summary(ConversationSummary("s1", time.time(), 1, 2, "Greeting", [], [], [], ""))
        stats = store.get_stats()
        assert stats["total_sessions"] == 1
        assert stats["total_turns"] == 2
        store.close()

    def test_multiple_sessions(self, tmp_path):
        store = ConversationStore(db_path=tmp_path / "conv.db")
        store.save_summary(ConversationSummary("s1", time.time() - 3600, 5, 10, "Morning session", [], [], [], ""))
        store.save_summary(ConversationSummary("s2", time.time(), 3, 6, "Afternoon session", [], [], [], ""))
        results = store.get_recent_summaries(5)
        assert len(results) == 2
        assert results[0].summary == "Afternoon session"  # Most recent first
        store.close()
