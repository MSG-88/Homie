"""Unit tests for all memory layers.

Covers:
- WorkingMemory: update/get/snapshot, conversation buffer, maxlen, clear
- EpisodicMemory: record, recall, delete (mocked DB + VectorStore)
- SemanticMemory: learn, reinforce, forget, profile (real in-memory DB)
- ForgettingCurve: relevance calculation, decay_all archival
- MemoryConsolidator: digest creation, fact extraction (mocked engine)
"""
from __future__ import annotations

import json
import math
import threading
from datetime import timedelta
from unittest.mock import MagicMock, call

import pytest

from homie_core.memory.working import WorkingMemory
from homie_core.memory.episodic import EpisodicMemory
from homie_core.memory.semantic import SemanticMemory
from homie_core.memory.forgetting import ForgettingCurve
from homie_core.memory.consolidator import MemoryConsolidator
from homie_core.utils import utc_now


# ---------------------------------------------------------------------------
# Fixtures for real DB-backed memories
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    from homie_core.storage.database import Database
    d = Database(tmp_path / "mem_test.db")
    d.initialize()
    return d


@pytest.fixture
def vector_store(tmp_path):
    from homie_core.storage.vectors import VectorStore
    vs = VectorStore(tmp_path / "chroma")
    vs.initialize()
    return vs


@pytest.fixture
def semantic(db):
    return SemanticMemory(db=db)


@pytest.fixture
def episodic(db, vector_store):
    return EpisodicMemory(db=db, vector_store=vector_store)


@pytest.fixture
def forgetting(db):
    return ForgettingCurve(db=db, decay_rate=0.1)


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------

class TestWorkingMemory:
    def test_update_and_get(self):
        wm = WorkingMemory()
        wm.update("app", "VS Code")
        assert wm.get("app") == "VS Code"

    def test_get_missing_key_returns_default(self):
        wm = WorkingMemory()
        assert wm.get("missing") is None
        assert wm.get("missing", "fallback") == "fallback"

    def test_snapshot_returns_copy(self):
        wm = WorkingMemory()
        wm.update("k", "v")
        snap = wm.snapshot()
        snap["k"] = "mutated"
        assert wm.get("k") == "v"

    def test_multiple_keys(self):
        wm = WorkingMemory()
        wm.update("a", 1)
        wm.update("b", 2)
        snap = wm.snapshot()
        assert snap == {"a": 1, "b": 2}

    def test_overwrite_key(self):
        wm = WorkingMemory()
        wm.update("k", "old")
        wm.update("k", "new")
        assert wm.get("k") == "new"

    def test_add_message_basic(self):
        wm = WorkingMemory()
        wm.add_message("user", "hello")
        msgs = wm.get_conversation()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    def test_message_has_timestamp(self):
        wm = WorkingMemory()
        wm.add_message("assistant", "hi")
        msgs = wm.get_conversation()
        assert "timestamp" in msgs[0]

    def test_conversation_multiple_turns(self):
        wm = WorkingMemory()
        wm.add_message("user", "q1")
        wm.add_message("assistant", "a1")
        wm.add_message("user", "q2")
        msgs = wm.get_conversation()
        assert len(msgs) == 3
        assert msgs[0]["content"] == "q1"
        assert msgs[2]["content"] == "q2"

    def test_conversation_max_turns_respected(self):
        wm = WorkingMemory(max_conversation_turns=3)
        for i in range(6):
            wm.add_message("user", f"msg{i}")
        msgs = wm.get_conversation()
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg3"

    def test_clear_resets_state_and_conversation(self):
        wm = WorkingMemory()
        wm.update("key", "val")
        wm.add_message("user", "hello")
        wm.clear()
        assert wm.snapshot() == {}
        assert wm.get_conversation() == []

    def test_thread_safety_concurrent_updates(self):
        wm = WorkingMemory()
        errors = []

        def write(n):
            try:
                for _ in range(50):
                    wm.update(f"key_{n}", n)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_get_conversation_returns_list_copy(self):
        wm = WorkingMemory()
        wm.add_message("user", "hi")
        msgs = wm.get_conversation()
        msgs.clear()
        assert len(wm.get_conversation()) == 1


# ---------------------------------------------------------------------------
# EpisodicMemory (mocked DB + VectorStore)
# ---------------------------------------------------------------------------

class TestEpisodicMemoryMocked:
    def _make_episodic(self):
        db = MagicMock()
        vs = MagicMock()
        vs.query_episodes.return_value = []
        return EpisodicMemory(db=db, vector_store=vs), db, vs

    def test_record_calls_db_and_vs(self):
        em, db, vs = self._make_episodic()
        eid = em.record("User fixed a bug", mood="relieved", outcome="success")
        db.record_episode_meta.assert_called_once()
        vs.add_episode.assert_called_once()
        assert eid.startswith("ep_")

    def test_record_returns_unique_ids(self):
        em, _, _ = self._make_episodic()
        ids = {em.record(f"episode {i}") for i in range(10)}
        assert len(ids) == 10

    def test_recall_delegates_to_vector_store(self):
        em, _, vs = self._make_episodic()
        vs.query_episodes.return_value = [
            {
                "id": "ep_abc123",
                "text": "Debugged auth module",
                "metadata": {"mood": "focused", "outcome": "success"},
                "distance": 0.1,
            }
        ]
        results = em.recall("authentication debug", n=1)
        vs.query_episodes.assert_called_once_with("authentication debug", n=1)
        assert len(results) == 1
        assert results[0]["summary"] == "Debugged auth module"
        assert results[0]["mood"] == "focused"

    def test_recall_returns_empty_list_when_no_results(self):
        em, _, vs = self._make_episodic()
        vs.query_episodes.return_value = []
        assert em.recall("anything") == []

    def test_recall_enriches_with_mood_and_outcome(self):
        em, _, vs = self._make_episodic()
        vs.query_episodes.return_value = [
            {
                "id": "ep_xyz",
                "text": "Met project deadline",
                "metadata": {"mood": "happy", "outcome": "success", "tags": "work"},
                "distance": 0.05,
            }
        ]
        results = em.recall("project deadline")
        r = results[0]
        assert r["mood"] == "happy"
        assert r["outcome"] == "success"
        assert r["id"] == "ep_xyz"
        assert r["distance"] == 0.05

    def test_delete_calls_vector_store(self):
        em, _, vs = self._make_episodic()
        em.delete(["ep_aaa", "ep_bbb"])
        vs.delete_episodes.assert_called_once_with(["ep_aaa", "ep_bbb"])

    def test_record_with_context_tags_passes_metadata(self):
        em, _, vs = self._make_episodic()
        em.record("worked on project", context_tags=["coding", "work"])
        _, kwargs = vs.add_episode.call_args
        # tags should be passed in metadata or positional args
        call_args = vs.add_episode.call_args
        # metadata is the 3rd positional argument
        metadata = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("metadata", {})
        assert "coding" in metadata.get("tags", "")

    def test_record_without_optional_fields(self):
        em, db, vs = self._make_episodic()
        eid = em.record("Simple summary")
        assert eid is not None
        db.record_episode_meta.assert_called_once()


# ---------------------------------------------------------------------------
# EpisodicMemory (real DB + VectorStore)
# ---------------------------------------------------------------------------

class TestEpisodicMemoryIntegration:
    def test_record_and_recall(self, episodic):
        episodic.record(
            "User debugged a Python auth module",
            mood="frustrated",
            outcome="fixed",
            context_tags=["work", "coding"],
        )
        results = episodic.recall("authentication debugging", n=1)
        assert len(results) == 1
        assert "auth" in results[0]["summary"].lower()

    def test_recall_mood_preserved(self, episodic):
        episodic.record("Great team meeting today", mood="happy", context_tags=["work"])
        results = episodic.recall("team meeting", n=1)
        assert results[0]["mood"] == "happy"

    def test_recall_outcome_preserved(self, episodic):
        episodic.record("Shipped new feature", outcome="success", context_tags=["coding"])
        results = episodic.recall("feature shipping", n=1)
        assert results[0]["outcome"] == "success"

    def test_multiple_episodes_searchable(self, episodic):
        episodic.record("Python debugging session", context_tags=["coding"])
        episodic.record("Relaxing music break", context_tags=["leisure"])
        results = episodic.recall("Python code", n=2)
        assert any("Python" in r["summary"] for r in results)


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class TestSemanticMemory:
    def test_learn_fact(self, semantic):
        fact_id = semantic.learn("User prefers dark mode", confidence=0.8, tags=["preferences"])
        assert isinstance(fact_id, int)

    def test_get_facts_returns_learned(self, semantic):
        semantic.learn("User likes Python", confidence=0.9)
        facts = semantic.get_facts()
        assert any(f["fact"] == "User likes Python" for f in facts)

    def test_get_facts_confidence_filter(self, semantic):
        semantic.learn("Low confidence fact", confidence=0.3)
        semantic.learn("High confidence fact", confidence=0.8)
        high = semantic.get_facts(min_confidence=0.5)
        assert all(f["confidence"] >= 0.5 for f in high)
        assert any(f["fact"] == "High confidence fact" for f in high)

    def test_reinforce_increases_confidence(self, semantic):
        semantic.learn("User enjoys exercise", confidence=0.5)
        semantic.reinforce("User enjoys exercise", boost=0.2)
        facts = semantic.get_facts()
        match = next(f for f in facts if f["fact"] == "User enjoys exercise")
        assert match["confidence"] > 0.5

    def test_reinforce_caps_at_1_0(self, semantic):
        semantic.learn("Already certain", confidence=0.95)
        semantic.reinforce("Already certain", boost=0.1)
        facts = semantic.get_facts()
        match = next(f for f in facts if f["fact"] == "Already certain")
        assert match["confidence"] <= 1.0

    def test_forget_topic_archives_tagged_facts(self, semantic):
        semantic.learn("Loves rock music", confidence=0.8, tags=["music"])
        semantic.learn("Prefers Python", confidence=0.9, tags=["work"])
        semantic.forget_topic("music")
        facts = semantic.get_facts()
        assert all("music" not in str(f.get("tags", "")) for f in facts)

    def test_forget_fact_by_id(self, semantic):
        fid = semantic.learn("Temporary fact", confidence=0.7)
        semantic.forget_fact(fid)
        facts = semantic.get_facts()
        assert all(f["fact"] != "Temporary fact" for f in facts)

    def test_set_and_get_profile(self, semantic):
        semantic.set_profile("work", {"role": "engineer", "stack": ["Python", "React"]})
        profile = semantic.get_profile("work")
        assert profile["role"] == "engineer"
        assert "Python" in profile["stack"]

    def test_get_nonexistent_profile_returns_none(self, semantic):
        result = semantic.get_profile("nonexistent_domain")
        assert result is None

    def test_get_all_profiles(self, semantic):
        semantic.set_profile("work", {"role": "dev"})
        semantic.set_profile("personal", {"hobby": "hiking"})
        profiles = semantic.get_all_profiles()
        assert "work" in profiles
        assert "personal" in profiles

    def test_multiple_facts(self, semantic):
        for i in range(5):
            semantic.learn(f"Fact number {i}", confidence=0.6 + i * 0.05)
        facts = semantic.get_facts()
        assert len(facts) >= 5


# ---------------------------------------------------------------------------
# ForgettingCurve
# ---------------------------------------------------------------------------

class TestForgettingCurve:
    def test_recent_high_access_relevant(self, forgetting):
        score = forgetting.calculate_relevance(
            base_score=0.9,
            last_accessed=utc_now().isoformat(),
            access_count=20,
        )
        assert score > 0.5

    def test_old_low_access_decays_significantly(self, forgetting):
        old = (utc_now() - timedelta(days=60)).isoformat()
        score = forgetting.calculate_relevance(
            base_score=0.5, last_accessed=old, access_count=1
        )
        assert score < 0.2

    def test_ancient_fact_nearly_zero(self, forgetting):
        ancient = (utc_now() - timedelta(days=365)).isoformat()
        score = forgetting.calculate_relevance(
            base_score=0.9, last_accessed=ancient, access_count=1
        )
        assert score < 0.1

    def test_higher_access_count_slows_decay(self, forgetting):
        date = (utc_now() - timedelta(days=30)).isoformat()
        low = forgetting.calculate_relevance(0.8, date, access_count=1)
        high = forgetting.calculate_relevance(0.8, date, access_count=50)
        assert high > low

    def test_invalid_timestamp_returns_base_score(self, forgetting):
        score = forgetting.calculate_relevance(0.7, "not-a-date", access_count=5)
        assert score == 0.7

    def test_decay_rate_affects_result(self, db):
        date = (utc_now() - timedelta(days=10)).isoformat()
        slow = ForgettingCurve(db=db, decay_rate=0.01)
        fast = ForgettingCurve(db=db, decay_rate=0.5)
        slow_score = slow.calculate_relevance(1.0, date, access_count=1)
        fast_score = fast.calculate_relevance(1.0, date, access_count=1)
        assert slow_score > fast_score

    def test_decay_all_archives_old_facts(self, db):
        old = (utc_now() - timedelta(days=365)).isoformat()
        db._conn.execute(
            "INSERT INTO semantic_memory "
            "(fact, confidence, source_count, tags, created_at, last_confirmed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("Very old fact", 0.2, 1, "[]", old, old),
        )
        db._conn.commit()
        curve = ForgettingCurve(db=db, decay_rate=0.1)
        archived = curve.decay_all(threshold=0.05)
        assert archived >= 1

    def test_decay_all_preserves_recent_facts(self, db, semantic):
        semantic.learn("New fact today", confidence=0.9)
        curve = ForgettingCurve(db=db, decay_rate=0.1)
        archived = curve.decay_all(threshold=0.05)
        # Recent facts should not be archived
        assert archived == 0

    def test_decay_all_returns_zero_if_no_facts(self, db):
        curve = ForgettingCurve(db=db, decay_rate=0.1)
        archived = curve.decay_all(threshold=0.05)
        assert archived == 0


# ---------------------------------------------------------------------------
# MemoryConsolidator
# ---------------------------------------------------------------------------

class TestMemoryConsolidator:
    def _make_consolidator(self, generate_response: str = "User debugged auth."):
        engine = MagicMock()
        engine.generate.return_value = generate_response
        return MemoryConsolidator(model_engine=engine), engine

    def test_empty_conversation_returns_empty_summary(self):
        consolidator, engine = self._make_consolidator()
        wm = WorkingMemory()
        result = consolidator.create_session_digest(wm)
        assert result["summary"] == ""
        engine.generate.assert_not_called()

    def test_non_empty_conversation_calls_engine(self):
        consolidator, engine = self._make_consolidator("User worked on project.")
        wm = WorkingMemory()
        wm.add_message("user", "help me debug this")
        wm.add_message("assistant", "sure, here's how")
        result = consolidator.create_session_digest(wm)
        engine.generate.assert_called_once()
        assert "User worked on project." in result["summary"]

    def test_digest_includes_context(self):
        consolidator, engine = self._make_consolidator("Summary here.")
        wm = WorkingMemory()
        wm.update("activity_type", "coding")
        wm.add_message("user", "question")
        consolidator.create_session_digest(wm)
        prompt = engine.generate.call_args[0][0]
        assert "coding" in prompt

    def test_digest_has_required_keys(self):
        consolidator, _ = self._make_consolidator()
        wm = WorkingMemory()
        wm.add_message("user", "hi")
        result = consolidator.create_session_digest(wm)
        assert "summary" in result
        assert "mood" in result
        assert "key_events" in result

    def test_extract_facts_valid_json(self):
        consolidator, engine = self._make_consolidator()
        engine.generate.return_value = '["User prefers Python", "User works remotely"]'
        facts = consolidator.extract_facts("User prefers Python and works remotely.")
        assert "User prefers Python" in facts
        assert "User works remotely" in facts

    def test_extract_facts_invalid_json_returns_empty(self):
        consolidator, engine = self._make_consolidator()
        engine.generate.return_value = "not json at all"
        facts = consolidator.extract_facts("some summary")
        assert facts == []

    def test_extract_facts_empty_array(self):
        consolidator, engine = self._make_consolidator()
        engine.generate.return_value = "[]"
        facts = consolidator.extract_facts("nothing to extract")
        assert facts == []

    def test_extract_facts_filters_non_strings(self):
        consolidator, engine = self._make_consolidator()
        engine.generate.return_value = '["valid fact", 42, null, "another fact"]'
        facts = consolidator.extract_facts("mixed types")
        assert all(isinstance(f, str) for f in facts)
        assert "valid fact" in facts
