"""Tests for EmailIndexer."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from homie_core.knowledge.email_indexer import EmailIndexer
from homie_core.knowledge.graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(tmp_path: Path) -> KnowledgeGraph:
    return KnowledgeGraph(tmp_path / "kg.db")


def _make_email_db(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    """Create an in-memory-style SQLite email cache and return path."""
    db_path = tmp_path / "email_cache.db"
    db = sqlite3.connect(str(db_path))
    db.execute(
        """
        CREATE TABLE emails (
            id          TEXT PRIMARY KEY,
            sender      TEXT,
            subject     TEXT,
            date        TEXT
        )
        """
    )
    if rows:
        for row in rows:
            db.execute(
                "INSERT INTO emails (id, sender, subject, date) VALUES (?, ?, ?, ?)",
                (
                    row.get("id", "uid"),
                    row.get("sender", ""),
                    row.get("subject", ""),
                    row.get("date", "2026-01-01 00:00:00"),
                ),
            )
    db.commit()
    db.close()
    return db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmailIndexerNoGraph:
    def test_no_graph_returns_zero_stats(self, tmp_path):
        db_path = _make_email_db(tmp_path, [{"id": "1", "sender": "bob@example.com", "subject": "Hi"}])
        indexer = EmailIndexer(cache_db_path=db_path)  # no graph
        stats = indexer.index_recent()
        assert stats == {"indexed": 0, "entities_created": 0, "relationships_created": 0}


class TestEmailIndexerNoTable:
    def test_no_email_table_returns_zero_stats(self, tmp_path):
        # DB exists but has no emails table
        db_path = tmp_path / "empty.db"
        db = sqlite3.connect(str(db_path))
        db.close()

        graph = _make_graph(tmp_path)
        indexer = EmailIndexer(cache_db_path=db_path, graph=graph)
        stats = indexer.index_recent()
        assert stats == {"indexed": 0, "entities_created": 0, "relationships_created": 0}

    def test_missing_db_path_returns_zero_stats(self, tmp_path):
        graph = _make_graph(tmp_path)
        # Point at non-existent path but within a non-existent sub-directory
        bad_path = tmp_path / "no_such_dir" / "no.db"
        indexer = EmailIndexer(cache_db_path=bad_path, graph=graph)
        # SQLite will actually create the file — but table won't exist
        stats = indexer.index_recent()
        assert stats == {"indexed": 0, "entities_created": 0, "relationships_created": 0}


class TestEmailIndexerPersonEntity:
    def test_sender_creates_person_entity(self, tmp_path):
        db_path = _make_email_db(tmp_path, [
            {"id": "1", "sender": "Alice <alice@example.com>", "subject": "Hello", "date": "2026-03-01 10:00:00"},
        ])
        graph = _make_graph(tmp_path)
        indexer = EmailIndexer(cache_db_path=db_path, graph=graph)
        stats = indexer.index_recent(days=30)

        assert stats["indexed"] == 1
        persons = graph.find_entities(entity_type="person")
        assert len(persons) == 1
        assert persons[0].name == "Alice"

    def test_plain_email_sender_creates_entity(self, tmp_path):
        db_path = _make_email_db(tmp_path, [
            {"id": "1", "sender": "bob@example.com", "subject": "Sup", "date": "2026-03-01 10:00:00"},
        ])
        graph = _make_graph(tmp_path)
        indexer = EmailIndexer(cache_db_path=db_path, graph=graph)
        indexer.index_recent(days=30)

        persons = graph.find_entities(entity_type="person")
        assert len(persons) == 1
        assert "bob" in persons[0].name.lower() or "example" in persons[0].name.lower()


class TestEmailIndexerDocumentEntity:
    def test_subject_creates_document_entity(self, tmp_path):
        db_path = _make_email_db(tmp_path, [
            {"id": "1", "sender": "carol@example.com", "subject": "Q1 Report", "date": "2026-03-01 10:00:00"},
        ])
        graph = _make_graph(tmp_path)
        indexer = EmailIndexer(cache_db_path=db_path, graph=graph)
        indexer.index_recent(days=30)

        docs = graph.find_entities(entity_type="document")
        assert len(docs) == 1
        assert docs[0].name == "Q1 Report"

    def test_authored_relationship_created(self, tmp_path):
        db_path = _make_email_db(tmp_path, [
            {"id": "1", "sender": "Dave <dave@example.com>", "subject": "Sprint Review", "date": "2026-03-01 10:00:00"},
        ])
        graph = _make_graph(tmp_path)
        indexer = EmailIndexer(cache_db_path=db_path, graph=graph)
        stats = indexer.index_recent(days=30)

        assert stats["relationships_created"] == 1
        persons = graph.find_entities(entity_type="person")
        assert persons
        rels = graph.get_relationships(persons[0].id, direction="outgoing")
        assert any(r.relation == "authored" for r in rels)


class TestEmailIndexerStats:
    def test_stats_reflect_multiple_emails(self, tmp_path):
        db_path = _make_email_db(tmp_path, [
            {"id": "1", "sender": "Alice <a@x.com>", "subject": "Intro", "date": "2026-03-01 10:00:00"},
            {"id": "2", "sender": "Bob <b@x.com>", "subject": "Follow up", "date": "2026-03-02 10:00:00"},
        ])
        graph = _make_graph(tmp_path)
        indexer = EmailIndexer(cache_db_path=db_path, graph=graph)
        stats = indexer.index_recent(days=30)

        assert stats["indexed"] == 2
        # 2 senders + 2 subjects
        assert stats["entities_created"] == 4
        assert stats["relationships_created"] == 2

    def test_empty_table_returns_zero_stats(self, tmp_path):
        db_path = _make_email_db(tmp_path, [])
        graph = _make_graph(tmp_path)
        indexer = EmailIndexer(cache_db_path=db_path, graph=graph)
        stats = indexer.index_recent(days=30)
        assert stats == {"indexed": 0, "entities_created": 0, "relationships_created": 0}

    def test_subject_truncated_to_100_chars(self, tmp_path):
        long_subject = "A" * 150
        db_path = _make_email_db(tmp_path, [
            {"id": "1", "sender": "e@example.com", "subject": long_subject, "date": "2026-03-01 10:00:00"},
        ])
        graph = _make_graph(tmp_path)
        indexer = EmailIndexer(cache_db_path=db_path, graph=graph)
        indexer.index_recent(days=30)

        docs = graph.find_entities(entity_type="document")
        assert len(docs) == 1
        assert len(docs[0].name) == 100
