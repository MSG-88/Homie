# Knowledge Evolution System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Homie's knowledge from shallow in-memory triples to a persistent, temporally-versioned knowledge graph with guided intake, entity resolution, relationship inference, and contradiction detection.

**Architecture:** Extend the existing `homie_core/knowledge/` graph (Entity, Relationship, KnowledgeGraph) with temporal versioning, then build intake pipeline and reasoning engine as new modules under `adaptive_learning/knowledge/`. Wire the existing KnowledgeBuilder to persist through the graph instead of in-memory dicts.

**Tech Stack:** Python 3.11+, SQLite (existing knowledge graph), existing RAG parsers, existing EntityExtractor (spaCy + pattern-based), Python AST for surface code extraction.

**Spec:** `docs/superpowers/specs/2026-03-22-knowledge-evolution-system-design.md`

**Important:** The project already has `homie_core/knowledge/` with Entity/Relationship models, a KnowledgeGraph SQLite store, and an EntityExtractor. We extend these rather than duplicating.

**Implementation Notes (from review):**
1. Task 3 (`KnowledgeGraphStore`) creates a new store with temporal columns. At implementation time, consider wrapping the existing `homie_core/knowledge/graph.KnowledgeGraph` and adding temporal methods rather than building from scratch. If the existing schema is incompatible with temporal fields, the new store is acceptable but should be the single source of truth.
2. The spec calls for a `KnowledgeGraphProbe` registered with the HealthWatchdog. Add this as a small task during integration (Task 13 or 14) — create `src/homie_core/self_healing/probes/knowledge_graph_probe.py` that checks entity/relationship counts and query health.
3. The `EntityExtractor` in `homie_core/knowledge/extractor.py` can be reused by the deep analyzer. At implementation time, consider using it alongside or instead of the LLM extraction prompt.
4. Graph tables live in their own SQLite file (not in LearningStorage) for isolation. This is a deliberate deviation from the spec.

---

## Chunk 1: Temporal Versioning for Knowledge Graph

### Task 1: Add Temporal Fields to Relationship Model

**Files:**
- Modify: `src/homie_core/knowledge/models.py`
- Test: `tests/unit/knowledge_evolution/test_temporal_models.py`

- [ ] **Step 1: Create test directory**

```bash
mkdir -p tests/unit/knowledge_evolution
```

- [ ] **Step 2: Write failing test**

```python
# tests/unit/knowledge_evolution/__init__.py
```

```python
# tests/unit/knowledge_evolution/test_temporal_models.py
import time
import pytest
from homie_core.knowledge.models import Relationship, Entity


class TestTemporalRelationship:
    def test_relationship_has_temporal_fields(self):
        rel = Relationship(
            subject_id="e1",
            relation="works_at",
            object_id="e2",
            valid_from=time.time(),
            valid_until=None,
        )
        assert rel.valid_from > 0
        assert rel.valid_until is None

    def test_relationship_is_current(self):
        rel = Relationship(
            subject_id="e1",
            relation="works_at",
            object_id="e2",
            valid_from=time.time() - 100,
            valid_until=None,
        )
        assert rel.is_current is True

    def test_relationship_is_superseded(self):
        rel = Relationship(
            subject_id="e1",
            relation="works_at",
            object_id="e2",
            valid_from=time.time() - 200,
            valid_until=time.time() - 100,
        )
        assert rel.is_current is False

    def test_entity_has_aliases(self):
        ent = Entity(
            name="Python",
            entity_type="technology",
            aliases=["Python3", "CPython"],
        )
        assert "Python3" in ent.aliases
        assert len(ent.aliases) == 2

    def test_entity_default_aliases_empty(self):
        ent = Entity(name="Go", entity_type="technology")
        assert ent.aliases == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_temporal_models.py -v`
Expected: FAIL — Relationship missing valid_from/valid_until, Entity missing aliases

- [ ] **Step 4: Read existing models.py and add temporal fields**

Read `src/homie_core/knowledge/models.py` first. Add to the Relationship dataclass:
```python
    valid_from: float = field(default_factory=time.time)
    valid_until: Optional[float] = None

    @property
    def is_current(self) -> bool:
        return self.valid_until is None
```

Add to the Entity dataclass:
```python
    aliases: list[str] = field(default_factory=list)
```

Add `import time` and `from typing import Optional` if not present.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_temporal_models.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/knowledge/models.py tests/unit/knowledge_evolution/__init__.py tests/unit/knowledge_evolution/test_temporal_models.py
git commit -m "feat(knowledge): add temporal versioning fields to Relationship model"
```

---

### Task 2: Temporal Manager

**Files:**
- Create: `src/homie_core/adaptive_learning/knowledge/graph/__init__.py`
- Create: `src/homie_core/adaptive_learning/knowledge/graph/temporal.py`
- Test: `tests/unit/knowledge_evolution/test_temporal.py`

- [ ] **Step 1: Create directory**

```bash
mkdir -p src/homie_core/adaptive_learning/knowledge/graph
```

- [ ] **Step 2: Write failing test**

```python
# tests/unit/knowledge_evolution/test_temporal.py
import time
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.knowledge.graph.temporal import TemporalManager


class TestTemporalManager:
    def test_supersede_relationship(self):
        graph = MagicMock()
        old_rel = MagicMock(valid_until=None, subject_id="e1", relation="works_at", object_id="e2")
        graph.find_current_relationships.return_value = [old_rel]
        tm = TemporalManager(graph_store=graph)
        tm.supersede("e1", "works_at", "e3")
        assert old_rel.valid_until is not None  # was set
        graph.update_relationship.assert_called()

    def test_no_supersede_when_no_existing(self):
        graph = MagicMock()
        graph.find_current_relationships.return_value = []
        tm = TemporalManager(graph_store=graph)
        tm.supersede("e1", "works_at", "e3")
        graph.update_relationship.assert_not_called()

    def test_confidence_decay(self):
        tm = TemporalManager(graph_store=MagicMock(), decay_rate=0.99)
        base_confidence = 0.8
        age_days = 70
        decayed = tm.apply_decay(base_confidence, age_days)
        assert decayed < base_confidence
        assert decayed > 0

    def test_no_decay_for_fresh_facts(self):
        tm = TemporalManager(graph_store=MagicMock(), decay_rate=0.99)
        assert tm.apply_decay(0.8, age_days=0) == pytest.approx(0.8)

    def test_query_at_time(self):
        graph = MagicMock()
        tm = TemporalManager(graph_store=graph)
        tm.query_at_time("e1", "works_at", timestamp=time.time())
        graph.find_relationships_at_time.assert_called()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_temporal.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Write implementation**

```python
# src/homie_core/adaptive_learning/knowledge/graph/__init__.py
"""Knowledge graph extensions — temporal versioning, queries."""
```

```python
# src/homie_core/adaptive_learning/knowledge/graph/temporal.py
"""Temporal versioning — manages time ranges on relationships, confidence decay."""

import time
from typing import Any, Optional


class TemporalManager:
    """Manages temporal versioning and confidence decay for knowledge relationships."""

    def __init__(self, graph_store, decay_rate: float = 0.99) -> None:
        self._graph = graph_store
        self._decay_rate = decay_rate

    def supersede(self, subject_id: str, relation: str, new_object_id: str) -> None:
        """Supersede existing current relationships with a new one."""
        now = time.time()
        existing = self._graph.find_current_relationships(subject_id, relation)
        for rel in existing:
            if rel.object_id != new_object_id:
                rel.valid_until = now
                self._graph.update_relationship(rel)

    def apply_decay(self, base_confidence: float, age_days: float) -> float:
        """Apply time-based confidence decay."""
        return base_confidence * (self._decay_rate ** age_days)

    def query_at_time(
        self, subject_id: str, relation: str, timestamp: float
    ) -> list:
        """Query relationships that were valid at a specific timestamp."""
        return self._graph.find_relationships_at_time(subject_id, relation, timestamp)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_temporal.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/graph/ tests/unit/knowledge_evolution/test_temporal.py
git commit -m "feat(knowledge): add TemporalManager for versioning and confidence decay"
```

---

### Task 3: Knowledge Graph Store Extension

**Files:**
- Create: `src/homie_core/adaptive_learning/knowledge/graph/store.py`
- Test: `tests/unit/knowledge_evolution/test_graph_store.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/knowledge_evolution/test_graph_store.py
import time
import pytest
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore


class TestKnowledgeGraphStore:
    def test_add_and_get_entity(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        eid = store.add_entity("Python", "technology", aliases=["Python3", "CPython"])
        entity = store.get_entity(eid)
        assert entity is not None
        assert entity["name"] == "Python"
        assert "Python3" in entity["aliases"]

    def test_add_and_get_relationship(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        e1 = store.add_entity("User", "person")
        e2 = store.add_entity("Google", "organization")
        rid = store.add_relationship(e1, "works_at", e2, confidence=0.9, source="conversation")
        rels = store.get_relationships(subject_id=e1)
        assert len(rels) == 1
        assert rels[0]["relation"] == "works_at"
        assert rels[0]["valid_until"] is None

    def test_find_current_relationships(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        e1 = store.add_entity("User", "person")
        e2 = store.add_entity("Google", "organization")
        e3 = store.add_entity("Anthropic", "organization")
        # Old superseded relationship
        store.add_relationship(e1, "works_at", e2, confidence=0.9, source="conv", valid_until=time.time() - 100)
        # Current relationship
        store.add_relationship(e1, "works_at", e3, confidence=0.95, source="conv")
        current = store.find_current_relationships(e1, "works_at")
        assert len(current) == 1
        assert current[0]["object_id"] == e3

    def test_find_entity_by_name(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        store.add_entity("Python", "technology", aliases=["Python3"])
        result = store.find_entity_by_name("Python")
        assert result is not None
        result2 = store.find_entity_by_name("Python3")
        assert result2 is not None  # found via alias

    def test_find_entity_by_name_case_insensitive(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        store.add_entity("Python", "technology")
        assert store.find_entity_by_name("python") is not None

    def test_update_relationship_valid_until(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        e1 = store.add_entity("A", "thing")
        e2 = store.add_entity("B", "thing")
        rid = store.add_relationship(e1, "uses", e2, confidence=0.8, source="test")
        now = time.time()
        store.update_relationship_valid_until(rid, now)
        rels = store.get_relationships(subject_id=e1)
        assert rels[0]["valid_until"] is not None

    def test_find_relationships_at_time(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        e1 = store.add_entity("User", "person")
        e2 = store.add_entity("Google", "organization")
        past = time.time() - 1000
        store.add_relationship(e1, "works_at", e2, confidence=0.9, source="test", valid_from=past - 500, valid_until=past - 100)
        # Query at a time when the relationship was valid
        results = store.find_relationships_at_time(e1, "works_at", past - 300)
        assert len(results) == 1

    def test_entity_count(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        store.add_entity("A", "thing")
        store.add_entity("B", "thing")
        assert store.entity_count() == 2

    def test_close(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        store.close()
        store.close()  # double close safe
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_graph_store.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/knowledge/graph/store.py
"""KnowledgeGraphStore — SQLite CRUD for entities and temporal relationships."""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional


class KnowledgeGraphStore:
    """Persistent knowledge graph with temporal versioning."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Create database and tables."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                aliases TEXT NOT NULL DEFAULT '[]',
                properties TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kge_name ON kg_entities(name COLLATE NOCASE);
            CREATE INDEX IF NOT EXISTS idx_kge_type ON kg_entities(entity_type);

            CREATE TABLE IF NOT EXISTS kg_relationships (
                id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                object_id TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                source TEXT NOT NULL DEFAULT '',
                valid_from REAL NOT NULL,
                valid_until REAL,
                properties TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                FOREIGN KEY (subject_id) REFERENCES kg_entities(id),
                FOREIGN KEY (object_id) REFERENCES kg_entities(id)
            );
            CREATE INDEX IF NOT EXISTS idx_kgr_subject ON kg_relationships(subject_id);
            CREATE INDEX IF NOT EXISTS idx_kgr_object ON kg_relationships(object_id);
            CREATE INDEX IF NOT EXISTS idx_kgr_relation ON kg_relationships(relation);
            CREATE INDEX IF NOT EXISTS idx_kgr_valid ON kg_relationships(valid_until);
        """)
        self._conn.commit()

    def add_entity(
        self,
        name: str,
        entity_type: str,
        aliases: Optional[list[str]] = None,
        properties: Optional[dict] = None,
    ) -> str:
        """Add an entity. Returns entity ID."""
        if self._conn is None:
            return ""
        eid = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO kg_entities (id, name, entity_type, aliases, properties, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (eid, name, entity_type, json.dumps(aliases or []), json.dumps(properties or {}), now, now),
            )
            self._conn.commit()
        return eid

    def get_entity(self, entity_id: str) -> Optional[dict]:
        """Get entity by ID."""
        if self._conn is None:
            return None
        row = self._conn.execute("SELECT * FROM kg_entities WHERE id = ?", (entity_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["aliases"] = json.loads(d["aliases"])
        d["properties"] = json.loads(d["properties"])
        return d

    def find_entity_by_name(self, name: str) -> Optional[dict]:
        """Find entity by name or alias (case-insensitive)."""
        if self._conn is None:
            return None
        # Try exact name match first
        row = self._conn.execute(
            "SELECT * FROM kg_entities WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if row:
            d = dict(row)
            d["aliases"] = json.loads(d["aliases"])
            d["properties"] = json.loads(d["properties"])
            return d
        # Search aliases
        rows = self._conn.execute("SELECT * FROM kg_entities").fetchall()
        name_lower = name.lower()
        for row in rows:
            aliases = json.loads(row["aliases"])
            if any(a.lower() == name_lower for a in aliases):
                d = dict(row)
                d["aliases"] = json.loads(d["aliases"])
                d["properties"] = json.loads(d["properties"])
                return d
        return None

    def add_relationship(
        self,
        subject_id: str,
        relation: str,
        object_id: str,
        confidence: float = 0.5,
        source: str = "",
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
        properties: Optional[dict] = None,
    ) -> str:
        """Add a relationship. Returns relationship ID."""
        if self._conn is None:
            return ""
        rid = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO kg_relationships (id, subject_id, relation, object_id, confidence, source, valid_from, valid_until, properties, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, subject_id, relation, object_id, confidence, source, valid_from or now, valid_until, json.dumps(properties or {}), now),
            )
            self._conn.commit()
        return rid

    def get_relationships(self, subject_id: Optional[str] = None, object_id: Optional[str] = None) -> list[dict]:
        """Get relationships by subject and/or object."""
        if self._conn is None:
            return []
        clauses, params = [], []
        if subject_id:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        if object_id:
            clauses.append("object_id = ?")
            params.append(object_id)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(f"SELECT * FROM kg_relationships WHERE {where}", params).fetchall()
        return [dict(r) for r in rows]

    def find_current_relationships(self, subject_id: str, relation: str) -> list[dict]:
        """Find current (non-superseded) relationships."""
        if self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT * FROM kg_relationships WHERE subject_id = ? AND relation = ? AND valid_until IS NULL",
            (subject_id, relation),
        ).fetchall()
        return [dict(r) for r in rows]

    def find_relationships_at_time(self, subject_id: str, relation: str, timestamp: float) -> list[dict]:
        """Find relationships valid at a specific timestamp."""
        if self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT * FROM kg_relationships WHERE subject_id = ? AND relation = ? AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?)",
            (subject_id, relation, timestamp, timestamp),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_relationship_valid_until(self, relationship_id: str, valid_until: float) -> None:
        """Set valid_until on a relationship (supersede it)."""
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE kg_relationships SET valid_until = ? WHERE id = ?",
                (valid_until, relationship_id),
            )
            self._conn.commit()

    def entity_count(self) -> int:
        """Count total entities."""
        if self._conn is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) as c FROM kg_entities").fetchone()
        return row["c"]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_graph_store.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/graph/store.py tests/unit/knowledge_evolution/test_graph_store.py
git commit -m "feat(knowledge): add KnowledgeGraphStore with temporal relationships"
```

---

### Task 4: Graph Query Interface

**Files:**
- Create: `src/homie_core/adaptive_learning/knowledge/graph/query.py`
- Test: `tests/unit/knowledge_evolution/test_graph_query.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/knowledge_evolution/test_graph_query.py
import time
import pytest
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore
from homie_core.adaptive_learning.knowledge.graph.query import GraphQuery


class TestGraphQuery:
    def _setup_graph(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        user = store.add_entity("User", "person")
        homie = store.add_entity("Homie", "project")
        python = store.add_entity("Python", "technology")
        chromadb = store.add_entity("ChromaDB", "technology")
        store.add_relationship(user, "works_on", homie, confidence=0.95, source="conversation")
        store.add_relationship(homie, "uses", python, confidence=0.9, source="code_scan")
        store.add_relationship(homie, "uses", chromadb, confidence=0.85, source="code_scan")
        return store, {"user": user, "homie": homie, "python": python, "chromadb": chromadb}

    def test_get_entity_relationships(self, tmp_path):
        store, ids = self._setup_graph(tmp_path)
        query = GraphQuery(store=store)
        rels = query.get_entity_relationships(ids["homie"])
        assert len(rels) >= 2  # uses Python, uses ChromaDB

    def test_get_related_entities(self, tmp_path):
        store, ids = self._setup_graph(tmp_path)
        query = GraphQuery(store=store)
        related = query.get_related_entities(ids["homie"], relation="uses")
        names = [e["name"] for e in related]
        assert "Python" in names
        assert "ChromaDB" in names

    def test_traverse_one_hop(self, tmp_path):
        store, ids = self._setup_graph(tmp_path)
        query = GraphQuery(store=store)
        # User works_on Homie → Homie uses Python
        reachable = query.traverse(ids["user"], max_hops=2)
        entity_ids = [e["id"] for e in reachable]
        assert ids["python"] in entity_ids

    def test_get_entity_summary(self, tmp_path):
        store, ids = self._setup_graph(tmp_path)
        query = GraphQuery(store=store)
        summary = query.get_entity_summary(ids["homie"])
        assert summary["name"] == "Homie"
        assert len(summary["relationships"]) >= 2

    def test_search_entities(self, tmp_path):
        store, ids = self._setup_graph(tmp_path)
        query = GraphQuery(store=store)
        results = query.search_entities("python")
        assert len(results) >= 1
        assert results[0]["name"] == "Python"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_graph_query.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/knowledge/graph/query.py
"""GraphQuery — query current/historical facts and traverse relationships."""

from typing import Optional
from .store import KnowledgeGraphStore


class GraphQuery:
    """High-level query interface for the knowledge graph."""

    def __init__(self, store: KnowledgeGraphStore) -> None:
        self._store = store

    def get_entity_relationships(self, entity_id: str, current_only: bool = True) -> list[dict]:
        """Get all relationships for an entity (as subject or object)."""
        as_subject = self._store.get_relationships(subject_id=entity_id)
        as_object = self._store.get_relationships(object_id=entity_id)
        all_rels = as_subject + as_object
        if current_only:
            all_rels = [r for r in all_rels if r.get("valid_until") is None]
        return all_rels

    def get_related_entities(self, entity_id: str, relation: Optional[str] = None) -> list[dict]:
        """Get entities related to a given entity via current relationships."""
        rels = self._store.find_current_relationships(entity_id, relation) if relation else [
            r for r in self._store.get_relationships(subject_id=entity_id) if r.get("valid_until") is None
        ]
        entities = []
        for r in rels:
            obj = self._store.get_entity(r["object_id"])
            if obj:
                entities.append(obj)
        return entities

    def traverse(self, start_entity_id: str, max_hops: int = 2) -> list[dict]:
        """Traverse the graph from a starting entity up to max_hops."""
        visited = set()
        result = []
        queue = [(start_entity_id, 0)]

        while queue:
            eid, depth = queue.pop(0)
            if eid in visited or depth > max_hops:
                continue
            visited.add(eid)

            entity = self._store.get_entity(eid)
            if entity and eid != start_entity_id:
                result.append(entity)

            if depth < max_hops:
                rels = [r for r in self._store.get_relationships(subject_id=eid) if r.get("valid_until") is None]
                for r in rels:
                    if r["object_id"] not in visited:
                        queue.append((r["object_id"], depth + 1))

        return result

    def get_entity_summary(self, entity_id: str) -> dict:
        """Get a summary of an entity and its relationships."""
        entity = self._store.get_entity(entity_id)
        if entity is None:
            return {}
        rels = self.get_entity_relationships(entity_id, current_only=True)
        return {**entity, "relationships": rels}

    def search_entities(self, query: str, entity_type: Optional[str] = None) -> list[dict]:
        """Search entities by name (case-insensitive)."""
        result = self._store.find_entity_by_name(query)
        if result:
            if entity_type and result["entity_type"] != entity_type:
                return []
            return [result]
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_graph_query.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/graph/query.py tests/unit/knowledge_evolution/test_graph_query.py
git commit -m "feat(knowledge): add GraphQuery with traversal and search"
```

---

## Chunk 2: Guided Intake Pipeline

### Task 5: Source Scanner

**Files:**
- Create: `src/homie_core/adaptive_learning/knowledge/intake/__init__.py`
- Create: `src/homie_core/adaptive_learning/knowledge/intake/scanner.py`
- Test: `tests/unit/knowledge_evolution/test_scanner.py`

- [ ] **Step 1: Create directory**

```bash
mkdir -p src/homie_core/adaptive_learning/knowledge/intake
```

- [ ] **Step 2: Write failing test**

```python
# tests/unit/knowledge_evolution/test_scanner.py
import pytest
from homie_core.adaptive_learning.knowledge.intake.scanner import SourceScanner, FileInfo


class TestSourceScanner:
    def test_scan_directory(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (tmp_path / "readme.md").write_text("# Readme")
        scanner = SourceScanner()
        files = scanner.scan_directory(tmp_path)
        assert len(files) == 3

    def test_file_info_has_metadata(self, tmp_path):
        (tmp_path / "test.py").write_text("x = 1")
        scanner = SourceScanner()
        files = scanner.scan_directory(tmp_path)
        assert files[0].path is not None
        assert files[0].file_type in ("python", "unknown")
        assert files[0].size_bytes > 0

    def test_filters_by_extension(self, tmp_path):
        (tmp_path / "code.py").write_text("x = 1")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "data.json").write_text("{}")
        scanner = SourceScanner(include_extensions={".py", ".json"})
        files = scanner.scan_directory(tmp_path)
        assert len(files) == 2

    def test_skips_hidden_and_venv(self, tmp_path):
        (tmp_path / "visible.py").write_text("x = 1")
        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "config").write_text("git")
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "pip.py").write_text("pip")
        scanner = SourceScanner()
        files = scanner.scan_directory(tmp_path)
        paths = [str(f.path) for f in files]
        assert not any(".git" in p for p in paths)
        assert not any(".venv" in p for p in paths)

    def test_respects_max_depth(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (tmp_path / "top.py").write_text("x")
        (deep / "deep.py").write_text("y")
        scanner = SourceScanner(max_depth=1)
        files = scanner.scan_directory(tmp_path)
        names = [f.path.name for f in files]
        assert "top.py" in names
        assert "deep.py" not in names
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_scanner.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Write implementation**

```python
# src/homie_core/adaptive_learning/knowledge/intake/__init__.py
"""Guided intake pipeline — scan, extract, and ingest knowledge from sources."""
```

```python
# src/homie_core/adaptive_learning/knowledge/intake/scanner.py
"""Source scanner — enumerates files and detects types."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_EXT_TO_TYPE = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
    ".rb": "ruby", ".sh": "bash", ".kt": "kotlin",
    ".md": "markdown", ".txt": "text", ".rst": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".html": "html", ".css": "css", ".sql": "sql",
    ".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".pptx": "pptx",
}

_SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules", ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs"}


@dataclass
class FileInfo:
    path: Path
    file_type: str
    size_bytes: int
    extension: str


class SourceScanner:
    """Scans directories and enumerates files with metadata."""

    def __init__(
        self,
        include_extensions: Optional[set[str]] = None,
        max_depth: int = 10,
        max_file_size_mb: float = 10.0,
    ) -> None:
        self._include = include_extensions
        self._max_depth = max_depth
        self._max_size = int(max_file_size_mb * 1024 * 1024)

    def scan_directory(self, root: Path | str, depth: int = 0) -> list[FileInfo]:
        """Recursively scan a directory for files."""
        root = Path(root)
        results = []

        if depth > self._max_depth:
            return results

        try:
            entries = sorted(root.iterdir())
        except PermissionError:
            return results

        for entry in entries:
            if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
                continue

            if entry.is_dir():
                results.extend(self.scan_directory(entry, depth + 1))
            elif entry.is_file():
                if entry.stat().st_size > self._max_size:
                    continue
                ext = entry.suffix.lower()
                if self._include and ext not in self._include:
                    continue
                file_type = _EXT_TO_TYPE.get(ext, "unknown")
                results.append(FileInfo(
                    path=entry,
                    file_type=file_type,
                    size_bytes=entry.stat().st_size,
                    extension=ext,
                ))

        return results
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_scanner.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/intake/ tests/unit/knowledge_evolution/test_scanner.py
git commit -m "feat(knowledge): add source scanner for guided intake"
```

---

### Task 6: Surface Extractor

**Files:**
- Create: `src/homie_core/adaptive_learning/knowledge/intake/surface_extractor.py`
- Test: `tests/unit/knowledge_evolution/test_surface_extractor.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/knowledge_evolution/test_surface_extractor.py
import pytest
from pathlib import Path
from homie_core.adaptive_learning.knowledge.intake.surface_extractor import SurfaceExtractor


class TestSurfaceExtractor:
    def test_extract_python_classes(self, tmp_path):
        code = tmp_path / "example.py"
        code.write_text("class MyClass:\n    def method(self):\n        pass\n\nclass Other:\n    pass\n")
        ext = SurfaceExtractor()
        result = ext.extract(code, file_type="python")
        assert "MyClass" in result["classes"]
        assert "Other" in result["classes"]

    def test_extract_python_functions(self, tmp_path):
        code = tmp_path / "funcs.py"
        code.write_text("def hello():\n    pass\n\ndef world(x, y):\n    return x + y\n")
        ext = SurfaceExtractor()
        result = ext.extract(code, file_type="python")
        assert "hello" in result["functions"]
        assert "world" in result["functions"]

    def test_extract_python_imports(self, tmp_path):
        code = tmp_path / "imports.py"
        code.write_text("import os\nfrom pathlib import Path\nimport json\n")
        ext = SurfaceExtractor()
        result = ext.extract(code, file_type="python")
        assert "os" in result["imports"]
        assert "pathlib" in result["imports"]

    def test_extract_markdown_headings(self, tmp_path):
        doc = tmp_path / "readme.md"
        doc.write_text("# Title\n## Section 1\nSome text.\n### Subsection\nMore text.\n")
        ext = SurfaceExtractor()
        result = ext.extract(doc, file_type="markdown")
        assert "Title" in result["headings"]
        assert "Section 1" in result["headings"]

    def test_extract_unknown_returns_minimal(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("some data here")
        ext = SurfaceExtractor()
        result = ext.extract(f, file_type="unknown")
        assert "line_count" in result

    def test_extract_returns_entities(self, tmp_path):
        code = tmp_path / "entities.py"
        code.write_text("class UserService:\n    '''Handles user operations.'''\n    pass\n")
        ext = SurfaceExtractor()
        result = ext.extract(code, file_type="python")
        assert len(result.get("entities", [])) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_surface_extractor.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/knowledge/intake/surface_extractor.py
"""Surface extractor — quick AST/regex extraction without LLM."""

import ast
import re
from pathlib import Path
from typing import Any


class SurfaceExtractor:
    """Extracts surface-level knowledge from files using AST and regex."""

    def extract(self, file_path: Path, file_type: str) -> dict[str, Any]:
        """Extract surface knowledge from a file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            return {"error": "unreadable"}

        base = {
            "file": str(file_path),
            "file_type": file_type,
            "line_count": content.count("\n") + 1,
            "size_bytes": len(content.encode("utf-8")),
        }

        if file_type == "python":
            base.update(self._extract_python(content))
        elif file_type == "markdown":
            base.update(self._extract_markdown(content))
        elif file_type in ("javascript", "typescript"):
            base.update(self._extract_js(content))
        else:
            base.update(self._extract_generic(content))

        # Generate entity list from extracted names
        entities = []
        for cls in base.get("classes", []):
            entities.append({"name": cls, "type": "class", "source_file": str(file_path)})
        for fn in base.get("functions", []):
            entities.append({"name": fn, "type": "function", "source_file": str(file_path)})
        base["entities"] = entities

        return base

    def _extract_python(self, content: str) -> dict:
        """Extract from Python using AST."""
        classes, functions, imports = [], [], []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Skip methods (inside classes)
                    if not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree) if hasattr(p, 'body') and node in getattr(p, 'body', [])):
                        functions.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split(".")[0])
        except SyntaxError:
            # Fallback to regex
            classes = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
            functions = re.findall(r"^def\s+(\w+)", content, re.MULTILINE)
            imports = re.findall(r"^(?:import|from)\s+(\w+)", content, re.MULTILINE)

        return {
            "classes": classes,
            "functions": functions,
            "imports": list(set(imports)),
        }

    def _extract_markdown(self, content: str) -> dict:
        """Extract headings from markdown."""
        headings = re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)
        return {"headings": headings}

    def _extract_js(self, content: str) -> dict:
        """Extract from JavaScript/TypeScript using regex."""
        classes = re.findall(r"(?:class|interface)\s+(\w+)", content)
        functions = re.findall(r"(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:async\s*)?\(|[\(])", content)
        imports = re.findall(r"(?:import|require)\s*\(?['\"]([^'\"]+)", content)
        return {"classes": classes, "functions": functions, "imports": imports}

    def _extract_generic(self, content: str) -> dict:
        """Minimal extraction for unknown file types."""
        return {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_surface_extractor.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/intake/surface_extractor.py tests/unit/knowledge_evolution/test_surface_extractor.py
git commit -m "feat(knowledge): add surface extractor with Python AST and regex"
```

---

### Task 7: Value Scorer

**Files:**
- Create: `src/homie_core/adaptive_learning/knowledge/intake/value_scorer.py`
- Test: `tests/unit/knowledge_evolution/test_value_scorer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/knowledge_evolution/test_value_scorer.py
import pytest
from homie_core.adaptive_learning.knowledge.intake.value_scorer import ValueScorer


class TestValueScorer:
    def test_score_by_import_count(self):
        scorer = ValueScorer()
        extractions = [
            {"file": "core.py", "imports": [], "classes": ["Core"], "functions": []},
            {"file": "utils.py", "imports": ["core"], "classes": [], "functions": ["helper"]},
            {"file": "main.py", "imports": ["core", "utils"], "classes": [], "functions": ["main"]},
        ]
        # Build import reference counts
        import_counts = {"core": 2, "utils": 1}
        scores = scorer.score_files(extractions, import_counts=import_counts)
        # core.py should score highest (most referenced)
        assert scores["core.py"] > scores["utils.py"]

    def test_score_by_size(self):
        scorer = ValueScorer()
        extractions = [
            {"file": "big.py", "line_count": 500, "classes": [], "functions": [], "imports": []},
            {"file": "small.py", "line_count": 10, "classes": [], "functions": [], "imports": []},
        ]
        scores = scorer.score_files(extractions)
        assert scores["big.py"] > scores["small.py"]

    def test_select_top_percent(self):
        scorer = ValueScorer(top_percent=50)
        extractions = [
            {"file": f"file{i}.py", "line_count": i * 10, "classes": [], "functions": [], "imports": []}
            for i in range(10)
        ]
        selected = scorer.select_for_deep_pass(extractions)
        assert len(selected) == 5  # top 50%

    def test_empty_extractions(self):
        scorer = ValueScorer()
        assert scorer.score_files([]) == {}
        assert scorer.select_for_deep_pass([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_value_scorer.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/knowledge/intake/value_scorer.py
"""Value scorer — determines which files deserve deep LLM analysis."""

import math
from typing import Any, Optional


class ValueScorer:
    """Scores files by value to determine which get deep analysis."""

    def __init__(self, top_percent: int = 20) -> None:
        self._top_percent = top_percent

    def score_files(
        self,
        extractions: list[dict[str, Any]],
        import_counts: Optional[dict[str, int]] = None,
    ) -> dict[str, float]:
        """Score each file. Higher = more valuable for deep analysis."""
        if not extractions:
            return {}

        import_counts = import_counts or {}
        scores = {}

        for ext in extractions:
            file_key = ext.get("file", "")
            score = 0.0

            # Size score — larger files have more to extract
            line_count = ext.get("line_count", 0)
            score += min(math.log(line_count + 1) / 6.0, 1.0)  # caps at ~400 lines

            # Class/function density — more definitions = more architecture
            classes = len(ext.get("classes", []))
            functions = len(ext.get("functions", []))
            score += min((classes + functions) / 10.0, 1.0)

            # Import reference count — how many other files import this one
            basename = file_key.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].replace(".py", "")
            ref_count = import_counts.get(basename, 0)
            score += min(ref_count / 5.0, 1.0)

            scores[file_key] = score

        return scores

    def select_for_deep_pass(
        self,
        extractions: list[dict[str, Any]],
        import_counts: Optional[dict[str, int]] = None,
    ) -> list[dict]:
        """Select the top N% of files for deep analysis."""
        if not extractions:
            return []

        scores = self.score_files(extractions, import_counts)
        sorted_files = sorted(extractions, key=lambda e: scores.get(e.get("file", ""), 0), reverse=True)
        count = max(1, len(sorted_files) * self._top_percent // 100)
        return sorted_files[:count]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_value_scorer.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/intake/value_scorer.py tests/unit/knowledge_evolution/test_value_scorer.py
git commit -m "feat(knowledge): add value scorer for adaptive depth intake"
```

---

### Task 8: Deep Analyzer

**Files:**
- Create: `src/homie_core/adaptive_learning/knowledge/intake/deep_analyzer.py`
- Test: `tests/unit/knowledge_evolution/test_deep_analyzer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/knowledge_evolution/test_deep_analyzer.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.knowledge.intake.deep_analyzer import DeepAnalyzer


class TestDeepAnalyzer:
    def test_analyze_returns_entities_and_relationships(self):
        inference_fn = MagicMock(return_value='{"entities": [{"name": "UserService", "type": "class"}], "relationships": [{"subject": "UserService", "predicate": "handles", "object": "user operations"}]}')
        analyzer = DeepAnalyzer(inference_fn=inference_fn)
        result = analyzer.analyze("class UserService:\n    pass", file_path="service.py")
        assert "entities" in result
        assert "relationships" in result

    def test_analyze_without_llm_returns_empty(self):
        analyzer = DeepAnalyzer(inference_fn=None)
        result = analyzer.analyze("some code", file_path="test.py")
        assert result["entities"] == []
        assert result["relationships"] == []

    def test_handles_malformed_llm_response(self):
        inference_fn = MagicMock(return_value="not valid json")
        analyzer = DeepAnalyzer(inference_fn=inference_fn)
        result = analyzer.analyze("code", file_path="test.py")
        assert result["entities"] == []

    def test_truncates_long_content(self):
        inference_fn = MagicMock(return_value='{"entities": [], "relationships": []}')
        analyzer = DeepAnalyzer(inference_fn=inference_fn, max_content_chars=100)
        long_content = "x" * 500
        analyzer.analyze(long_content, file_path="test.py")
        # Should have been called with truncated content
        call_args = inference_fn.call_args[0][0]
        assert len(call_args) < 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_deep_analyzer.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/knowledge/intake/deep_analyzer.py
"""Deep analyzer — LLM-powered knowledge extraction from file content."""

import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """Analyze this source code/document and extract structured knowledge.

File: {file_path}
Content:
{content}

Extract as JSON:
{{
  "entities": [{{"name": "...", "type": "class|function|module|concept|technology"}}],
  "relationships": [{{"subject": "...", "predicate": "uses|depends_on|implements|handles|contains", "object": "..."}}]
}}

Be concise. Only extract clearly stated facts, not speculation."""


class DeepAnalyzer:
    """LLM-powered deep extraction of entities and relationships."""

    def __init__(
        self,
        inference_fn: Optional[Callable[[str], str]] = None,
        max_content_chars: int = 8000,
    ) -> None:
        self._infer = inference_fn
        self._max_chars = max_content_chars

    def analyze(self, content: str, file_path: str = "") -> dict[str, Any]:
        """Analyze content and extract entities and relationships."""
        if self._infer is None:
            return {"entities": [], "relationships": []}

        # Truncate if needed
        if len(content) > self._max_chars:
            content = content[:self._max_chars] + "\n... (truncated)"

        prompt = _EXTRACTION_PROMPT.format(file_path=file_path, content=content)

        try:
            response = self._infer(prompt)
            return self._parse_response(response)
        except Exception:
            logger.warning("Deep analysis failed for %s", file_path)
            return {"entities": [], "relationships": []}

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response as JSON."""
        # Try to find JSON in the response
        try:
            # Look for JSON block
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return {
                    "entities": data.get("entities", []),
                    "relationships": data.get("relationships", []),
                }
        except (json.JSONDecodeError, ValueError):
            pass
        return {"entities": [], "relationships": []}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_deep_analyzer.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/intake/deep_analyzer.py tests/unit/knowledge_evolution/test_deep_analyzer.py
git commit -m "feat(knowledge): add deep analyzer for LLM-powered extraction"
```

---

### Task 9: Intake Pipeline Coordinator

**Files:**
- Create: `src/homie_core/adaptive_learning/knowledge/intake/pipeline.py`
- Test: `tests/unit/knowledge_evolution/test_intake_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/knowledge_evolution/test_intake_pipeline.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.knowledge.intake.pipeline import IntakePipeline


class TestIntakePipeline:
    def test_ingest_directory(self, tmp_path):
        (tmp_path / "main.py").write_text("class App:\n    pass\n\ndef run():\n    pass\n")
        (tmp_path / "utils.py").write_text("import os\ndef helper():\n    pass\n")
        graph_store = MagicMock()
        graph_store.find_entity_by_name.return_value = None
        graph_store.add_entity.return_value = "eid-123"
        pipeline = IntakePipeline(graph_store=graph_store, inference_fn=None)
        result = pipeline.ingest(tmp_path)
        assert result["files_scanned"] >= 2
        assert result["entities_created"] >= 0
        assert graph_store.add_entity.called

    def test_ingest_single_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("class Service:\n    pass\n")
        graph_store = MagicMock()
        graph_store.find_entity_by_name.return_value = None
        graph_store.add_entity.return_value = "eid"
        pipeline = IntakePipeline(graph_store=graph_store, inference_fn=None)
        result = pipeline.ingest(f)
        assert result["files_scanned"] == 1

    def test_ingest_empty_directory(self, tmp_path):
        graph_store = MagicMock()
        pipeline = IntakePipeline(graph_store=graph_store, inference_fn=None)
        result = pipeline.ingest(tmp_path)
        assert result["files_scanned"] == 0

    def test_reports_deep_pass_count(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"class C{i}:\n    pass\n" * 20)
        graph_store = MagicMock()
        graph_store.find_entity_by_name.return_value = None
        graph_store.add_entity.return_value = "eid"
        # No inference_fn = no deep pass
        pipeline = IntakePipeline(graph_store=graph_store, inference_fn=None, deep_pass_top_percent=20)
        result = pipeline.ingest(tmp_path)
        assert result["deep_analyzed"] == 0  # no LLM available
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_intake_pipeline.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/knowledge/intake/pipeline.py
"""Intake pipeline — coordinates scanning, extraction, and graph ingestion."""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from ..graph.store import KnowledgeGraphStore
from .deep_analyzer import DeepAnalyzer
from .scanner import SourceScanner
from .surface_extractor import SurfaceExtractor
from .value_scorer import ValueScorer

logger = logging.getLogger(__name__)


class IntakePipeline:
    """Orchestrates guided intake: scan → surface extract → score → deep analyze → graph."""

    def __init__(
        self,
        graph_store: KnowledgeGraphStore,
        inference_fn: Optional[Callable[[str], str]] = None,
        deep_pass_top_percent: int = 20,
        max_deep_files: int = 50,
    ) -> None:
        self._graph = graph_store
        self._scanner = SourceScanner()
        self._surface = SurfaceExtractor()
        self._scorer = ValueScorer(top_percent=deep_pass_top_percent)
        self._deep = DeepAnalyzer(inference_fn=inference_fn)
        self._max_deep = max_deep_files
        self._has_llm = inference_fn is not None

    def ingest(self, source: Path | str) -> dict[str, Any]:
        """Ingest a directory or file into the knowledge graph."""
        source = Path(source)
        result = {"files_scanned": 0, "entities_created": 0, "relationships_created": 0, "deep_analyzed": 0}

        # Scan
        if source.is_file():
            from .scanner import FileInfo
            files = [FileInfo(path=source, file_type=self._detect_type(source), size_bytes=source.stat().st_size, extension=source.suffix)]
        elif source.is_dir():
            files = self._scanner.scan_directory(source)
        else:
            return result

        result["files_scanned"] = len(files)

        # Surface pass
        extractions = []
        for fi in files:
            ext = self._surface.extract(fi.path, fi.file_type)
            extractions.append(ext)
            # Add entities to graph
            for entity in ext.get("entities", []):
                self._add_entity_to_graph(entity)
                result["entities_created"] += 1

        # Deep pass (if LLM available)
        if self._has_llm and extractions:
            selected = self._scorer.select_for_deep_pass(extractions)[:self._max_deep]
            for ext in selected:
                file_path = ext.get("file", "")
                try:
                    content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                    deep = self._deep.analyze(content, file_path=file_path)
                    for entity in deep.get("entities", []):
                        self._add_entity_to_graph(entity)
                        result["entities_created"] += 1
                    for rel in deep.get("relationships", []):
                        self._add_relationship_to_graph(rel)
                        result["relationships_created"] += 1
                    result["deep_analyzed"] += 1
                except Exception:
                    logger.warning("Deep analysis failed for %s", file_path)

        return result

    def _add_entity_to_graph(self, entity: dict) -> Optional[str]:
        """Add an entity to the graph, deduplicating by name."""
        name = entity.get("name", "")
        if not name:
            return None
        existing = self._graph.find_entity_by_name(name)
        if existing:
            return existing["id"]
        return self._graph.add_entity(
            name=name,
            entity_type=entity.get("type", "concept"),
        )

    def _add_relationship_to_graph(self, rel: dict) -> None:
        """Add a relationship to the graph."""
        subject_name = rel.get("subject", "")
        object_name = rel.get("object", "")
        predicate = rel.get("predicate", "related_to")
        if not subject_name or not object_name:
            return

        sub = self._graph.find_entity_by_name(subject_name)
        obj = self._graph.find_entity_by_name(object_name)
        if not sub:
            sub_id = self._graph.add_entity(subject_name, "concept")
        else:
            sub_id = sub["id"]
        if not obj:
            obj_id = self._graph.add_entity(object_name, "concept")
        else:
            obj_id = obj["id"]

        self._graph.add_relationship(sub_id, predicate, obj_id, confidence=0.75, source="deep_analysis")

    def _detect_type(self, path: Path) -> str:
        from .scanner import _EXT_TO_TYPE
        return _EXT_TO_TYPE.get(path.suffix.lower(), "unknown")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_intake_pipeline.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/intake/pipeline.py tests/unit/knowledge_evolution/test_intake_pipeline.py
git commit -m "feat(knowledge): add intake pipeline coordinator"
```

---

## Chunk 3: Reasoning Engine

### Task 10: Entity Resolver

**Files:**
- Create: `src/homie_core/adaptive_learning/knowledge/reasoning/__init__.py`
- Create: `src/homie_core/adaptive_learning/knowledge/reasoning/entity_resolver.py`
- Test: `tests/unit/knowledge_evolution/test_entity_resolver.py`

- [ ] **Step 1: Create directory**

```bash
mkdir -p src/homie_core/adaptive_learning/knowledge/reasoning
```

- [ ] **Step 2: Write failing test**

```python
# tests/unit/knowledge_evolution/test_entity_resolver.py
import pytest
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore
from homie_core.adaptive_learning.knowledge.reasoning.entity_resolver import EntityResolver


class TestEntityResolver:
    def _setup(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        return store

    def test_exact_name_match(self, tmp_path):
        store = self._setup(tmp_path)
        e1 = store.add_entity("Python", "technology")
        resolver = EntityResolver(graph_store=store)
        match = resolver.resolve("Python", "technology")
        assert match is not None
        assert match["id"] == e1

    def test_alias_match(self, tmp_path):
        store = self._setup(tmp_path)
        e1 = store.add_entity("Python", "technology", aliases=["Python3", "CPython"])
        resolver = EntityResolver(graph_store=store)
        match = resolver.resolve("Python3", "technology")
        assert match is not None
        assert match["id"] == e1

    def test_no_match_returns_none(self, tmp_path):
        store = self._setup(tmp_path)
        resolver = EntityResolver(graph_store=store)
        assert resolver.resolve("Nonexistent", "thing") is None

    def test_fuzzy_match(self, tmp_path):
        store = self._setup(tmp_path)
        store.add_entity("ChromaDB", "technology")
        resolver = EntityResolver(graph_store=store, fuzzy_threshold=0.7)
        match = resolver.resolve("chromadb", "technology")
        assert match is not None

    def test_merge_entities(self, tmp_path):
        store = self._setup(tmp_path)
        e1 = store.add_entity("Python", "technology", aliases=["Python3"])
        e2 = store.add_entity("python", "technology", aliases=["CPython"])
        resolver = EntityResolver(graph_store=store)
        merged_id = resolver.merge(e1, e2)
        # After merge, the surviving entity should have combined aliases
        entity = store.get_entity(merged_id)
        assert "Python3" in entity["aliases"] or "CPython" in entity["aliases"]
```

- [ ] **Step 2b: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_entity_resolver.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/knowledge/reasoning/__init__.py
"""Knowledge reasoning — entity resolution, inference, contradiction detection."""
```

```python
# src/homie_core/adaptive_learning/knowledge/reasoning/entity_resolver.py
"""Entity resolver — detects and merges duplicate entities."""

import json
import logging
from typing import Optional

from ..graph.store import KnowledgeGraphStore

logger = logging.getLogger(__name__)


def _similarity(a: str, b: str) -> float:
    """Simple normalized string similarity."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # Jaccard on character bigrams
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1))
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 1.0 if a == b else 0.0
    return len(ba & bb) / len(ba | bb)


class EntityResolver:
    """Detects and merges duplicate entities in the knowledge graph."""

    def __init__(
        self,
        graph_store: KnowledgeGraphStore,
        fuzzy_threshold: float = 0.85,
    ) -> None:
        self._graph = graph_store
        self._threshold = fuzzy_threshold

    def resolve(self, name: str, entity_type: str) -> Optional[dict]:
        """Try to find an existing entity matching this name."""
        # Exact name match
        existing = self._graph.find_entity_by_name(name)
        if existing and existing["entity_type"] == entity_type:
            return existing

        # Case-insensitive match already handled by find_entity_by_name
        if existing:
            return existing

        # Fuzzy match against all entities of same type — only if DB is small
        if self._graph.entity_count() > 1000:
            return None  # skip fuzzy for large graphs

        # Simple fuzzy: check if similarity exceeds threshold
        if self._graph._conn is None:
            return None
        rows = self._graph._conn.execute(
            "SELECT * FROM kg_entities WHERE entity_type = ?", (entity_type,)
        ).fetchall()
        for row in rows:
            if _similarity(name, row["name"]) >= self._threshold:
                d = dict(row)
                d["aliases"] = json.loads(d["aliases"])
                d["properties"] = json.loads(d["properties"])
                return d

        return None

    def merge(self, keep_id: str, remove_id: str) -> str:
        """Merge two entities — keep one, repoint relationships from other."""
        keep = self._graph.get_entity(keep_id)
        remove = self._graph.get_entity(remove_id)
        if not keep or not remove:
            return keep_id

        # Merge aliases
        combined_aliases = list(set(keep.get("aliases", []) + remove.get("aliases", []) + [remove["name"]]))
        if self._graph._conn:
            self._graph._conn.execute(
                "UPDATE kg_entities SET aliases = ? WHERE id = ?",
                (json.dumps(combined_aliases), keep_id),
            )
            # Repoint relationships
            self._graph._conn.execute(
                "UPDATE kg_relationships SET subject_id = ? WHERE subject_id = ?",
                (keep_id, remove_id),
            )
            self._graph._conn.execute(
                "UPDATE kg_relationships SET object_id = ? WHERE object_id = ?",
                (keep_id, remove_id),
            )
            # Remove merged entity
            self._graph._conn.execute("DELETE FROM kg_entities WHERE id = ?", (remove_id,))
            self._graph._conn.commit()

        logger.info("Merged entity '%s' into '%s'", remove["name"], keep["name"])
        return keep_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_entity_resolver.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/reasoning/ tests/unit/knowledge_evolution/test_entity_resolver.py
git commit -m "feat(knowledge): add entity resolver with fuzzy matching and merge"
```

---

### Task 11: Inference Engine

**Files:**
- Create: `src/homie_core/adaptive_learning/knowledge/reasoning/inference_engine.py`
- Test: `tests/unit/knowledge_evolution/test_inference_engine.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/knowledge_evolution/test_inference_engine.py
import pytest
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore
from homie_core.adaptive_learning.knowledge.reasoning.inference_engine import InferenceEngine


class TestInferenceEngine:
    def _setup(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        user = store.add_entity("User", "person")
        homie = store.add_entity("Homie", "project")
        python = store.add_entity("Python", "technology")
        store.add_relationship(user, "works_on", homie, confidence=0.95, source="conversation")
        store.add_relationship(homie, "uses", python, confidence=0.9, source="code_scan")
        return store, {"user": user, "homie": homie, "python": python}

    def test_infer_transitive(self, tmp_path):
        store, ids = self._setup(tmp_path)
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        # Should infer: User works_with Python
        assert len(inferred) >= 1
        assert any(r["relation"] == "works_with" for r in inferred)

    def test_inferred_has_lower_confidence(self, tmp_path):
        store, ids = self._setup(tmp_path)
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        for r in inferred:
            assert r["confidence"] < 0.95  # lower than source facts

    def test_inferred_marked_as_inference_source(self, tmp_path):
        store, ids = self._setup(tmp_path)
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        for r in inferred:
            assert r["source"] == "inference"

    def test_no_duplicate_inference(self, tmp_path):
        store, ids = self._setup(tmp_path)
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred1 = engine.run_inference()
        inferred2 = engine.run_inference()
        # Second run should not create duplicates
        all_rels = store.get_relationships(subject_id=ids["user"])
        inference_rels = [r for r in all_rels if r["source"] == "inference"]
        assert len(inference_rels) <= len(inferred1)

    def test_respects_max_hops(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        a = store.add_entity("A", "thing")
        b = store.add_entity("B", "thing")
        c = store.add_entity("C", "thing")
        d = store.add_entity("D", "thing")
        store.add_relationship(a, "uses", b, confidence=0.9, source="test")
        store.add_relationship(b, "uses", c, confidence=0.9, source="test")
        store.add_relationship(c, "uses", d, confidence=0.9, source="test")
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        # Should not infer A->D (3 hops)
        assert not any(r["subject_id"] == a and r["object_id"] == d for r in inferred)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_inference_engine.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/knowledge/reasoning/inference_engine.py
"""Inference engine — derives new relationships from existing graph connections."""

import logging
from typing import Any

from ..graph.store import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# Transitive inference rules: (rel1, rel2) → inferred_relation, confidence_multiplier
_INFERENCE_RULES = [
    (("works_on", "uses"), "works_with", 0.7),
    (("depends_on", "depends_on"), "indirectly_depends_on", 0.5),
    (("member_of", "located_in"), "located_in", 0.4),
    (("uses", "uses"), "indirectly_uses", 0.5),
    (("works_on", "depends_on"), "works_with", 0.6),
]


class InferenceEngine:
    """Derives new relationships from existing graph connections."""

    def __init__(
        self,
        graph_store: KnowledgeGraphStore,
        max_hops: int = 2,
    ) -> None:
        self._graph = graph_store
        self._max_hops = max_hops

    def run_inference(self) -> list[dict[str, Any]]:
        """Run inference rules and return newly created relationships."""
        if self._graph._conn is None:
            return []

        inferred = []

        # Get all current relationships
        all_rels = self._graph._conn.execute(
            "SELECT * FROM kg_relationships WHERE valid_until IS NULL AND source != 'inference'"
        ).fetchall()
        all_rels = [dict(r) for r in all_rels]

        # Build adjacency: {subject_id: [(relation, object_id, confidence)]}
        adj: dict[str, list[tuple[str, str, float]]] = {}
        for r in all_rels:
            adj.setdefault(r["subject_id"], []).append((r["relation"], r["object_id"], r["confidence"]))

        # Apply rules (1-hop transitive)
        for (rel1, rel2), inferred_rel, conf_mult in _INFERENCE_RULES:
            for r in all_rels:
                if r["relation"] != rel1:
                    continue
                # Look for second hop from r's object
                for next_rel, next_obj, next_conf in adj.get(r["object_id"], []):
                    if next_rel != rel2:
                        continue
                    if next_obj == r["subject_id"]:
                        continue  # skip self-loops

                    # Check if inference already exists
                    existing = self._graph.find_current_relationships(r["subject_id"], inferred_rel)
                    if any(e["object_id"] == next_obj for e in existing):
                        continue

                    confidence = min(r["confidence"], next_conf) * conf_mult
                    rid = self._graph.add_relationship(
                        r["subject_id"],
                        inferred_rel,
                        next_obj,
                        confidence=confidence,
                        source="inference",
                    )
                    inferred.append({
                        "id": rid,
                        "subject_id": r["subject_id"],
                        "relation": inferred_rel,
                        "object_id": next_obj,
                        "confidence": confidence,
                        "source": "inference",
                    })

        logger.info("Inference produced %d new relationships", len(inferred))
        return inferred
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_inference_engine.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/reasoning/inference_engine.py tests/unit/knowledge_evolution/test_inference_engine.py
git commit -m "feat(knowledge): add inference engine with transitive rules"
```

---

### Task 12: Contradiction Detector

**Files:**
- Create: `src/homie_core/adaptive_learning/knowledge/reasoning/contradiction_detector.py`
- Test: `tests/unit/knowledge_evolution/test_contradiction_detector.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/knowledge_evolution/test_contradiction_detector.py
import time
import pytest
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore
from homie_core.adaptive_learning.knowledge.reasoning.contradiction_detector import ContradictionDetector


class TestContradictionDetector:
    def _setup(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        return store

    def test_detect_contradiction(self, tmp_path):
        store = self._setup(tmp_path)
        user = store.add_entity("User", "person")
        google = store.add_entity("Google", "organization")
        anthropic = store.add_entity("Anthropic", "organization")
        store.add_relationship(user, "works_at", google, confidence=0.8, source="conversation")
        store.add_relationship(user, "works_at", anthropic, confidence=0.9, source="conversation")
        detector = ContradictionDetector(graph_store=store)
        contradictions = detector.detect()
        assert len(contradictions) >= 1

    def test_no_contradiction_for_different_predicates(self, tmp_path):
        store = self._setup(tmp_path)
        user = store.add_entity("User", "person")
        google = store.add_entity("Google", "organization")
        store.add_relationship(user, "works_at", google, confidence=0.9, source="conv")
        store.add_relationship(user, "admires", google, confidence=0.7, source="conv")
        detector = ContradictionDetector(graph_store=store)
        contradictions = detector.detect()
        assert len(contradictions) == 0

    def test_resolve_by_confidence(self, tmp_path):
        store = self._setup(tmp_path)
        user = store.add_entity("User", "person")
        google = store.add_entity("Google", "organization")
        anthropic = store.add_entity("Anthropic", "organization")
        r1 = store.add_relationship(user, "works_at", google, confidence=0.7, source="conv")
        r2 = store.add_relationship(user, "works_at", anthropic, confidence=0.95, source="conv")
        detector = ContradictionDetector(graph_store=store)
        resolved = detector.resolve_all()
        assert len(resolved) >= 1
        # Lower confidence should be superseded
        google_rels = store.find_current_relationships(user, "works_at")
        assert len(google_rels) == 1
        assert google_rels[0]["object_id"] == anthropic

    def test_no_resolve_needed_when_clean(self, tmp_path):
        store = self._setup(tmp_path)
        user = store.add_entity("User", "person")
        google = store.add_entity("Google", "organization")
        store.add_relationship(user, "works_at", google, confidence=0.9, source="conv")
        detector = ContradictionDetector(graph_store=store)
        resolved = detector.resolve_all()
        assert len(resolved) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_contradiction_detector.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/knowledge/reasoning/contradiction_detector.py
"""Contradiction detector — finds and resolves conflicting facts."""

import logging
import time
from collections import defaultdict
from typing import Any

from ..graph.store import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# Predicates that are typically single-valued (one object at a time)
_SINGLE_VALUED_PREDICATES = {
    "works_at", "lives_in", "located_in", "primary_language",
    "current_role", "reports_to", "managed_by",
}


class ContradictionDetector:
    """Detects and resolves contradictory relationships in the knowledge graph."""

    def __init__(self, graph_store: KnowledgeGraphStore) -> None:
        self._graph = graph_store

    def detect(self) -> list[dict[str, Any]]:
        """Find contradictions — same subject + single-valued predicate with multiple current objects."""
        if self._graph._conn is None:
            return []

        contradictions = []
        current_rels = self._graph._conn.execute(
            "SELECT * FROM kg_relationships WHERE valid_until IS NULL"
        ).fetchall()

        # Group by (subject_id, relation)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in current_rels:
            r = dict(r)
            groups[(r["subject_id"], r["relation"])].append(r)

        for (subj, rel), rels in groups.items():
            if rel not in _SINGLE_VALUED_PREDICATES:
                continue
            if len(rels) <= 1:
                continue
            # Multiple current values for single-valued predicate = contradiction
            contradictions.append({
                "subject_id": subj,
                "relation": rel,
                "conflicting_relationships": rels,
            })

        return contradictions

    def resolve_all(self) -> list[dict]:
        """Detect and resolve all contradictions. Returns list of resolutions."""
        contradictions = self.detect()
        resolutions = []

        for contradiction in contradictions:
            rels = contradiction["conflicting_relationships"]
            # Keep highest confidence, supersede others
            rels_sorted = sorted(rels, key=lambda r: (r["confidence"], r["created_at"]), reverse=True)
            winner = rels_sorted[0]
            now = time.time()

            for loser in rels_sorted[1:]:
                self._graph.update_relationship_valid_until(loser["id"], now)
                resolutions.append({
                    "subject_id": contradiction["subject_id"],
                    "relation": contradiction["relation"],
                    "kept": winner["object_id"],
                    "superseded": loser["object_id"],
                    "reason": "higher_confidence",
                })
                logger.info(
                    "Resolved contradiction: %s %s — kept %s, superseded %s",
                    contradiction["subject_id"], contradiction["relation"],
                    winner["object_id"], loser["object_id"],
                )

        return resolutions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_contradiction_detector.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/reasoning/contradiction_detector.py tests/unit/knowledge_evolution/test_contradiction_detector.py
git commit -m "feat(knowledge): add contradiction detector with auto-resolution"
```

---

## Chunk 4: Integration & Config

### Task 13: Wire Graph into KnowledgeBuilder

**Files:**
- Modify: `src/homie_core/adaptive_learning/knowledge/builder.py`
- Test: `tests/unit/knowledge_evolution/test_builder_integration.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/knowledge_evolution/test_builder_integration.py
import pytest
from homie_core.adaptive_learning.knowledge.builder import KnowledgeBuilder
from homie_core.adaptive_learning.storage import LearningStorage


class TestKnowledgeBuilderGraphIntegration:
    def test_has_graph_store(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")
        assert builder.graph_store is not None

    def test_has_intake_pipeline(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")
        assert builder.intake is not None

    def test_has_graph_query(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")
        assert builder.graph_query is not None

    def test_ingest_source(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")
        # Create a test source
        src_dir = tmp_path / "project"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("class App:\n    pass\n")
        result = builder.ingest_source(src_dir)
        assert result["files_scanned"] >= 1

    def test_process_turn_still_works(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")
        facts = builder.process_turn("I work at Google", "Nice!")
        assert len(facts) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_builder_integration.py -v`
Expected: FAIL — KnowledgeBuilder doesn't accept graph_db_path

- [ ] **Step 3: Modify builder.py**

Read `src/homie_core/adaptive_learning/knowledge/builder.py` first. Update the `__init__` to accept `graph_db_path` and wire in the graph store, intake pipeline, and query interface:

```python
# src/homie_core/adaptive_learning/knowledge/builder.py
"""KnowledgeBuilder — coordinates conversation mining, project tracking, profiling, and knowledge graph."""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from ..observation.signals import LearningSignal
from ..storage import LearningStorage
from .behavioral_profiler import BehavioralProfiler
from .conversation_miner import ConversationMiner
from .graph.query import GraphQuery
from .graph.store import KnowledgeGraphStore
from .intake.pipeline import IntakePipeline
from .project_tracker import ProjectTracker

logger = logging.getLogger(__name__)


class KnowledgeBuilder:
    """Coordinates all knowledge-building engines including the knowledge graph."""

    def __init__(
        self,
        storage: LearningStorage,
        inference_fn: Optional[Callable[[str], str]] = None,
        graph_db_path: Optional[Path | str] = None,
    ) -> None:
        self._storage = storage
        self.miner = ConversationMiner(storage=storage, inference_fn=inference_fn)
        self.project_tracker = ProjectTracker(storage=storage)
        self.profiler = BehavioralProfiler()

        # Knowledge graph (optional — graceful if no path provided)
        self.graph_store: Optional[KnowledgeGraphStore] = None
        self.graph_query: Optional[GraphQuery] = None
        self.intake: Optional[IntakePipeline] = None

        if graph_db_path:
            self.graph_store = KnowledgeGraphStore(db_path=graph_db_path)
            self.graph_store.initialize()
            self.graph_query = GraphQuery(store=self.graph_store)
            self.intake = IntakePipeline(
                graph_store=self.graph_store,
                inference_fn=inference_fn,
            )

    def on_signal(self, signal: LearningSignal) -> None:
        """Process knowledge-related signals."""
        data = signal.data
        if "hour" in data:
            hour = data["hour"]
            for key in ("app", "activity", "topic"):
                if key in data:
                    self.profiler.record_observation(hour, key, data[key])

    def process_turn(self, user_message: str, response: str) -> list[str]:
        """Process a conversation turn for knowledge extraction."""
        return self.miner.process_turn(user_message, response)

    def ingest_source(self, source: Path | str) -> dict[str, Any]:
        """Ingest a directory or file into the knowledge graph."""
        if self.intake is None:
            return {"error": "Knowledge graph not initialized", "files_scanned": 0}
        return self.intake.ingest(source)

    def get_work_hours(self) -> list[int]:
        """Get detected work hours."""
        return self.profiler.get_work_hours()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_builder_integration.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run existing KnowledgeBuilder tests to ensure no regression**

Run: `python -m pytest tests/unit/adaptive_learning/test_knowledge_builder.py -v`
Expected: All existing tests PASS (graph_db_path is optional, defaults to None)

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/adaptive_learning/knowledge/builder.py tests/unit/knowledge_evolution/test_builder_integration.py
git commit -m "feat(knowledge): wire graph store and intake pipeline into KnowledgeBuilder"
```

---

### Task 14: Config & Boot Integration

**Files:**
- Modify: `src/homie_core/config.py`
- Modify: `homie.config.yaml`
- Modify: `src/homie_core/adaptive_learning/learner.py`
- Test: `tests/unit/knowledge_evolution/test_kg_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/knowledge_evolution/test_kg_config.py
import pytest
from homie_core.config import KnowledgeGraphConfig


class TestKnowledgeGraphConfig:
    def test_defaults(self):
        cfg = KnowledgeGraphConfig()
        assert cfg.enabled is True
        assert cfg.intake.surface_pass is True
        assert cfg.intake.deep_pass is True
        assert cfg.intake.deep_pass_top_percent == 20
        assert cfg.reasoning.entity_resolution is True
        assert cfg.reasoning.max_inference_hops == 2
        assert cfg.temporal.confidence_decay_rate == 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/knowledge_evolution/test_kg_config.py -v`
Expected: FAIL — KnowledgeGraphConfig not found

- [ ] **Step 3: Add config classes and update files**

Read `src/homie_core/config.py`. Add before HomieConfig:

```python
class IntakeConfig(BaseModel):
    surface_pass: bool = True
    deep_pass: bool = True
    deep_pass_top_percent: int = 20
    max_deep_files_per_batch: int = 50

class ReasoningConfig(BaseModel):
    entity_resolution: bool = True
    relationship_inference: bool = True
    max_inference_hops: int = 2
    inference_batch_interval: int = 300

class TemporalConfig(BaseModel):
    confidence_decay_rate: float = 0.99
    contradiction_auto_resolve: bool = True

class KnowledgeGraphConfig(BaseModel):
    enabled: bool = True
    intake: IntakeConfig = Field(default_factory=IntakeConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
```

Add to HomieConfig: `knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)`

Add `knowledge_graph:` section to `homie.config.yaml` with matching defaults.

Update `src/homie_core/adaptive_learning/learner.py` to pass `graph_db_path` when constructing KnowledgeBuilder:

```python
# In AdaptiveLearner.__init__, change:
self.knowledge_builder = KnowledgeBuilder(storage=self._storage)
# To:
graph_path = Path(db_path).parent / "knowledge_graph.db" if graph_db_path is None else graph_db_path
self.knowledge_builder = KnowledgeBuilder(
    storage=self._storage,
    graph_db_path=graph_path,
)
```

Add `graph_db_path: Optional[Path | str] = None` parameter to `AdaptiveLearner.__init__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/knowledge_evolution/test_kg_config.py -v`
Expected: PASS

- [ ] **Step 5: Run all tests to check for regressions**

Run: `python -m pytest tests/unit/adaptive_learning/ tests/unit/knowledge_evolution/ tests/integration/ -q --tb=short`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/config.py homie.config.yaml src/homie_core/adaptive_learning/learner.py tests/unit/knowledge_evolution/test_kg_config.py
git commit -m "feat(knowledge): add KnowledgeGraphConfig and wire graph into AdaptiveLearner"
```

---

### Task 15: Integration Test

**Files:**
- Create: `tests/integration/test_knowledge_evolution_lifecycle.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_knowledge_evolution_lifecycle.py
"""Integration test: knowledge evolution lifecycle."""
import pytest
from pathlib import Path
from homie_core.adaptive_learning.knowledge.builder import KnowledgeBuilder
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore
from homie_core.adaptive_learning.knowledge.reasoning.entity_resolver import EntityResolver
from homie_core.adaptive_learning.knowledge.reasoning.inference_engine import InferenceEngine
from homie_core.adaptive_learning.knowledge.reasoning.contradiction_detector import ContradictionDetector
from homie_core.adaptive_learning.storage import LearningStorage


class TestKnowledgeEvolutionLifecycle:
    def test_intake_creates_graph_entities(self, tmp_path):
        """Guided intake populates the knowledge graph."""
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")

        # Create source files
        src = tmp_path / "project"
        src.mkdir()
        (src / "main.py").write_text("class Application:\n    pass\n\nclass Database:\n    pass\n")
        (src / "utils.py").write_text("import os\ndef helper():\n    return True\n")

        result = builder.ingest_source(src)
        assert result["files_scanned"] == 2
        assert builder.graph_store.entity_count() > 0

    def test_inference_derives_relationships(self, tmp_path):
        """Inference engine derives new facts from existing ones."""
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        user = store.add_entity("User", "person")
        homie = store.add_entity("Homie", "project")
        python = store.add_entity("Python", "technology")
        store.add_relationship(user, "works_on", homie, confidence=0.95, source="conversation")
        store.add_relationship(homie, "uses", python, confidence=0.9, source="code_scan")

        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        assert len(inferred) >= 1  # User works_with Python

    def test_contradiction_resolution_with_temporal_versioning(self, tmp_path):
        """Contradictions are resolved via temporal versioning."""
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        user = store.add_entity("User", "person")
        google = store.add_entity("Google", "organization")
        anthropic = store.add_entity("Anthropic", "organization")
        store.add_relationship(user, "works_at", google, confidence=0.7, source="old_conv")
        store.add_relationship(user, "works_at", anthropic, confidence=0.95, source="recent_conv")

        detector = ContradictionDetector(graph_store=store)
        resolutions = detector.resolve_all()
        assert len(resolutions) == 1

        # Only one current works_at relationship
        current = store.find_current_relationships(user, "works_at")
        assert len(current) == 1
        assert current[0]["object_id"] == anthropic

        # Old relationship still exists but superseded
        all_rels = store.get_relationships(subject_id=user)
        assert len(all_rels) == 2  # both exist

    def test_full_lifecycle(self, tmp_path):
        """Full flow: intake → inference → contradiction resolution."""
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()

        # Seed some entities
        user = store.add_entity("Developer", "person")
        project = store.add_entity("MyApp", "project")
        store.add_relationship(user, "works_on", project, confidence=0.9, source="conversation")

        # Add technology via code scan
        python = store.add_entity("Python", "technology")
        store.add_relationship(project, "uses", python, confidence=0.85, source="code_scan")

        # Run inference
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        assert len(inferred) >= 1

        # Verify graph is queryable
        from homie_core.adaptive_learning.knowledge.graph.query import GraphQuery
        query = GraphQuery(store=store)
        related = query.traverse(user, max_hops=2)
        names = [e["name"] for e in related]
        assert "MyApp" in names
        assert "Python" in names
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/integration/test_knowledge_evolution_lifecycle.py -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Run ALL tests**

Run: `python -m pytest tests/unit/self_healing/ tests/unit/adaptive_learning/ tests/unit/knowledge_evolution/ tests/integration/ -q --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_knowledge_evolution_lifecycle.py
git commit -m "test(knowledge): add integration tests for knowledge evolution lifecycle"
```
