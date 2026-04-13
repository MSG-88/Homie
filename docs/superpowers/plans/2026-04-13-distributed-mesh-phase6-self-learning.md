# Distributed Mesh — Phase 6: Self-Learning Loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Homie collects implicit feedback signals from every interaction across all mesh nodes, aggregates them on the Hub, and triggers fine-tuning when enough data accumulates. Updated models propagate to all Spokes via mesh events.

**Architecture:** `FeedbackCollector` captures implicit signals (accepted, regenerated, corrected, ignored) per interaction. `FeedbackStore` persists signals in SQLite. `TrainingTrigger` monitors accumulated feedback and decides when to trigger a fine-tuning cycle. `ModelDistributor` notifies Spokes when new models are available. All signals sync as mesh events.

**Tech Stack:** Python 3.11+, SQLite, existing mesh events, existing finetune pipeline

**Builds on:** Phase 1-5 (mesh identity, events, sync, inference, context, tasks)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/homie_core/mesh/feedback_collector.py` | Capture implicit learning signals from interactions |
| `src/homie_core/mesh/feedback_store.py` | SQLite storage for feedback signals |
| `src/homie_core/mesh/training_trigger.py` | Monitor feedback and decide when to fine-tune |
| `src/homie_core/mesh/model_distributor.py` | Notify mesh nodes about new models |
| `tests/unit/test_mesh/test_feedback_collector.py` | Collector tests |
| `tests/unit/test_mesh/test_feedback_store.py` | Store tests |
| `tests/unit/test_mesh/test_training_trigger.py` | Trigger tests |
| `tests/unit/test_mesh/test_model_distributor.py` | Distributor tests |
| `tests/integration/test_self_learning_loop.py` | End-to-end learning loop test |

---

### Task 1: Feedback Collector — Capture Learning Signals

**Files:**
- Create: `src/homie_core/mesh/feedback_collector.py`
- Test: `tests/unit/test_mesh/test_feedback_collector.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_feedback_collector.py
from homie_core.mesh.feedback_collector import FeedbackCollector, FeedbackSignal, SignalType


def test_signal_types():
    assert SignalType.ACCEPTED == "accepted"
    assert SignalType.REGENERATED == "regenerated"
    assert SignalType.CORRECTED == "corrected"
    assert SignalType.IGNORED == "ignored"
    assert SignalType.RATED == "rated"


def test_create_signal():
    sig = FeedbackSignal(
        signal_type=SignalType.ACCEPTED,
        query="What is Python?",
        response_preview="Python is a programming language...",
        node_id="desktop",
        activity_context="coding",
    )
    assert sig.signal_id  # Auto-generated
    assert sig.signal_type == "accepted"
    assert sig.timestamp != ""


def test_signal_to_dict_roundtrip():
    sig = FeedbackSignal(
        signal_type=SignalType.CORRECTED,
        query="fix this", response_preview="here's the fix",
        node_id="laptop", activity_context="debugging",
    )
    d = sig.to_dict()
    restored = FeedbackSignal.from_dict(d)
    assert restored.signal_id == sig.signal_id
    assert restored.signal_type == "corrected"


def test_collector_record_accepted():
    collector = FeedbackCollector(node_id="desktop")
    sig = collector.record_accepted(query="hello", response="Hi there!")
    assert sig.signal_type == SignalType.ACCEPTED
    assert sig.response_preview == "Hi there!"
    assert len(collector.signals) == 1


def test_collector_record_regenerated():
    collector = FeedbackCollector(node_id="desktop")
    sig = collector.record_regenerated(query="explain X", original="bad", regenerated="better")
    assert sig.signal_type == SignalType.REGENERATED
    assert len(collector.signals) == 1


def test_collector_record_corrected():
    collector = FeedbackCollector(node_id="desktop")
    sig = collector.record_corrected(query="what is 2+2?", original="5", correction="4")
    assert sig.signal_type == SignalType.CORRECTED
    assert "correction" in sig.metadata


def test_collector_record_ignored():
    collector = FeedbackCollector(node_id="desktop")
    sig = collector.record_ignored(query="random stuff", response="...")
    assert sig.signal_type == SignalType.IGNORED


def test_collector_flush():
    collector = FeedbackCollector(node_id="desktop")
    collector.record_accepted(query="q1", response="r1")
    collector.record_accepted(query="q2", response="r2")
    flushed = collector.flush()
    assert len(flushed) == 2
    assert len(collector.signals) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_feedback_collector.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/feedback_collector.py
"""Feedback collector — captures implicit learning signals from interactions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from homie_core.mesh.events import generate_ulid
from homie_core.utils import utc_now


class SignalType:
    ACCEPTED = "accepted"
    REGENERATED = "regenerated"
    CORRECTED = "corrected"
    IGNORED = "ignored"
    RATED = "rated"


@dataclass
class FeedbackSignal:
    """A single implicit feedback signal from an interaction."""

    signal_type: str
    query: str
    response_preview: str
    node_id: str
    activity_context: str = ""
    signal_id: str = field(default_factory=generate_ulid)
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())
    rating: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id, "signal_type": self.signal_type,
            "query": self.query, "response_preview": self.response_preview,
            "node_id": self.node_id, "activity_context": self.activity_context,
            "timestamp": self.timestamp, "rating": self.rating,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeedbackSignal:
        return cls(
            signal_id=d["signal_id"], signal_type=d["signal_type"],
            query=d["query"], response_preview=d["response_preview"],
            node_id=d["node_id"], activity_context=d.get("activity_context", ""),
            timestamp=d.get("timestamp", ""), rating=d.get("rating"),
            metadata=d.get("metadata", {}),
        )


class FeedbackCollector:
    """Collects feedback signals from user interactions on a single node."""

    def __init__(self, node_id: str):
        self._node_id = node_id
        self._signals: list[FeedbackSignal] = []

    @property
    def signals(self) -> list[FeedbackSignal]:
        return list(self._signals)

    def record_accepted(self, query: str, response: str) -> FeedbackSignal:
        sig = FeedbackSignal(
            signal_type=SignalType.ACCEPTED,
            query=query, response_preview=response[:200],
            node_id=self._node_id,
        )
        self._signals.append(sig)
        return sig

    def record_regenerated(
        self, query: str, original: str, regenerated: str,
    ) -> FeedbackSignal:
        sig = FeedbackSignal(
            signal_type=SignalType.REGENERATED,
            query=query, response_preview=regenerated[:200],
            node_id=self._node_id,
            metadata={"original_preview": original[:200]},
        )
        self._signals.append(sig)
        return sig

    def record_corrected(
        self, query: str, original: str, correction: str,
    ) -> FeedbackSignal:
        sig = FeedbackSignal(
            signal_type=SignalType.CORRECTED,
            query=query, response_preview=original[:200],
            node_id=self._node_id,
            metadata={"correction": correction[:200]},
        )
        self._signals.append(sig)
        return sig

    def record_ignored(self, query: str, response: str) -> FeedbackSignal:
        sig = FeedbackSignal(
            signal_type=SignalType.IGNORED,
            query=query, response_preview=response[:200],
            node_id=self._node_id,
        )
        self._signals.append(sig)
        return sig

    def flush(self) -> list[FeedbackSignal]:
        """Return all collected signals and clear the buffer."""
        result = list(self._signals)
        self._signals.clear()
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_feedback_collector.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/feedback_collector.py tests/unit/test_mesh/test_feedback_collector.py
git commit -m "feat(mesh): add FeedbackCollector for implicit learning signal capture"
```

---

### Task 2: Feedback Store — SQLite Persistence

**Files:**
- Create: `src/homie_core/mesh/feedback_store.py`
- Test: `tests/unit/test_mesh/test_feedback_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_feedback_store.py
from homie_core.mesh.feedback_collector import FeedbackSignal, SignalType
from homie_core.mesh.feedback_store import FeedbackStore


def test_store_initialize(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    store.initialize()


def test_save_and_get(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    store.initialize()
    sig = FeedbackSignal(signal_type=SignalType.ACCEPTED, query="hi",
                         response_preview="hello", node_id="n1")
    store.save(sig)
    loaded = store.get(sig.signal_id)
    assert loaded is not None
    assert loaded.signal_type == "accepted"
    assert loaded.query == "hi"


def test_count_by_type(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    store.initialize()
    for _ in range(3):
        store.save(FeedbackSignal(signal_type=SignalType.ACCEPTED, query="q",
                                  response_preview="r", node_id="n1"))
    for _ in range(2):
        store.save(FeedbackSignal(signal_type=SignalType.CORRECTED, query="q",
                                  response_preview="r", node_id="n1"))
    counts = store.count_by_type()
    assert counts["accepted"] == 3
    assert counts["corrected"] == 2


def test_total_count(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    store.initialize()
    assert store.total_count() == 0
    for i in range(5):
        store.save(FeedbackSignal(signal_type=SignalType.ACCEPTED, query=f"q{i}",
                                  response_preview="r", node_id="n1"))
    assert store.total_count() == 5


def test_signals_since(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    store.initialize()
    import time
    sigs = []
    for i in range(4):
        s = FeedbackSignal(signal_type=SignalType.ACCEPTED, query=f"q{i}",
                           response_preview="r", node_id="n1")
        store.save(s)
        sigs.append(s)
        time.sleep(0.002)
    since = store.signals_since(sigs[1].signal_id, limit=100)
    assert len(since) == 2


def test_get_training_pairs(tmp_path):
    """Corrections produce training pairs (query, bad_response, correction)."""
    store = FeedbackStore(tmp_path / "feedback.db")
    store.initialize()
    store.save(FeedbackSignal(
        signal_type=SignalType.CORRECTED, query="what is 2+2?",
        response_preview="5", node_id="n1",
        metadata={"correction": "4"},
    ))
    store.save(FeedbackSignal(
        signal_type=SignalType.ACCEPTED, query="hello",
        response_preview="Hi!", node_id="n1",
    ))
    pairs = store.get_training_pairs()
    assert len(pairs) == 2
    # Correction pair
    correction_pair = [p for p in pairs if p["type"] == "dpo"][0]
    assert correction_pair["query"] == "what is 2+2?"
    assert correction_pair["rejected"] == "5"
    assert correction_pair["chosen"] == "4"
    # Accepted pair
    accepted_pair = [p for p in pairs if p["type"] == "sft"][0]
    assert accepted_pair["query"] == "hello"
    assert accepted_pair["response"] == "Hi!"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_feedback_store.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/feedback_store.py
"""Feedback store — SQLite persistence for learning signals."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from homie_core.mesh.feedback_collector import FeedbackSignal, SignalType


class FeedbackStore:
    """SQLite store for feedback signals used in self-learning."""

    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback_signals (
                signal_id TEXT PRIMARY KEY,
                signal_type TEXT NOT NULL,
                query TEXT NOT NULL,
                response_preview TEXT NOT NULL,
                node_id TEXT NOT NULL,
                activity_context TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                rating REAL,
                metadata_json TEXT DEFAULT '{}'
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback_signals(signal_type)"
        )
        self._conn.commit()

    def save(self, signal: FeedbackSignal) -> None:
        self._conn.execute("""
            INSERT OR IGNORE INTO feedback_signals
                (signal_id, signal_type, query, response_preview, node_id,
                 activity_context, timestamp, rating, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.signal_id, signal.signal_type, signal.query,
            signal.response_preview, signal.node_id, signal.activity_context,
            signal.timestamp, signal.rating, json.dumps(signal.metadata),
        ))
        self._conn.commit()

    def get(self, signal_id: str) -> Optional[FeedbackSignal]:
        row = self._conn.execute(
            "SELECT * FROM feedback_signals WHERE signal_id = ?", (signal_id,)
        ).fetchone()
        return self._row_to_signal(row) if row else None

    def total_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM feedback_signals").fetchone()[0]

    def count_by_type(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT signal_type, COUNT(*) as cnt FROM feedback_signals GROUP BY signal_type"
        ).fetchall()
        return {row["signal_type"]: row["cnt"] for row in rows}

    def signals_since(self, after_signal_id: str, limit: int = 1000) -> list[FeedbackSignal]:
        rows = self._conn.execute(
            "SELECT * FROM feedback_signals WHERE signal_id > ? ORDER BY signal_id ASC LIMIT ?",
            (after_signal_id, limit),
        ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def get_training_pairs(self) -> list[dict]:
        """Extract training pairs from feedback signals.

        - CORRECTED -> DPO pair (query, rejected=original, chosen=correction)
        - ACCEPTED -> SFT pair (query, response)
        """
        pairs = []
        rows = self._conn.execute(
            "SELECT * FROM feedback_signals WHERE signal_type IN (?, ?) ORDER BY signal_id",
            (SignalType.CORRECTED, SignalType.ACCEPTED),
        ).fetchall()

        for row in rows:
            sig = self._row_to_signal(row)
            if sig.signal_type == SignalType.CORRECTED:
                pairs.append({
                    "type": "dpo",
                    "query": sig.query,
                    "rejected": sig.response_preview,
                    "chosen": sig.metadata.get("correction", ""),
                })
            elif sig.signal_type == SignalType.ACCEPTED:
                pairs.append({
                    "type": "sft",
                    "query": sig.query,
                    "response": sig.response_preview,
                })
        return pairs

    @staticmethod
    def _row_to_signal(row: sqlite3.Row) -> FeedbackSignal:
        return FeedbackSignal(
            signal_id=row["signal_id"], signal_type=row["signal_type"],
            query=row["query"], response_preview=row["response_preview"],
            node_id=row["node_id"], activity_context=row["activity_context"],
            timestamp=row["timestamp"], rating=row["rating"],
            metadata=json.loads(row["metadata_json"]),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_feedback_store.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/feedback_store.py tests/unit/test_mesh/test_feedback_store.py
git commit -m "feat(mesh): add FeedbackStore for persistent learning signal storage"
```

---

### Task 3: Training Trigger — Automated Fine-Tune Decision

**Files:**
- Create: `src/homie_core/mesh/training_trigger.py`
- Test: `tests/unit/test_mesh/test_training_trigger.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_training_trigger.py
from homie_core.mesh.feedback_collector import FeedbackSignal, SignalType
from homie_core.mesh.feedback_store import FeedbackStore
from homie_core.mesh.training_trigger import TrainingTrigger


def test_not_ready_when_empty(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    store.initialize()
    trigger = TrainingTrigger(feedback_store=store)
    assert trigger.should_trigger() is False


def test_triggers_at_signal_threshold(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    store.initialize()
    trigger = TrainingTrigger(feedback_store=store, min_signals=10)
    for i in range(10):
        store.save(FeedbackSignal(signal_type=SignalType.ACCEPTED, query=f"q{i}",
                                  response_preview="r", node_id="n1"))
    assert trigger.should_trigger() is True


def test_not_ready_below_threshold(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    store.initialize()
    trigger = TrainingTrigger(feedback_store=store, min_signals=100)
    for i in range(5):
        store.save(FeedbackSignal(signal_type=SignalType.ACCEPTED, query=f"q{i}",
                                  response_preview="r", node_id="n1"))
    assert trigger.should_trigger() is False


def test_triggers_at_correction_threshold(tmp_path):
    """Many corrections trigger even before total threshold."""
    store = FeedbackStore(tmp_path / "fb.db")
    store.initialize()
    trigger = TrainingTrigger(feedback_store=store, min_signals=1000, min_corrections=5)
    for i in range(5):
        store.save(FeedbackSignal(signal_type=SignalType.CORRECTED, query=f"q{i}",
                                  response_preview="bad", node_id="n1",
                                  metadata={"correction": "good"}))
    assert trigger.should_trigger() is True


def test_get_training_summary(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    store.initialize()
    for i in range(3):
        store.save(FeedbackSignal(signal_type=SignalType.ACCEPTED, query=f"q{i}",
                                  response_preview="r", node_id="n1"))
    store.save(FeedbackSignal(signal_type=SignalType.CORRECTED, query="fix",
                              response_preview="bad", node_id="n1",
                              metadata={"correction": "good"}))
    trigger = TrainingTrigger(feedback_store=store, min_signals=2)
    summary = trigger.get_summary()
    assert summary["total_signals"] == 4
    assert summary["sft_pairs"] >= 3
    assert summary["dpo_pairs"] >= 1
    assert summary["ready"] is True


def test_mark_triggered(tmp_path):
    """After triggering, should_trigger returns False until new signals arrive."""
    store = FeedbackStore(tmp_path / "fb.db")
    store.initialize()
    trigger = TrainingTrigger(feedback_store=store, min_signals=3)
    for i in range(3):
        store.save(FeedbackSignal(signal_type=SignalType.ACCEPTED, query=f"q{i}",
                                  response_preview="r", node_id="n1"))
    assert trigger.should_trigger() is True
    trigger.mark_triggered()
    assert trigger.should_trigger() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_training_trigger.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/training_trigger.py
"""Training trigger — decides when to kick off a fine-tuning cycle."""
from __future__ import annotations

from homie_core.mesh.feedback_store import FeedbackStore


class TrainingTrigger:
    """Monitors feedback signals and decides when to trigger fine-tuning.

    Trigger conditions (any one):
    - Total signals >= min_signals
    - Corrections >= min_corrections
    """

    def __init__(
        self,
        feedback_store: FeedbackStore,
        min_signals: int = 500,
        min_corrections: int = 100,
    ):
        self._store = feedback_store
        self._min_signals = min_signals
        self._min_corrections = min_corrections
        self._last_triggered_count: int = 0

    def should_trigger(self) -> bool:
        """Check if enough new signals have accumulated to trigger training."""
        total = self._store.total_count()
        new_signals = total - self._last_triggered_count
        if new_signals <= 0:
            return False

        if new_signals >= self._min_signals:
            return True

        counts = self._store.count_by_type()
        corrections = counts.get("corrected", 0)
        # Only count corrections above last triggered baseline
        if corrections >= self._min_corrections:
            return True

        return False

    def mark_triggered(self) -> None:
        """Record that training was triggered at the current signal count."""
        self._last_triggered_count = self._store.total_count()

    def get_summary(self) -> dict:
        """Get a summary of training readiness."""
        total = self._store.total_count()
        counts = self._store.count_by_type()
        pairs = self._store.get_training_pairs()
        sft = sum(1 for p in pairs if p["type"] == "sft")
        dpo = sum(1 for p in pairs if p["type"] == "dpo")
        return {
            "total_signals": total,
            "by_type": counts,
            "sft_pairs": sft,
            "dpo_pairs": dpo,
            "new_since_last": total - self._last_triggered_count,
            "ready": self.should_trigger(),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_training_trigger.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/training_trigger.py tests/unit/test_mesh/test_training_trigger.py
git commit -m "feat(mesh): add TrainingTrigger for automated fine-tuning decisions"
```

---

### Task 4: Model Distributor — Notify Mesh of New Models

**Files:**
- Create: `src/homie_core/mesh/model_distributor.py`
- Test: `tests/unit/test_mesh/test_model_distributor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_model_distributor.py
from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.model_distributor import ModelDistributor


def test_announce_model_update(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    dist = ModelDistributor(mesh_manager=mgr)

    dist.announce_update(
        model_name="homie-v2",
        model_path="/models/homie-v2.gguf",
        score_improvement=0.05,
        cycle=3,
    )
    events = mgr.events_since(None)
    assert len(events) == 1
    assert events[0].category == "learning"
    assert events[0].event_type == "model_updated"
    assert events[0].payload["model_name"] == "homie-v2"
    assert events[0].payload["score_improvement"] == 0.05


def test_announce_training_started(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    dist = ModelDistributor(mesh_manager=mgr)

    dist.announce_training_started(cycle=4, sft_pairs=500, dpo_pairs=100)
    events = mgr.events_since(None)
    assert events[0].event_type == "training_started"
    assert events[0].payload["sft_pairs"] == 500


def test_announce_training_completed(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    dist = ModelDistributor(mesh_manager=mgr)

    dist.announce_training_completed(cycle=4, score=0.82, promoted=True)
    events = mgr.events_since(None)
    assert events[0].event_type == "training_completed"
    assert events[0].payload["promoted"] is True


def test_get_model_history(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    dist = ModelDistributor(mesh_manager=mgr)

    dist.announce_update(model_name="v1", model_path="/v1", score_improvement=0.03, cycle=1)
    dist.announce_update(model_name="v2", model_path="/v2", score_improvement=0.05, cycle=2)

    history = dist.get_model_history()
    assert len(history) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_model_distributor.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/model_distributor.py
"""Model distributor — announces model updates across the mesh."""
from __future__ import annotations

from homie_core.mesh.events import HomieEvent
from homie_core.mesh.mesh_manager import MeshManager


class ModelDistributor:
    """Emits learning events when models are trained or updated."""

    def __init__(self, mesh_manager: MeshManager):
        self._mgr = mesh_manager

    def announce_update(
        self,
        model_name: str,
        model_path: str,
        score_improvement: float,
        cycle: int,
    ) -> HomieEvent:
        """Announce that a new model version is available."""
        return self._mgr.emit("learning", "model_updated", {
            "model_name": model_name,
            "model_path": model_path,
            "score_improvement": score_improvement,
            "cycle": cycle,
        })

    def announce_training_started(
        self, cycle: int, sft_pairs: int, dpo_pairs: int,
    ) -> HomieEvent:
        return self._mgr.emit("learning", "training_started", {
            "cycle": cycle,
            "sft_pairs": sft_pairs,
            "dpo_pairs": dpo_pairs,
        })

    def announce_training_completed(
        self, cycle: int, score: float, promoted: bool,
    ) -> HomieEvent:
        return self._mgr.emit("learning", "training_completed", {
            "cycle": cycle,
            "score": score,
            "promoted": promoted,
        })

    def get_model_history(self) -> list[HomieEvent]:
        """Get all model update events."""
        events = self._mgr._event_store.events_by_category("learning")
        return [e for e in events if e.event_type == "model_updated"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_model_distributor.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/model_distributor.py tests/unit/test_mesh/test_model_distributor.py
git commit -m "feat(mesh): add ModelDistributor for mesh-wide model update announcements"
```

---

### Task 5: Integration Test — Full Self-Learning Loop

**Files:**
- Create: `tests/integration/test_self_learning_loop.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_self_learning_loop.py
"""End-to-end: feedback collection -> trigger check -> training announcement -> sync."""
import time

from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.feedback_collector import FeedbackCollector, SignalType
from homie_core.mesh.feedback_store import FeedbackStore
from homie_core.mesh.training_trigger import TrainingTrigger
from homie_core.mesh.model_distributor import ModelDistributor
from homie_core.mesh.sync_protocol import SyncRequest


def test_full_learning_loop(tmp_path):
    """Collect feedback on Spoke, sync to Hub, trigger training, announce model."""
    # Setup Hub
    hub_id = NodeIdentity.generate()
    hub_mgr = MeshManager(identity=hub_id, data_dir=tmp_path / "hub")

    # Setup Spoke
    spoke_id = NodeIdentity.generate()
    spoke_mgr = MeshManager(identity=spoke_id, data_dir=tmp_path / "spoke")

    # 1. Spoke collects feedback
    collector = FeedbackCollector(node_id=spoke_id.node_id)
    collector.record_accepted(query="What is Python?", response="A programming language")
    collector.record_corrected(query="2+2?", original="5", correction="4")
    collector.record_accepted(query="Hello", response="Hi there!")

    # 2. Spoke persists and emits as events
    spoke_store = FeedbackStore(tmp_path / "spoke" / "feedback.db")
    spoke_store.initialize()
    for sig in collector.flush():
        spoke_store.save(sig)
        spoke_mgr.emit("preference", "feedback_signal", sig.to_dict())

    assert spoke_mgr.event_count() == 3

    # 3. Hub syncs Spoke events
    req = SyncRequest(node_id=hub_id.node_id, last_event_id=None, vector_clock={})
    resp = spoke_mgr.handle_sync_request(req)
    hub_mgr.apply_sync_response(resp)
    assert hub_mgr.event_count() == 3

    # 4. Hub checks training trigger
    hub_store = FeedbackStore(tmp_path / "hub" / "feedback.db")
    hub_store.initialize()
    # Reconstruct signals from events
    for evt in hub_mgr.events_since(None):
        if evt.event_type == "feedback_signal":
            from homie_core.mesh.feedback_collector import FeedbackSignal
            sig = FeedbackSignal.from_dict(evt.payload)
            hub_store.save(sig)

    trigger = TrainingTrigger(feedback_store=hub_store, min_signals=3)
    assert trigger.should_trigger() is True

    # 5. Hub announces training
    distributor = ModelDistributor(mesh_manager=hub_mgr)
    distributor.announce_training_started(cycle=1, sft_pairs=2, dpo_pairs=1)
    distributor.announce_training_completed(cycle=1, score=0.85, promoted=True)
    distributor.announce_update(model_name="homie-v2", model_path="/models/v2.gguf",
                                score_improvement=0.05, cycle=1)

    # 6. Verify model history
    history = distributor.get_model_history()
    assert len(history) == 1
    assert history[0].payload["model_name"] == "homie-v2"

    # 7. Training events sync to Spoke
    req2 = SyncRequest(node_id=spoke_id.node_id, last_event_id=None, vector_clock={})
    resp2 = hub_mgr.handle_sync_request(req2)
    spoke_mgr.apply_sync_response(resp2)
    # Spoke now has: 3 feedback + 3 learning events = 6 total
    spoke_events = spoke_mgr.events_since(None)
    learning_events = [e for e in spoke_events if e.category == "learning"]
    assert len(learning_events) == 3
    assert any(e.event_type == "model_updated" for e in learning_events)

    trigger.mark_triggered()
    assert trigger.should_trigger() is False


def test_training_pairs_extraction(tmp_path):
    """Feedback store correctly extracts SFT and DPO training pairs."""
    store = FeedbackStore(tmp_path / "feedback.db")
    store.initialize()

    collector = FeedbackCollector(node_id="test")
    for i in range(5):
        store.save(collector.record_accepted(query=f"q{i}", response=f"good {i}"))
    for i in range(3):
        store.save(collector.record_corrected(
            query=f"fix{i}", original=f"bad{i}", correction=f"good{i}"))

    pairs = store.get_training_pairs()
    sft = [p for p in pairs if p["type"] == "sft"]
    dpo = [p for p in pairs if p["type"] == "dpo"]
    assert len(sft) == 5
    assert len(dpo) == 3
    assert all(p["response"].startswith("good") for p in sft)
    assert all(p["chosen"].startswith("good") for p in dpo)
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_self_learning_loop.py -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Run ALL mesh tests**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/ tests/unit/test_platform/ tests/unit/test_network/test_discovery_mesh.py tests/integration/test_mesh_smoke.py tests/integration/test_mesh_sync.py tests/integration/test_distributed_inference.py tests/integration/test_cross_device_flow.py tests/integration/test_distributed_tasks.py tests/integration/test_self_learning_loop.py -v`

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_self_learning_loop.py
git commit -m "feat(mesh): add self-learning loop integration tests"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | Feedback Collector | `mesh/feedback_collector.py` | 8 |
| 2 | Feedback Store | `mesh/feedback_store.py` | 6 |
| 3 | Training Trigger | `mesh/training_trigger.py` | 6 |
| 4 | Model Distributor | `mesh/model_distributor.py` | 4 |
| 5 | Integration Tests | `test_self_learning_loop.py` | 2 |

**Total: 5 tasks, 26 tests, 4 new source files, 5 new test files**
