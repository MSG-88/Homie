"""Homie Plugin: Skill Internalization via Local Reinforcement Learning

Inspired by SKILL0 (arXiv:2604.02268), this plugin enables Homie to internalize
frequently-used skills into a local fine-tuned model checkpoint rather than
retrieving and injecting them at inference time. It tracks skill usage patterns,
collects outcome signals (success/failure/user-corrections), and when a skill
crosses a confidence threshold, produces a LoRA training dataset that can be
applied to the local GGUF model via llama.cpp's train adapter tooling.

This reduces token overhead and retrieval noise by baking proven procedural
knowledge directly into model weights.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DB_NAME = "skill_internalization.db"
DEFAULT_MIN_EPISODES = 10
DEFAULT_SUCCESS_THRESHOLD = 0.85
DEFAULT_EXPORT_DIR = "skill_training_data"

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS skills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    description TEXT,
    prompt_template TEXT,
    created_at  REAL,
    updated_at  REAL,
    internalized INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id    INTEGER NOT NULL,
    input_text  TEXT,
    output_text TEXT,
    reward      REAL DEFAULT 0.0,
    signal      TEXT DEFAULT 'neutral',
    correction  TEXT,
    recorded_at REAL,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);

CREATE TABLE IF NOT EXISTS internalization_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id    INTEGER NOT NULL,
    episode_count INTEGER,
    avg_reward  REAL,
    dataset_path TEXT,
    status      TEXT DEFAULT 'pending',
    created_at  REAL,
    FOREIGN KEY(skill_id) REFERENCES skills(id)
);

CREATE INDEX IF NOT EXISTS idx_episodes_skill ON episodes(skill_id);
CREATE INDEX IF NOT EXISTS idx_episodes_signal ON episodes(signal);
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SkillRecord:
    """A registered skill that Homie can learn to internalize."""
    name: str
    description: str
    prompt_template: str
    id: Optional[int] = None
    internalized: bool = False


@dataclass
class Episode:
    """A single skill-use episode with an outcome signal."""
    skill_name: str
    input_text: str
    output_text: str
    reward: float = 0.0
    signal: str = "neutral"  # 'success' | 'failure' | 'corrected' | 'neutral'
    correction: Optional[str] = None


@dataclass
class InternalizationReport:
    """Summary produced after evaluating whether a skill is ready to bake in."""
    skill_name: str
    total_episodes: int
    success_rate: float
    avg_reward: float
    ready: bool
    reason: str


# ---------------------------------------------------------------------------
# Storage helper
# ---------------------------------------------------------------------------

class _Store:
    """Thin SQLite wrapper for skill internalization data."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- skills -------------------------------------------------------------

    def upsert_skill(self, skill: SkillRecord) -> int:
        now = time.time()
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM skills WHERE name = ?", (skill.name,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE skills SET description=?, prompt_template=?, updated_at=? WHERE id=?",
                    (skill.description, skill.prompt_template, now, row["id"]),
                )
                return row["id"]
            cur = conn.execute(
                "INSERT INTO skills (name, description, prompt_template, created_at, updated_at) VALUES (?,?,?,?,?)",
                (skill.name, skill.description, skill.prompt_template, now, now),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_skill_id(self, name: str) -> Optional[int]:
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM skills WHERE name = ?", (name,)).fetchone()
            return row["id"] if row else None

    def mark_internalized(self, skill_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE skills SET internalized=1, updated_at=? WHERE id=?", (time.time(), skill_id))

    def list_skills(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM skills ORDER BY name").fetchall()
            return [dict(r) for r in rows]

    # -- episodes -----------------------------------------------------------

    def record_episode(self, skill_id: int, ep: Episode) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO episodes (skill_id, input_text, output_text, reward, signal, correction, recorded_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (skill_id, ep.input_text, ep.output_text, ep.reward, ep.signal, ep.correction, time.time()),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_episodes(self, skill_id: int) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE skill_id=? ORDER BY recorded_at", (skill_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def skill_stats(self, skill_id: int) -> Dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM episodes WHERE skill_id=?", (skill_id,)).fetchone()["c"]
            successes = conn.execute(
                "SELECT COUNT(*) as c FROM episodes WHERE skill_id=? AND signal='success'", (skill_id,)
            ).fetchone()["c"]
            avg_reward = conn.execute(
                "SELECT COALESCE(AVG(reward), 0.0) as a FROM episodes WHERE skill_id=?", (skill_id,)
            ).fetchone()["a"]
            return {"total": total, "successes": successes, "avg_reward": avg_reward}

    # -- internalization runs ------------------------------------------------

    def record_run(self, skill_id: int, episode_count: int, avg_reward: float, dataset_path: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO internalization_runs (skill_id, episode_count, avg_reward, dataset_path, status, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (skill_id, episode_count, avg_reward, dataset_path, "exported", time.time()),
            )
            return cur.lastrowid  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class SkillInternalizationPlugin:
    """Tracks skill usage and produces LoRA-ready training data when a skill
    has been used successfully enough times to justify internalization.

    Lifecycle:
        activate()  -> initialises DB and config
        deactivate() -> flushes state, closes DB

    Core API:
        register_skill()   -> declare a new skill
        record_episode()   -> log a skill-use outcome
        evaluate_skill()   -> check if a skill is ready for internalization
        export_training_data() -> write JSONL for LoRA fine-tuning
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        min_episodes: int = DEFAULT_MIN_EPISODES,
        success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    ) -> None:
        self.data_dir = data_dir or Path.home() / ".homie" / "plugins" / "skill_internalization"
        self.min_episodes = min_episodes
        self.success_threshold = success_threshold
        self._store: Optional[_Store] = None
        self._active = False

    # -- lifecycle ----------------------------------------------------------

    def activate(self) -> None:
        """Initialise the plugin: create data directory and open the database."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.data_dir / DEFAULT_DB_NAME
        self._store = _Store(db_path)
        self._active = True
        logger.info("SkillInternalizationPlugin activated  db=%s", db_path)

    def deactivate(self) -> None:
        """Tear down the plugin gracefully."""
        self._store = None
        self._active = False
        logger.info("SkillInternalizationPlugin deactivated")

    @property
    def store(self) -> _Store:
        if self._store is None:
            raise RuntimeError("Plugin is not activated. Call activate() first.")
        return self._store

    # -- public API ---------------------------------------------------------

    def register_skill(self, skill: SkillRecord) -> int:
        """Register or update a skill definition.

        Returns the skill's database ID.
        """
        return self.store.upsert_skill(skill)

    def record_episode(self, episode: Episode) -> int:
        """Record a single skill-use episode with an outcome signal.

        The *reward* field uses a simple scale:
            1.0  = success, no correction needed
            0.5  = success after minor user correction
            0.0  = neutral / no signal
           -1.0  = failure

        Returns the episode's database ID.
        """
        skill_id = self.store.get_skill_id(episode.skill_name)
        if skill_id is None:
            raise ValueError(f"Unknown skill '{episode.skill_name}'. Register it first.")
        return self.store.record_episode(skill_id, episode)

    def evaluate_skill(self, skill_name: str) -> InternalizationReport:
        """Decide whether *skill_name* is ready to be internalized.

        Readiness requires:
        1. At least ``min_episodes`` recorded episodes.
        2. A success rate >= ``success_threshold``.
        """
        skill_id = self.store.get_skill_id(skill_name)
        if skill_id is None:
            return InternalizationReport(
                skill_name=skill_name, total_episodes=0, success_rate=0.0,
                avg_reward=0.0, ready=False, reason="Skill not registered.",
            )

        stats = self.store.skill_stats(skill_id)
        total = stats["total"]
        success_rate = stats["successes"] / total if total > 0 else 0.0
        avg_reward = stats["avg_reward"]

        if total < self.min_episodes:
            reason = f"Need {self.min_episodes - total} more episodes (have {total})."
            ready = False
        elif success_rate < self.success_threshold:
            reason = f"Success rate {success_rate:.1%} < threshold {self.success_threshold:.1%}."
            ready = False
        else:
            reason = "Skill meets internalization criteria."
            ready = True

        return InternalizationReport(
            skill_name=skill_name, total_episodes=total,
            success_rate=success_rate, avg_reward=avg_reward,
            ready=ready, reason=reason,
        )

    def export_training_data(self, skill_name: str) -> Path:
        """Export successful episodes as JSONL suitable for LoRA fine-tuning.

        Each line is a JSON object with ``instruction``, ``input``, and
        ``output`` fields â€” the standard Alpaca format understood by most
        local fine-tuning tools (llama.cpp ``finetune``, axolotl, etc.).

        If an episode was corrected by the user the *corrected* output is
        used as the gold label, teaching the model the right behaviour.

        Returns the path to the exported JSONL file.
        """
        skill_id = self.store.get_skill_id(skill_name)
        if skill_id is None:
            raise ValueError(f"Unknown skill '{skill_name}'.")

        episodes = self.store.get_episodes(skill_id)
        export_dir = self.data_dir / DEFAULT_EXPORT_DIR
        export_dir.mkdir(parents=True, exist_ok=True)
        out_path = export_dir / f"{skill_name}.jsonl"

        # Retrieve the skill's prompt template for the instruction field
        skills = self.store.list_skills()
        skill_meta = next((s for s in skills if s["name"] == skill_name), {})
        instruction = skill_meta.get("prompt_template", f"Perform the '{skill_name}' skill.")

        count = 0
        with open(out_path, "w", encoding="utf-8") as fh:
            for ep in episodes:
                # Only include positive-signal episodes
                if ep["signal"] not in ("success", "corrected"):
                    continue
                gold_output = ep["correction"] if ep["correction"] else ep["output_text"]
                record = {
                    "instruction": instruction,
                    "input": ep["input_text"],
                    "output": gold_output,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        stats = self.store.skill_stats(skill_id)
        self.store.record_run(skill_id, count, stats["avg_reward"], str(out_path))
        logger.info("Exported %d episodes for '%s' -> %s", count, skill_name, out_path)
        return out_path

    def list_skills(self) -> List[Dict[str, Any]]:
        """Return all registered skills with their internalization status."""
        return self.store.list_skills()

    def skill_summary(self, skill_name: str) -> Dict[str, Any]:
        """Return a combined summary: registration info + episode stats + readiness."""
        report = self.evaluate_skill(skill_name)
        return {
            "skill_name": report.skill_name,
            "total_episodes": report.total_episodes,
            "success_rate": round(report.success_rate, 4),
            "avg_reward": round(report.avg_reward, 4),
            "ready_for_internalization": report.ready,
            "reason": report.reason,
        }


# ---------------------------------------------------------------------------
# Module-level convenience (matches Homie plugin conventions)
# ---------------------------------------------------------------------------

_plugin_instance: Optional[SkillInternalizationPlugin] = None


def activate(data_dir: Optional[Path] = None, **kwargs: Any) -> SkillInternalizationPlugin:
    """Activate the skill-internalization plugin and return the instance."""
    global _plugin_instance
    _plugin_instance = SkillInternalizationPlugin(data_dir=data_dir, **kwargs)
    _plugin_instance.activate()
    return _plugin_instance


def deactivate() -> None:
    """Deactivate the global plugin instance."""
    global _plugin_instance
    if _plugin_instance is not None:
        _plugin_instance.deactivate()
        _plugin_instance = None


def register() -> Dict[str, Any]:
    """Return plugin metadata for Homie's plugin registry."""
    return {
        "name": "skill_internalization",
        "version": "0.1.0",
        "description": (
            "Tracks skill usage episodes and exports LoRA training data "
            "so frequently-used skills can be baked into the local model."
        ),
        "author": "Homie Contributors",
        "activate": activate,
        "deactivate": deactivate,
    }
