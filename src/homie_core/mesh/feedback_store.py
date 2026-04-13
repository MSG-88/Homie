"""FeedbackStore — SQLite persistence for learning signals."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from homie_core.mesh.feedback_collector import FeedbackSignal, SignalType


class FeedbackStore:
    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback_signals (
                signal_id       TEXT PRIMARY KEY,
                signal_type     TEXT NOT NULL,
                query           TEXT NOT NULL,
                response_preview TEXT NOT NULL,
                node_id         TEXT NOT NULL,
                activity_context TEXT NOT NULL DEFAULT '',
                timestamp       TEXT NOT NULL,
                rating          INTEGER,
                metadata_json   TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fs_type ON feedback_signals(signal_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fs_ts ON feedback_signals(timestamp, signal_id)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, sig: FeedbackSignal) -> None:
        rating: Optional[int] = sig.metadata.get("rating") if sig.metadata else None
        self._conn.execute(
            """
            INSERT OR REPLACE INTO feedback_signals
                (signal_id, signal_type, query, response_preview, node_id,
                 activity_context, timestamp, rating, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sig.signal_id,
                sig.signal_type,
                sig.query,
                sig.response_preview,
                sig.node_id,
                sig.activity_context,
                sig.timestamp,
                rating,
                json.dumps(sig.metadata),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read — single record
    # ------------------------------------------------------------------

    def get(self, signal_id: str) -> Optional[FeedbackSignal]:
        row = self._conn.execute(
            "SELECT * FROM feedback_signals WHERE signal_id = ?", (signal_id,)
        ).fetchone()
        return self._row_to_signal(row) if row else None

    # ------------------------------------------------------------------
    # Read — aggregates
    # ------------------------------------------------------------------

    def total_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM feedback_signals"
        ).fetchone()[0]

    def count_by_type(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT signal_type, COUNT(*) AS cnt FROM feedback_signals GROUP BY signal_type"
        ).fetchall()
        return {row["signal_type"]: row["cnt"] for row in rows}

    # ------------------------------------------------------------------
    # Read — slices
    # ------------------------------------------------------------------

    def signals_since(
        self, after_signal_id: Optional[str], limit: int = 1000
    ) -> list[FeedbackSignal]:
        """Return signals whose signal_id (ULID) is lexicographically > after_signal_id."""
        if after_signal_id is None:
            rows = self._conn.execute(
                "SELECT * FROM feedback_signals ORDER BY signal_id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM feedback_signals WHERE signal_id > ? ORDER BY signal_id ASC LIMIT ?",
                (after_signal_id, limit),
            ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    # ------------------------------------------------------------------
    # Training pairs
    # ------------------------------------------------------------------

    def get_training_pairs(self) -> list[dict[str, Any]]:
        """Return DPO pairs (CORRECTED) and SFT pairs (ACCEPTED)."""
        rows = self._conn.execute(
            "SELECT * FROM feedback_signals WHERE signal_type IN (?, ?) ORDER BY signal_id ASC",
            (SignalType.CORRECTED, SignalType.ACCEPTED),
        ).fetchall()

        pairs: list[dict[str, Any]] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if row["signal_type"] == SignalType.CORRECTED:
                pairs.append(
                    {
                        "type": "dpo",
                        "query": row["query"],
                        "rejected": row["response_preview"],
                        "chosen": metadata.get("correction", ""),
                    }
                )
            elif row["signal_type"] == SignalType.ACCEPTED:
                pairs.append(
                    {
                        "type": "sft",
                        "query": row["query"],
                        "response": row["response_preview"],
                    }
                )
        return pairs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_signal(row: sqlite3.Row) -> FeedbackSignal:
        sig = FeedbackSignal(
            signal_type=row["signal_type"],
            query=row["query"],
            response_preview=row["response_preview"],
            node_id=row["node_id"],
            activity_context=row["activity_context"] or "",
            metadata=json.loads(row["metadata_json"]),
        )
        sig.signal_id = row["signal_id"]
        sig.timestamp = row["timestamp"]
        return sig
