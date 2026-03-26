"""Persistent conversation memory — remembers across sessions.

Stores conversation summaries and extracted facts from each session
so Homie can reference past interactions naturally.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationSummary:
    """Summary of a single conversation session."""
    session_id: str
    timestamp: float
    duration_minutes: float
    turn_count: int
    summary: str                    # Natural language summary
    key_topics: list[str]          # Main topics discussed
    facts_learned: list[str]       # New facts learned about user
    action_items: list[str]        # Things Homie committed to do
    user_mood: str = ""            # Detected mood: "positive", "neutral", "stressed", etc.


class ConversationStore:
    """Persistent storage for conversation history across sessions."""

    def __init__(self, db_path: str | Path = "~/.homie/memory/conversations.db"):
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._init_db()

    def _init_db(self):
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                duration_minutes REAL DEFAULT 0,
                turn_count INTEGER DEFAULT 0,
                summary TEXT NOT NULL,
                key_topics TEXT,
                facts_learned TEXT,
                action_items TEXT,
                user_mood TEXT DEFAULT 'neutral',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp DESC);

            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES conversations(session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_turns(session_id);
        """)
        self._conn.commit()

    def save_turn(self, session_id: str, role: str, content: str) -> None:
        """Save an individual conversation turn."""
        self._conn.execute(
            "INSERT INTO conversation_turns (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content[:5000], time.time()),  # Cap at 5000 chars
        )
        self._conn.commit()

    def save_summary(self, summary: ConversationSummary) -> None:
        """Save a conversation summary at end of session."""
        self._conn.execute(
            """INSERT OR REPLACE INTO conversations
               (session_id, timestamp, duration_minutes, turn_count, summary,
                key_topics, facts_learned, action_items, user_mood, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                summary.session_id, summary.timestamp, summary.duration_minutes,
                summary.turn_count, summary.summary,
                json.dumps(summary.key_topics),
                json.dumps(summary.facts_learned),
                json.dumps(summary.action_items),
                summary.user_mood, time.time(),
            ),
        )
        self._conn.commit()

    def get_recent_summaries(self, limit: int = 5) -> list[ConversationSummary]:
        """Get the N most recent conversation summaries."""
        rows = self._conn.execute(
            "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def search_conversations(self, query: str, limit: int = 5) -> list[ConversationSummary]:
        """Search conversation summaries by keyword."""
        rows = self._conn.execute(
            "SELECT * FROM conversations WHERE summary LIKE ? OR key_topics LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def get_session_turns(self, session_id: str) -> list[dict]:
        """Get all turns from a specific session."""
        rows = self._conn.execute(
            "SELECT role, content, timestamp FROM conversation_turns WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

    def get_context_for_prompt(self, max_summaries: int = 3) -> str:
        """Build a context block from recent conversations for the system prompt."""
        summaries = self.get_recent_summaries(max_summaries)
        if not summaries:
            return ""

        lines = ["Recent conversation history:"]
        for s in summaries:
            from datetime import datetime
            dt = datetime.fromtimestamp(s.timestamp)
            time_str = dt.strftime("%b %d, %I:%M %p")
            lines.append(f"  [{time_str}] {s.summary}")
            if s.action_items:
                for item in s.action_items[:2]:
                    lines.append(f"    - Pending: {item}")

        return "\n".join(lines)

    def generate_summary_from_turns(self, session_id: str, inference_fn=None) -> ConversationSummary:
        """Generate a summary from stored turns, optionally using LLM."""
        turns = self.get_session_turns(session_id)
        if not turns:
            return ConversationSummary(
                session_id=session_id, timestamp=time.time(),
                duration_minutes=0, turn_count=0, summary="Empty session",
                key_topics=[], facts_learned=[], action_items=[],
            )

        turn_count = len(turns)
        first_ts = turns[0]["timestamp"]
        last_ts = turns[-1]["timestamp"]
        duration = (last_ts - first_ts) / 60

        # Extract topics and facts heuristically
        user_msgs = [t["content"] for t in turns if t["role"] == "user"]

        # Simple keyword extraction for topics
        all_text = " ".join(user_msgs).lower()
        topic_keywords = {
            "email": ["email", "inbox", "unread", "gmail"],
            "coding": ["code", "python", "function", "bug", "git", "commit"],
            "project": ["project", "working on", "homie", "feature"],
            "security": ["security", "password", "api key", "vault"],
            "planning": ["plan", "schedule", "meeting", "standup"],
            "linkedin": ["linkedin", "post", "profile"],
        }
        topics = [topic for topic, kws in topic_keywords.items() if any(kw in all_text for kw in kws)]

        # Detect mood
        mood = "neutral"
        if any(w in all_text for w in ["stressed", "overwhelmed", "frustrated", "tired"]):
            mood = "stressed"
        elif any(w in all_text for w in ["great", "awesome", "happy", "excited", "promoted"]):
            mood = "positive"
        elif any(w in all_text for w in ["thanks", "helpful", "appreciate"]):
            mood = "grateful"

        # Build summary
        if inference_fn:
            try:
                convo_text = "\n".join(f"{t['role']}: {t['content'][:200]}" for t in turns[:20])
                prompt = f"""Summarize this conversation in 1-2 sentences. Also list any action items or facts learned about the user.

{convo_text}

Format:
Summary: <1-2 sentences>
Facts: <comma-separated facts learned, or "none">
Actions: <comma-separated action items, or "none">"""
                raw = inference_fn(prompt=prompt, max_tokens=200, temperature=0.3)
                # Parse response
                summary_text = raw.split("Summary:")[-1].split("Facts:")[0].strip() if "Summary:" in raw else raw[:200]
                facts_text = raw.split("Facts:")[-1].split("Actions:")[0].strip() if "Facts:" in raw else ""
                actions_text = raw.split("Actions:")[-1].strip() if "Actions:" in raw else ""

                facts = [f.strip() for f in facts_text.split(",") if f.strip() and f.strip().lower() != "none"]
                actions = [a.strip() for a in actions_text.split(",") if a.strip() and a.strip().lower() != "none"]
            except Exception as exc:
                logger.debug("LLM summary failed: %s", exc)
                summary_text = f"Discussed {', '.join(topics[:3]) if topics else 'various topics'} over {turn_count} turns."
                facts = []
                actions = []
        else:
            summary_text = f"Discussed {', '.join(topics[:3]) if topics else 'various topics'} over {turn_count} turns."
            facts = []
            actions = []

        return ConversationSummary(
            session_id=session_id, timestamp=first_ts,
            duration_minutes=round(duration, 1), turn_count=turn_count,
            summary=summary_text, key_topics=topics,
            facts_learned=facts, action_items=actions, user_mood=mood,
        )

    def get_stats(self) -> dict:
        """Get conversation statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        total_turns = self._conn.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0]
        return {"total_sessions": total, "total_turns": total_turns}

    def close(self):
        if self._conn:
            self._conn.close()

    def _row_to_summary(self, row) -> ConversationSummary:
        return ConversationSummary(
            session_id=row[0], timestamp=row[1], duration_minutes=row[2],
            turn_count=row[3], summary=row[4],
            key_topics=json.loads(row[5]) if row[5] else [],
            facts_learned=json.loads(row[6]) if row[6] else [],
            action_items=json.loads(row[7]) if row[7] else [],
            user_mood=row[8] or "neutral",
        )
