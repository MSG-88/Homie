# Homie AI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fully local, privacy-first personal AI assistant with integrated model engine, living memory, behavioral intelligence, voice pipeline, and plugin ecosystem.

**Architecture:** Framework (`homie_core`) + Application (`homie_app`), monolithic Python service with clean module boundaries. GGUF/SafeTensors models loaded directly via llama-cpp-python/transformers.

**Tech Stack:** Python 3.11+, llama-cpp-python, transformers, faster-whisper, OpenWakeWord, Piper, FastAPI, ChromaDB, SQLite, pywin32, psutil, pystray, huggingface_hub

**Phases:** 12 phases, each independently testable. Execute in order — each phase builds on the previous.

---

## Phase 1: Project Scaffold & Configuration

### Task 1.1: Initialize Project Structure

**Files:**
- Create: `pyproject.toml`
- Create: `src/homie_core/__init__.py`
- Create: `src/homie_app/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "homie-ai"
version = "0.1.0"
description = "Fully local, privacy-first personal AI assistant"
requires-python = ">=3.11"
dependencies = [
    "pyyaml>=6.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
]
model = [
    "llama-cpp-python>=0.2",
    "transformers>=4.40",
    "accelerate>=0.30",
    "huggingface-hub>=0.23",
]
voice = [
    "faster-whisper>=1.0",
    "openwakeword>=0.6",
    "piper-tts>=1.2",
    "pyaudio>=0.2",
    "sounddevice>=0.4",
]
context = [
    "pywin32>=306; sys_platform == 'win32'",
    "psutil>=5.9",
    "watchdog>=4.0",
]
storage = [
    "chromadb>=0.5",
    "cryptography>=42.0",
]
app = [
    "fastapi>=0.111",
    "uvicorn>=0.30",
    "pystray>=0.19",
    "Pillow>=10.0",
    "pynput>=1.7",
    "apscheduler>=3.10",
]
all = ["homie-ai[model,voice,context,storage,app]"]

[project.scripts]
homie = "homie_app.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 2: Create package init files**

`src/homie_core/__init__.py`:
```python
"""Homie Core — Framework for building local AI assistants."""
__version__ = "0.1.0"
```

`src/homie_app/__init__.py`:
```python
"""Homie App — Local AI assistant application."""
```

`tests/__init__.py`: empty file

**Step 3: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: initialize homie-ai project scaffold"
```

---

### Task 1.2: Configuration System

**Files:**
- Create: `src/homie_core/config.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_config.py`
- Create: `default_config.yaml`

**Step 1: Write failing tests**

`tests/unit/test_config.py`:
```python
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from homie_core.config import HomieConfig, load_config


@pytest.fixture
def tmp_config(tmp_path):
    cfg = {
        "llm": {"model_path": "models/test.gguf", "backend": "gguf"},
        "voice": {"enabled": False},
        "storage": {"path": str(tmp_path / ".homie")},
        "privacy": {"data_retention_days": 30},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def test_load_config_from_file(tmp_config):
    cfg = load_config(tmp_config)
    assert cfg.llm.backend == "gguf"
    assert cfg.voice.enabled is False


def test_load_config_defaults():
    cfg = load_config()
    assert cfg.llm.backend == "gguf"
    assert cfg.storage.path is not None


def test_config_env_override(tmp_config, monkeypatch):
    monkeypatch.setenv("HOMIE_LLM_BACKEND", "transformers")
    cfg = load_config(tmp_config)
    assert cfg.llm.backend == "transformers"


def test_config_data_dir_created(tmp_path):
    cfg_data = {"storage": {"path": str(tmp_path / ".homie")}}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg_data))
    cfg = load_config(p)
    assert Path(cfg.storage.path).exists()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'homie_core'`

**Step 3: Implement config**

`src/homie_core/config.py`:
```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model_path: str = ""
    backend: str = "gguf"  # "gguf" or "transformers"
    context_length: int = 4096
    temperature: float = 0.7
    max_tokens: int = 1024
    gpu_layers: int = -1  # -1 = all layers on GPU


class VoiceConfig(BaseModel):
    enabled: bool = False
    wake_word: str = "hey homie"
    stt_model: str = "large-v3"
    tts_voice: str = "default"
    mode: str = "text_only"  # "always_on", "push_to_talk", "text_only"
    hotkey: str = "F9"


class StorageConfig(BaseModel):
    path: str = Field(default_factory=lambda: str(Path.home() / ".homie"))
    db_name: str = "homie.db"
    chroma_dir: str = "chroma"
    models_dir: str = "models"
    backup_dir: str = "backups"
    log_dir: str = "logs"


class PrivacyConfig(BaseModel):
    data_retention_days: int = 30
    max_storage_mb: int = 512
    observers: dict[str, bool] = Field(default_factory=lambda: {
        "media": False,
        "browsing": False,
        "work": True,
        "social": False,
        "routine": True,
        "emotional": False,
    })


class PluginConfig(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    plugin_dirs: list[str] = Field(default_factory=list)


class HomieConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    user_name: str = ""


def _apply_env_overrides(cfg: HomieConfig) -> HomieConfig:
    env_map = {
        "HOMIE_LLM_BACKEND": ("llm", "backend"),
        "HOMIE_LLM_MODEL_PATH": ("llm", "model_path"),
        "HOMIE_VOICE_ENABLED": ("voice", "enabled"),
        "HOMIE_STORAGE_PATH": ("storage", "path"),
        "HOMIE_USER_NAME": ("user_name",),
    }
    for env_var, path in env_map.items():
        val = os.environ.get(env_var)
        if val is None:
            continue
        obj = cfg
        for key in path[:-1]:
            obj = getattr(obj, key)
        field = path[-1]
        field_type = type(getattr(obj, field))
        if field_type is bool:
            val = val.lower() in ("true", "1", "yes")
        setattr(obj, field, field_type(val))
    return cfg


def load_config(path: Optional[Path | str] = None) -> HomieConfig:
    data = {}
    if path is not None:
        path = Path(path)
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
    elif Path("homie.config.yaml").exists():
        data = yaml.safe_load(Path("homie.config.yaml").read_text()) or {}

    cfg = HomieConfig(**data)
    cfg = _apply_env_overrides(cfg)

    storage_path = Path(cfg.storage.path)
    storage_path.mkdir(parents=True, exist_ok=True)

    return cfg
```

**Step 4: Run tests**

```bash
pytest tests/unit/test_config.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/homie_core/config.py tests/unit/ default_config.yaml
git commit -m "feat: add configuration system with env overrides"
```

---

### Task 1.3: Utility Helpers

**Files:**
- Create: `src/homie_core/utils.py`
- Create: `tests/unit/test_utils.py`

**Step 1: Write failing tests**

`tests/unit/test_utils.py`:
```python
from datetime import datetime, timezone

from homie_core.utils import utc_now, privacy_tag, truncate_text


def test_utc_now_returns_utc():
    now = utc_now()
    assert now.tzinfo == timezone.utc


def test_privacy_tag():
    tagged = privacy_tag({"data": "hello"}, tags=["personal", "ephemeral"])
    assert tagged["_privacy_tags"] == ["personal", "ephemeral"]
    assert tagged["data"] == "hello"


def test_truncate_text():
    assert truncate_text("hello world", 5) == "hello..."
    assert truncate_text("hi", 10) == "hi"
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

`src/homie_core/utils.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def privacy_tag(data: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    return {**data, "_privacy_tags": tags}


def truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/homie_core/utils.py tests/unit/test_utils.py
git commit -m "feat: add utility helpers"
```

---

## Phase 2: Storage Layer

### Task 2.1: SQLite Database

**Files:**
- Create: `src/homie_core/storage/__init__.py`
- Create: `src/homie_core/storage/database.py`
- Create: `tests/unit/test_storage/__init__.py`
- Create: `tests/unit/test_storage/test_database.py`

**Step 1: Write failing tests**

`tests/unit/test_storage/test_database.py`:
```python
import pytest
from homie_core.storage.database import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    yield d
    d.close()


def test_initialize_creates_tables(db):
    tables = db.list_tables()
    assert "semantic_memory" in tables
    assert "beliefs" in tables
    assert "profile" in tables
    assert "feedback" in tables
    assert "episodes_meta" in tables


def test_store_and_retrieve_fact(db):
    db.store_fact("user prefers Python", confidence=0.9, tags=["work"])
    facts = db.get_facts(min_confidence=0.5)
    assert len(facts) == 1
    assert facts[0]["fact"] == "user prefers Python"
    assert facts[0]["confidence"] == 0.9


def test_store_belief(db):
    db.store_belief("likes concise responses", confidence=0.85, source_count=10, context_tags=["communication"])
    beliefs = db.get_beliefs()
    assert len(beliefs) == 1
    assert beliefs[0]["belief"] == "likes concise responses"


def test_record_feedback(db):
    db.record_feedback(channel="correction", content="prefers spaces over tabs", context={"app": "vscode"})
    feedback = db.get_recent_feedback(limit=10)
    assert len(feedback) == 1
    assert feedback[0]["channel"] == "correction"


def test_store_profile_domain(db):
    db.store_profile("music", {"top_genres": ["electronic", "lo-fi"]})
    profile = db.get_profile("music")
    assert profile["top_genres"] == ["electronic", "lo-fi"]


def test_update_profile_domain(db):
    db.store_profile("music", {"top_genres": ["electronic"]})
    db.store_profile("music", {"top_genres": ["electronic", "lo-fi"]})
    profile = db.get_profile("music")
    assert len(profile["top_genres"]) == 2
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

`src/homie_core/storage/__init__.py`: empty

`src/homie_core/storage/database.py`:
```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from homie_core.utils import utc_now


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                source_count INTEGER NOT NULL DEFAULT 1,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                last_confirmed TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS beliefs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                belief TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                source_count INTEGER NOT NULL DEFAULT 1,
                context_tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS profile (
                domain TEXT PRIMARY KEY,
                data TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodes_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                mood TEXT,
                outcome TEXT,
                context_tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS habits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                frequency TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                last_seen TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def list_tables(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return [r["name"] for r in rows]

    def store_fact(self, fact: str, confidence: float = 0.5, tags: list[str] | None = None) -> int:
        now = utc_now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO semantic_memory (fact, confidence, tags, created_at, last_confirmed) VALUES (?, ?, ?, ?, ?)",
            (fact, confidence, json.dumps(tags or []), now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_facts(self, min_confidence: float = 0.0, include_archived: bool = False) -> list[dict]:
        query = "SELECT * FROM semantic_memory WHERE confidence >= ?"
        params: list[Any] = [min_confidence]
        if not include_archived:
            query += " AND archived = 0"
        rows = self._conn.execute(query, params).fetchall()
        return [
            {**dict(r), "tags": json.loads(r["tags"])}
            for r in rows
        ]

    def store_belief(self, belief: str, confidence: float, source_count: int = 1, context_tags: list[str] | None = None) -> int:
        now = utc_now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO beliefs (belief, confidence, source_count, context_tags, created_at, last_updated) VALUES (?, ?, ?, ?, ?, ?)",
            (belief, confidence, source_count, json.dumps(context_tags or []), now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_beliefs(self, min_confidence: float = 0.0) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM beliefs WHERE confidence >= ?", (min_confidence,)
        ).fetchall()
        return [
            {**dict(r), "context_tags": json.loads(r["context_tags"])}
            for r in rows
        ]

    def record_feedback(self, channel: str, content: str, context: dict | None = None) -> int:
        now = utc_now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO feedback (channel, content, context, created_at) VALUES (?, ?, ?, ?)",
            (channel, content, json.dumps(context or {}), now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_recent_feedback(self, limit: int = 50, channel: str | None = None) -> list[dict]:
        query = "SELECT * FROM feedback"
        params: list[Any] = []
        if channel:
            query += " WHERE channel = ?"
            params.append(channel)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [{**dict(r), "context": json.loads(r["context"])} for r in rows]

    def store_profile(self, domain: str, data: dict) -> None:
        now = utc_now().isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO profile (domain, data, updated_at) VALUES (?, ?, ?)",
            (domain, json.dumps(data), now),
        )
        self._conn.commit()

    def get_profile(self, domain: str) -> dict | None:
        row = self._conn.execute(
            "SELECT data FROM profile WHERE domain = ?", (domain,)
        ).fetchone()
        if row:
            return json.loads(row["data"])
        return None

    def record_episode_meta(self, summary: str, mood: str | None = None, outcome: str | None = None, context_tags: list[str] | None = None) -> int:
        now = utc_now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO episodes_meta (summary, mood, outcome, context_tags, created_at) VALUES (?, ?, ?, ?, ?)",
            (summary, mood, outcome, json.dumps(context_tags or []), now),
        )
        self._conn.commit()
        return cur.lastrowid

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/homie_core/storage/ tests/unit/test_storage/
git commit -m "feat: add SQLite storage layer with memory, beliefs, profile, feedback tables"
```

---

### Task 2.2: ChromaDB Vector Store

**Files:**
- Create: `src/homie_core/storage/vectors.py`
- Create: `tests/unit/test_storage/test_vectors.py`

**Step 1: Write failing tests**

`tests/unit/test_storage/test_vectors.py`:
```python
import pytest
from homie_core.storage.vectors import VectorStore


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "chroma")
    s.initialize()
    return s


def test_add_and_query_episode(store):
    store.add_episode("ep1", "User debugged auth module for 2 hours", {"mood": "frustrated", "tags": '["work"]'})
    results = store.query_episodes("authentication debugging", n=1)
    assert len(results) == 1
    assert "auth" in results[0]["text"]


def test_add_and_query_file_chunk(store):
    store.add_file_chunk("f1", "def hello(): return 'world'", {"file_path": "src/main.py", "chunk_index": "0"})
    results = store.query_files("hello function", n=1)
    assert len(results) == 1
    assert "hello" in results[0]["text"]


def test_delete_by_id(store):
    store.add_episode("ep_del", "temporary episode", {})
    store.delete_episodes(["ep_del"])
    results = store.query_episodes("temporary", n=1)
    assert len(results) == 0 or "temporary" not in results[0].get("text", "")
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

`src/homie_core/storage/vectors.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._client: chromadb.ClientAPI | None = None
        self._episodes: chromadb.Collection | None = None
        self._files: chromadb.Collection | None = None

    def initialize(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._episodes = self._client.get_or_create_collection("episodes")
        self._files = self._client.get_or_create_collection("file_chunks")

    def add_episode(self, episode_id: str, text: str, metadata: dict[str, str]) -> None:
        self._episodes.upsert(ids=[episode_id], documents=[text], metadatas=[metadata])

    def query_episodes(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        results = self._episodes.query(query_texts=[query], n_results=n)
        return self._format_results(results)

    def delete_episodes(self, ids: list[str]) -> None:
        self._episodes.delete(ids=ids)

    def add_file_chunk(self, chunk_id: str, text: str, metadata: dict[str, str]) -> None:
        self._files.upsert(ids=[chunk_id], documents=[text], metadatas=[metadata])

    def query_files(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        results = self._files.query(query_texts=[query], n_results=n)
        return self._format_results(results)

    def delete_file_chunks(self, ids: list[str]) -> None:
        self._files.delete(ids=ids)

    @staticmethod
    def _format_results(results: dict) -> list[dict[str, Any]]:
        out = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for i, doc_id in enumerate(results["ids"][0]):
            entry = {
                "id": doc_id,
                "text": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            }
            if results.get("distances"):
                entry["distance"] = results["distances"][0][i]
            out.append(entry)
        return out
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/homie_core/storage/vectors.py tests/unit/test_storage/test_vectors.py
git commit -m "feat: add ChromaDB vector store for episodes and file chunks"
```

---

### Task 2.3: Encrypted Backup

**Files:**
- Create: `src/homie_core/storage/backup.py`
- Create: `tests/unit/test_storage/test_backup.py`

**Step 1: Write failing tests**

`tests/unit/test_storage/test_backup.py`:
```python
import pytest
from pathlib import Path
from homie_core.storage.backup import BackupManager


@pytest.fixture
def source_dir(tmp_path):
    src = tmp_path / "homie_data"
    src.mkdir()
    (src / "homie.db").write_text("fake db content")
    (src / "config.yaml").write_text("llm:\n  backend: gguf")
    sub = src / "chroma"
    sub.mkdir()
    (sub / "index.bin").write_bytes(b"\x00\x01\x02")
    return src


def test_backup_and_restore(source_dir, tmp_path):
    backup_dir = tmp_path / "backup"
    restore_dir = tmp_path / "restored"

    mgr = BackupManager()
    mgr.create_backup(source_dir, backup_dir, passphrase="testpass123")
    assert (backup_dir / "homie_backup.enc").exists()

    mgr.restore_backup(backup_dir, restore_dir, passphrase="testpass123")
    assert (restore_dir / "homie.db").read_text() == "fake db content"
    assert (restore_dir / "chroma" / "index.bin").read_bytes() == b"\x00\x01\x02"


def test_wrong_passphrase_fails(source_dir, tmp_path):
    backup_dir = tmp_path / "backup"
    restore_dir = tmp_path / "restored"

    mgr = BackupManager()
    mgr.create_backup(source_dir, backup_dir, passphrase="correct")
    with pytest.raises(Exception):
        mgr.restore_backup(backup_dir, restore_dir, passphrase="wrong")
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

`src/homie_core/storage/backup.py`:
```python
from __future__ import annotations

import io
import tarfile
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64
import os


class BackupManager:
    SALT_SIZE = 16
    ITERATIONS = 480_000

    def _derive_key(self, passphrase: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))

    def create_backup(self, source: Path, dest: Path, passphrase: str) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(source), arcname=".")
        plaintext = buf.getvalue()

        salt = os.urandom(self.SALT_SIZE)
        key = self._derive_key(passphrase, salt)
        f = Fernet(key)
        ciphertext = f.encrypt(plaintext)

        out_path = dest / "homie_backup.enc"
        out_path.write_bytes(salt + ciphertext)
        return out_path

    def restore_backup(self, backup_dir: Path, dest: Path, passphrase: str) -> None:
        enc_path = backup_dir / "homie_backup.enc"
        raw = enc_path.read_bytes()

        salt = raw[: self.SALT_SIZE]
        ciphertext = raw[self.SALT_SIZE :]
        key = self._derive_key(passphrase, salt)
        f = Fernet(key)
        try:
            plaintext = f.decrypt(ciphertext)
        except InvalidToken:
            raise ValueError("Wrong passphrase or corrupted backup")

        dest.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO(plaintext)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(str(dest))
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/homie_core/storage/backup.py tests/unit/test_storage/test_backup.py
git commit -m "feat: add AES-256 encrypted backup and restore"
```

---

## Phase 3: Hardware Detection & Model Engine

### Task 3.1: Hardware Detector

**Files:**
- Create: `src/homie_core/hardware/__init__.py`
- Create: `src/homie_core/hardware/detector.py`
- Create: `src/homie_core/hardware/profiles.py`
- Create: `tests/unit/test_hardware/__init__.py`
- Create: `tests/unit/test_hardware/test_detector.py`

**Step 1: Write failing tests**

`tests/unit/test_hardware/test_detector.py`:
```python
from homie_core.hardware.detector import HardwareInfo, detect_hardware
from homie_core.hardware.profiles import recommend_model


def test_detect_hardware_returns_info():
    info = detect_hardware()
    assert isinstance(info, HardwareInfo)
    assert info.ram_gb > 0
    assert info.os_name in ("Windows", "Linux", "Darwin")


def test_recommend_model_16gb():
    rec = recommend_model(gpu_vram_gb=16.0)
    assert rec["quant"] == "Q4_K_M"
    assert "72B" in rec["model"] or "70B" in rec["model"]


def test_recommend_model_8gb():
    rec = recommend_model(gpu_vram_gb=8.0)
    assert "8B" in rec["model"]


def test_recommend_model_no_gpu():
    rec = recommend_model(gpu_vram_gb=0)
    assert rec["format"] == "gguf"
    assert rec["backend"] == "cpu"
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

`src/homie_core/hardware/__init__.py`: empty

`src/homie_core/hardware/detector.py`:
```python
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field

import psutil


@dataclass
class GPUInfo:
    name: str = ""
    vram_mb: int = 0
    cuda_available: bool = False


@dataclass
class HardwareInfo:
    os_name: str = ""
    os_version: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0.0
    gpus: list[GPUInfo] = field(default_factory=list)
    has_microphone: bool = False

    @property
    def best_gpu_vram_gb(self) -> float:
        if not self.gpus:
            return 0.0
        return max(g.vram_mb for g in self.gpus) / 1024.0


def detect_hardware() -> HardwareInfo:
    info = HardwareInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        cpu_cores=psutil.cpu_count(logical=True) or 1,
        ram_gb=round(psutil.virtual_memory().total / (1024**3), 1),
    )
    info.gpus = _detect_gpus()
    info.has_microphone = _detect_microphone()
    return info


def _detect_gpus() -> list[GPUInfo]:
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if "," in line:
                    name, vram = line.split(",", 1)
                    gpus.append(GPUInfo(name=name.strip(), vram_mb=int(vram.strip()), cuda_available=True))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus


def _detect_microphone() -> bool:
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return any(d.get("max_input_channels", 0) > 0 for d in devices)
    except Exception:
        return False
```

`src/homie_core/hardware/profiles.py`:
```python
from __future__ import annotations

GPU_PROFILES = [
    {"min_vram": 16.0, "model": "Qwen2.5-72B-Instruct", "quant": "Q4_K_M", "format": "gguf", "backend": "cuda"},
    {"min_vram": 12.0, "model": "Qwen2.5-32B-Instruct", "quant": "Q4_K_M", "format": "gguf", "backend": "cuda"},
    {"min_vram": 8.0, "model": "Llama-3.1-8B-Instruct", "quant": "Q5_K_M", "format": "gguf", "backend": "cuda"},
    {"min_vram": 4.0, "model": "Phi-3-mini-4k-instruct", "quant": "Q4_K_M", "format": "gguf", "backend": "cuda"},
    {"min_vram": 0.0, "model": "Phi-3-mini-4k-instruct", "quant": "Q4_0", "format": "gguf", "backend": "cpu"},
]


def recommend_model(gpu_vram_gb: float) -> dict:
    for profile in GPU_PROFILES:
        if gpu_vram_gb >= profile["min_vram"]:
            return {
                "model": profile["model"],
                "quant": profile["quant"],
                "format": profile["format"],
                "backend": profile["backend"],
            }
    return GPU_PROFILES[-1]
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/homie_core/hardware/ tests/unit/test_hardware/
git commit -m "feat: add hardware detection and model recommendation profiles"
```

---

### Task 3.2: Model Registry & Downloader

**Files:**
- Create: `src/homie_core/model/__init__.py`
- Create: `src/homie_core/model/registry.py`
- Create: `src/homie_core/model/downloader.py`
- Create: `tests/unit/test_model/__init__.py`
- Create: `tests/unit/test_model/test_registry.py`

**Step 1: Write failing tests**

`tests/unit/test_model/test_registry.py`:
```python
import pytest
from pathlib import Path
from homie_core.model.registry import ModelRegistry, ModelEntry


@pytest.fixture
def registry(tmp_path):
    r = ModelRegistry(tmp_path / "models")
    r.initialize()
    return r


def test_register_local_model(registry, tmp_path):
    model_file = tmp_path / "test.gguf"
    model_file.write_bytes(b"fake gguf data")
    registry.register("test-model", model_file, format="gguf", params="7B")
    entry = registry.get("test-model")
    assert entry is not None
    assert entry.name == "test-model"
    assert entry.format == "gguf"


def test_list_models(registry, tmp_path):
    f1 = tmp_path / "m1.gguf"
    f1.write_bytes(b"data")
    registry.register("model-a", f1, format="gguf", params="7B")
    registry.register("model-b", f1, format="gguf", params="13B")
    models = registry.list_models()
    assert len(models) == 2


def test_remove_model(registry, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"data")
    registry.register("to-delete", f, format="gguf", params="7B")
    registry.remove("to-delete")
    assert registry.get("to-delete") is None


def test_active_model(registry, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"data")
    registry.register("my-model", f, format="gguf", params="7B")
    registry.set_active("my-model")
    assert registry.get_active().name == "my-model"
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

`src/homie_core/model/__init__.py`: empty

`src/homie_core/model/registry.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ModelEntry:
    name: str
    path: str
    format: str  # "gguf" or "safetensors"
    params: str  # "7B", "13B", "70B"
    repo_id: str = ""
    quant: str = ""
    active: bool = False


class ModelRegistry:
    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        self._registry_path = self.models_dir / "registry.json"
        self._entries: dict[str, ModelEntry] = {}

    def initialize(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if self._registry_path.exists():
            data = json.loads(self._registry_path.read_text())
            self._entries = {k: ModelEntry(**v) for k, v in data.items()}

    def _save(self) -> None:
        data = {k: asdict(v) for k, v in self._entries.items()}
        self._registry_path.write_text(json.dumps(data, indent=2))

    def register(self, name: str, path: Path | str, format: str, params: str, repo_id: str = "", quant: str = "") -> ModelEntry:
        entry = ModelEntry(name=name, path=str(path), format=format, params=params, repo_id=repo_id, quant=quant)
        self._entries[name] = entry
        self._save()
        return entry

    def get(self, name: str) -> Optional[ModelEntry]:
        return self._entries.get(name)

    def list_models(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def remove(self, name: str) -> None:
        self._entries.pop(name, None)
        self._save()

    def set_active(self, name: str) -> None:
        for entry in self._entries.values():
            entry.active = False
        if name in self._entries:
            self._entries[name].active = True
        self._save()

    def get_active(self) -> Optional[ModelEntry]:
        for entry in self._entries.values():
            if entry.active:
                return entry
        return None
```

`src/homie_core/model/downloader.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import hf_hub_download, snapshot_download


class ModelDownloader:
    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def download_gguf(
        self,
        repo_id: str,
        filename: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self.models_dir / repo_id.replace("/", "_")),
            resume_download=True,
        )
        return Path(local_path)

    def download_safetensors(
        self,
        repo_id: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(self.models_dir / repo_id.replace("/", "_")),
            ignore_patterns=["*.bin", "*.pt", "*.ot"],
            resume_download=True,
        )
        return Path(local_path)
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/homie_core/model/ tests/unit/test_model/
git commit -m "feat: add model registry and HuggingFace downloader"
```

---

### Task 3.3: GGUF Inference Backend

**Files:**
- Create: `src/homie_core/model/gguf_backend.py`
- Create: `src/homie_core/model/engine.py`
- Create: `tests/unit/test_model/test_engine.py`

**Step 1: Write failing tests**

`tests/unit/test_model/test_engine.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from homie_core.model.engine import ModelEngine


def test_engine_not_loaded_raises():
    engine = ModelEngine()
    with pytest.raises(RuntimeError, match="No model loaded"):
        engine.generate("hello")


def test_engine_load_and_generate():
    engine = ModelEngine()
    mock_backend = MagicMock()
    mock_backend.generate.return_value = "Hello! How can I help?"
    engine._backend = mock_backend
    engine._loaded = True

    result = engine.generate("Say hello")
    assert result == "Hello! How can I help?"
    mock_backend.generate.assert_called_once()


def test_engine_stream():
    engine = ModelEngine()
    mock_backend = MagicMock()
    mock_backend.stream.return_value = iter(["Hello", " world"])
    engine._backend = mock_backend
    engine._loaded = True

    chunks = list(engine.stream("Say hello"))
    assert chunks == ["Hello", " world"]
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

`src/homie_core/model/gguf_backend.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


class GGUFBackend:
    def __init__(self):
        self._model = None

    def load(self, model_path: str | Path, n_ctx: int = 4096, n_gpu_layers: int = -1, **kwargs) -> None:
        from llama_cpp import Llama
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            **kwargs,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        response = self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return response["choices"][0]["message"]["content"]

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        for chunk in self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content

    def unload(self) -> None:
        self._model = None
```

`src/homie_core/model/engine.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from homie_core.model.registry import ModelEntry


class ModelEngine:
    def __init__(self):
        self._backend = None
        self._loaded = False
        self._current_model: Optional[ModelEntry] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def current_model(self) -> Optional[ModelEntry]:
        return self._current_model

    def load(self, entry: ModelEntry, n_ctx: int = 4096, n_gpu_layers: int = -1) -> None:
        if entry.format == "gguf":
            from homie_core.model.gguf_backend import GGUFBackend
            backend = GGUFBackend()
            backend.load(entry.path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
            self._backend = backend
        elif entry.format == "safetensors":
            from homie_core.model.transformers_backend import TransformersBackend
            backend = TransformersBackend()
            backend.load(entry.path)
            self._backend = backend
        else:
            raise ValueError(f"Unsupported format: {entry.format}")
        self._loaded = True
        self._current_model = entry

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> str:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        return self._backend.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)

    def stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> Iterator[str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        return self._backend.stream(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)

    def unload(self) -> None:
        if self._backend:
            self._backend.unload()
        self._backend = None
        self._loaded = False
        self._current_model = None
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/homie_core/model/gguf_backend.py src/homie_core/model/engine.py tests/unit/test_model/test_engine.py
git commit -m "feat: add GGUF inference backend and model engine"
```

---

### Task 3.4: Transformers Backend

**Files:**
- Create: `src/homie_core/model/transformers_backend.py`

**Step 1: Implement**

`src/homie_core/model/transformers_backend.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


class TransformersBackend:
    def __init__(self):
        self._model = None
        self._tokenizer = None

    def load(self, model_path: str | Path, **kwargs) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path_str = str(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(path_str, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            path_str,
            device_map="auto",
            trust_remote_code=True,
            **kwargs,
        )

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> str:
        inputs = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        outputs = self._model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )
        new_tokens = outputs[0][inputs.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> Iterator[str]:
        from transformers import TextIteratorStreamer
        import threading

        inputs = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            "input_ids": inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else None,
            "do_sample": temperature > 0,
            "streamer": streamer,
        }
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        yield from streamer
        thread.join()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
```

**Step 2: Commit**

```bash
git add src/homie_core/model/transformers_backend.py
git commit -m "feat: add transformers/safetensors inference backend"
```

---

## Phase 4: Memory System

### Task 4.1: Working Memory

**Files:**
- Create: `src/homie_core/memory/__init__.py`
- Create: `src/homie_core/memory/working.py`
- Create: `tests/unit/test_memory/__init__.py`
- Create: `tests/unit/test_memory/test_working.py`

**Step 1: Write failing tests**

`tests/unit/test_memory/test_working.py`:
```python
import time
from homie_core.memory.working import WorkingMemory


def test_update_and_get_snapshot():
    wm = WorkingMemory(max_age_seconds=300)
    wm.update("active_app", "VS Code")
    wm.update("active_file", "main.py")
    snap = wm.snapshot()
    assert snap["active_app"] == "VS Code"
    assert snap["active_file"] == "main.py"


def test_conversation_buffer():
    wm = WorkingMemory()
    wm.add_message("user", "hello")
    wm.add_message("assistant", "hi there")
    msgs = wm.get_conversation()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"


def test_conversation_max_length():
    wm = WorkingMemory(max_conversation_turns=3)
    for i in range(5):
        wm.add_message("user", f"msg {i}")
    msgs = wm.get_conversation()
    assert len(msgs) == 3
    assert msgs[0]["content"] == "msg 2"


def test_clear():
    wm = WorkingMemory()
    wm.update("key", "val")
    wm.add_message("user", "hi")
    wm.clear()
    assert wm.snapshot() == {}
    assert wm.get_conversation() == []
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

`src/homie_core/memory/__init__.py`: empty

`src/homie_core/memory/working.py`:
```python
from __future__ import annotations

import threading
from collections import deque
from typing import Any

from homie_core.utils import utc_now


class WorkingMemory:
    def __init__(self, max_age_seconds: int = 300, max_conversation_turns: int = 50):
        self._state: dict[str, Any] = {}
        self._conversation: deque[dict] = deque(maxlen=max_conversation_turns)
        self._max_age = max_age_seconds
        self._lock = threading.Lock()

    def update(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def add_message(self, role: str, content: str) -> None:
        with self._lock:
            self._conversation.append({
                "role": role,
                "content": content,
                "timestamp": utc_now().isoformat(),
            })

    def get_conversation(self) -> list[dict]:
        with self._lock:
            return list(self._conversation)

    def clear(self) -> None:
        with self._lock:
            self._state.clear()
            self._conversation.clear()
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/homie_core/memory/ tests/unit/test_memory/
git commit -m "feat: add working memory with conversation buffer"
```

---

### Task 4.2: Episodic Memory

**Files:**
- Create: `src/homie_core/memory/episodic.py`
- Create: `tests/unit/test_memory/test_episodic.py`

**Step 1: Write failing tests**

`tests/unit/test_memory/test_episodic.py`:
```python
import pytest
from homie_core.memory.episodic import EpisodicMemory
from homie_core.storage.vectors import VectorStore
from homie_core.storage.database import Database


@pytest.fixture
def episodic(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    vs = VectorStore(tmp_path / "chroma")
    vs.initialize()
    em = EpisodicMemory(db=db, vector_store=vs)
    return em


def test_record_episode(episodic):
    eid = episodic.record(
        summary="User spent 2 hours debugging auth module",
        mood="frustrated",
        outcome="fixed",
        context_tags=["work", "coding"],
    )
    assert eid is not None


def test_recall_by_query(episodic):
    episodic.record(summary="Debugged Python auth module", mood="frustrated", context_tags=["work"])
    episodic.record(summary="Listened to lo-fi music while coding", mood="relaxed", context_tags=["music"])
    results = episodic.recall("authentication debugging", n=1)
    assert len(results) == 1
    assert "auth" in results[0]["summary"]


def test_recall_returns_mood(episodic):
    episodic.record(summary="Had a great meeting", mood="happy", context_tags=["work"])
    results = episodic.recall("meeting", n=1)
    assert results[0]["mood"] == "happy"
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

`src/homie_core/memory/episodic.py`:
```python
from __future__ import annotations

import uuid
from typing import Any, Optional

from homie_core.storage.database import Database
from homie_core.storage.vectors import VectorStore


class EpisodicMemory:
    def __init__(self, db: Database, vector_store: VectorStore):
        self._db = db
        self._vs = vector_store

    def record(
        self,
        summary: str,
        mood: Optional[str] = None,
        outcome: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
    ) -> str:
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        self._db.record_episode_meta(
            summary=summary, mood=mood, outcome=outcome, context_tags=context_tags,
        )
        metadata = {}
        if mood:
            metadata["mood"] = mood
        if outcome:
            metadata["outcome"] = outcome
        if context_tags:
            metadata["tags"] = ",".join(context_tags)
        self._vs.add_episode(episode_id, summary, metadata)
        return episode_id

    def recall(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        results = self._vs.query_episodes(query, n=n)
        enriched = []
        for r in results:
            entry = {
                "id": r["id"],
                "summary": r["text"],
                "mood": r.get("metadata", {}).get("mood"),
                "outcome": r.get("metadata", {}).get("outcome"),
                "distance": r.get("distance"),
            }
            enriched.append(entry)
        return enriched

    def delete(self, episode_ids: list[str]) -> None:
        self._vs.delete_episodes(episode_ids)
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/homie_core/memory/episodic.py tests/unit/test_memory/test_episodic.py
git commit -m "feat: add episodic memory with vector-based recall"
```

---

### Task 4.3: Semantic Memory

**Files:**
- Create: `src/homie_core/memory/semantic.py`
- Create: `tests/unit/test_memory/test_semantic.py`

**Step 1: Write failing tests**

`tests/unit/test_memory/test_semantic.py`:
```python
import pytest
from homie_core.memory.semantic import SemanticMemory
from homie_core.storage.database import Database


@pytest.fixture
def semantic(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    return SemanticMemory(db=db)


def test_learn_fact(semantic):
    semantic.learn("User prefers Python", confidence=0.9, tags=["work"])
    facts = semantic.get_facts(min_confidence=0.5)
    assert len(facts) == 1
    assert facts[0]["fact"] == "User prefers Python"


def test_reinforce_increases_confidence(semantic):
    semantic.learn("User likes dark mode", confidence=0.6, tags=["preferences"])
    semantic.reinforce("User likes dark mode", boost=0.1)
    facts = semantic.get_facts()
    matching = [f for f in facts if f["fact"] == "User likes dark mode"]
    assert matching[0]["confidence"] > 0.6


def test_forget_by_topic(semantic):
    semantic.learn("User likes rock music", confidence=0.8, tags=["music"])
    semantic.learn("User prefers Python", confidence=0.9, tags=["work"])
    semantic.forget_topic("music")
    facts = semantic.get_facts()
    assert all("music" not in str(f["tags"]) for f in facts)


def test_get_profile_summary(semantic):
    semantic.set_profile("work", {"role": "software engineer", "languages": ["Python"]})
    profile = semantic.get_profile("work")
    assert profile["role"] == "software engineer"
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

`src/homie_core/memory/semantic.py`:
```python
from __future__ import annotations

from typing import Any, Optional

from homie_core.storage.database import Database
from homie_core.utils import utc_now


class SemanticMemory:
    def __init__(self, db: Database):
        self._db = db

    def learn(self, fact: str, confidence: float = 0.5, tags: Optional[list[str]] = None) -> int:
        return self._db.store_fact(fact, confidence=confidence, tags=tags)

    def get_facts(self, min_confidence: float = 0.0) -> list[dict]:
        return self._db.get_facts(min_confidence=min_confidence)

    def reinforce(self, fact: str, boost: float = 0.05) -> None:
        facts = self._db.get_facts()
        for f in facts:
            if f["fact"] == fact:
                new_conf = min(1.0, f["confidence"] + boost)
                self._db._conn.execute(
                    "UPDATE semantic_memory SET confidence = ?, last_confirmed = ?, source_count = source_count + 1 WHERE id = ?",
                    (new_conf, utc_now().isoformat(), f["id"]),
                )
                self._db._conn.commit()
                return

    def forget_topic(self, tag: str) -> None:
        self._db._conn.execute(
            "UPDATE semantic_memory SET archived = 1 WHERE tags LIKE ?",
            (f'%"{tag}"%',),
        )
        self._db._conn.commit()

    def forget_fact(self, fact_id: int) -> None:
        self._db._conn.execute(
            "UPDATE semantic_memory SET archived = 1 WHERE id = ?", (fact_id,),
        )
        self._db._conn.commit()

    def set_profile(self, domain: str, data: dict[str, Any]) -> None:
        self._db.store_profile(domain, data)

    def get_profile(self, domain: str) -> Optional[dict[str, Any]]:
        return self._db.get_profile(domain)

    def get_all_profiles(self) -> dict[str, dict]:
        rows = self._db._conn.execute("SELECT domain, data FROM profile").fetchall()
        import json
        return {r["domain"]: json.loads(r["data"]) for r in rows}
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/homie_core/memory/semantic.py tests/unit/test_memory/test_semantic.py
git commit -m "feat: add semantic memory with facts, reinforcement, and profiles"
```

---

### Task 4.4: Memory Consolidator

**Files:**
- Create: `src/homie_core/memory/consolidator.py`
- Create: `tests/unit/test_memory/test_consolidator.py`

**Step 1: Write failing tests**

`tests/unit/test_memory/test_consolidator.py`:
```python
import pytest
from unittest.mock import MagicMock
from homie_core.memory.consolidator import MemoryConsolidator
from homie_core.memory.working import WorkingMemory


def test_create_session_digest():
    working = WorkingMemory()
    working.add_message("user", "How do I fix this auth bug?")
    working.add_message("assistant", "Check the token validation logic")
    working.add_message("user", "That fixed it, thanks!")
    working.update("active_app", "VS Code")

    mock_engine = MagicMock()
    mock_engine.generate.return_value = "User debugged an authentication token validation issue in VS Code and resolved it."

    consolidator = MemoryConsolidator(model_engine=mock_engine)
    digest = consolidator.create_session_digest(working)

    assert digest is not None
    assert "summary" in digest
    mock_engine.generate.assert_called_once()


def test_extract_facts_from_digest():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '["User knows Python authentication patterns", "User uses VS Code"]'

    consolidator = MemoryConsolidator(model_engine=mock_engine)
    facts = consolidator.extract_facts("User debugged auth in VS Code using Python")

    assert isinstance(facts, list)
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

`src/homie_core/memory/consolidator.py`:
```python
from __future__ import annotations

import json
from typing import Any, Optional

from homie_core.memory.working import WorkingMemory


class MemoryConsolidator:
    def __init__(self, model_engine):
        self._engine = model_engine

    def create_session_digest(self, working: WorkingMemory) -> dict[str, Any]:
        conversation = working.get_conversation()
        context = working.snapshot()

        if not conversation:
            return {"summary": "", "mood": None, "key_events": []}

        conv_text = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
        ctx_text = ", ".join(f"{k}={v}" for k, v in context.items())

        prompt = f"""Summarize this interaction session in 1-2 sentences. Include what the user was doing, their apparent mood, and the outcome.

Context: {ctx_text}

Conversation:
{conv_text}

Respond with ONLY the summary, nothing else."""

        summary = self._engine.generate(prompt, max_tokens=200, temperature=0.3)

        return {
            "summary": summary.strip(),
            "mood": None,
            "key_events": [],
            "context": context,
        }

    def extract_facts(self, episode_summary: str) -> list[str]:
        prompt = f"""From this session summary, extract any new facts learned about the user. Return a JSON array of strings. If no facts, return [].

Summary: {episode_summary}

Respond with ONLY the JSON array."""

        response = self._engine.generate(prompt, max_tokens=300, temperature=0.2)
        try:
            facts = json.loads(response.strip())
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, str)]
        except json.JSONDecodeError:
            pass
        return []
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/homie_core/memory/consolidator.py tests/unit/test_memory/test_consolidator.py
git commit -m "feat: add memory consolidator for session digests and fact extraction"
```

---

## Phases 5-12: Remaining Implementation

> The remaining phases follow the same TDD pattern. Each is summarized below with task lists. Full code will be written during execution.

---

## Phase 5: Brain & Orchestrator

### Task 5.1: Brain Orchestrator
- **Files:** `src/homie_core/brain/__init__.py`, `src/homie_core/brain/orchestrator.py`, `tests/unit/test_brain/test_orchestrator.py`
- Intent routing: takes user input + working memory context → decides action (respond, query plugin, suggest, execute)
- Injects relevant episodic + semantic memory into LLM prompt
- Manages conversation state

### Task 5.2: Action Planner
- **Files:** `src/homie_core/brain/planner.py`, `tests/unit/test_brain/test_planner.py`
- Converts natural language to structured actions via LLM
- Pydantic schema validation with repair pass
- Supported actions: respond, query_memory, run_plugin, suggest, teach

### Task 5.3: Suggestion Engine
- **Files:** `src/homie_core/brain/suggestion_engine.py`, `tests/unit/test_brain/test_suggestion_engine.py`
- Proactive suggestions based on context + beliefs + confidence scoring
- Adaptive timing: tracks accept/dismiss rates per context
- Suppresses during detected deep work

---

## Phase 6: Context Engine

### Task 6.1: Screen Monitor
- **Files:** `src/homie_core/context/screen_monitor.py`, `tests/unit/test_context/test_screen_monitor.py`
- Win32 API: `GetForegroundWindow`, `GetWindowText`, process name
- Polls every 2-3 seconds, feeds working memory

### Task 6.2: App Tracker
- **Files:** `src/homie_core/context/app_tracker.py`, `tests/unit/test_context/test_app_tracker.py`
- Tracks per-app usage duration, switch frequency
- Detects deep work (single app >45min) vs multitasking

### Task 6.3: File Indexer
- **Files:** `src/homie_core/context/file_indexer.py`, `tests/unit/test_context/test_file_indexer.py`
- watchdog file watcher on configured dirs
- Chunks text files, embeds into ChromaDB incrementally

### Task 6.4: Clipboard Monitor
- **Files:** `src/homie_core/context/clipboard.py`, `tests/unit/test_context/test_clipboard.py`
- Win32 clipboard hooks, history buffer (last 50 entries)

### Task 6.5: Context Aggregator
- **Files:** `src/homie_core/context/aggregator.py`, `tests/unit/test_context/test_aggregator.py`
- Merges all context sources into unified snapshot for working memory

---

## Phase 7: Feedback System

### Task 7.1: Feedback Collector
- **Files:** `src/homie_core/feedback/collector.py`, `tests/unit/test_feedback/test_collector.py`
- Five channels: corrections, preferences, explicit teaching, satisfaction, onboarding
- Stores via database

### Task 7.2: Pattern Detector
- **Files:** `src/homie_core/feedback/patterns.py`, `tests/unit/test_feedback/test_patterns.py`
- Clusters similar corrections, finds temporal/context correlations

### Task 7.3: Belief System
- **Files:** `src/homie_core/feedback/beliefs.py`, `tests/unit/test_feedback/test_beliefs.py`
- Weighted beliefs with confidence, decay, source tracking

### Task 7.4: Contradiction Resolver
- **Files:** `src/homie_core/feedback/contradictions.py`, `tests/unit/test_feedback/test_contradictions.py`
- Context splits, temporal evolution, ambiguity detection

### Task 7.5: Behavioral Adapter
- **Files:** `src/homie_core/feedback/adapter.py`, `tests/unit/test_feedback/test_adapter.py`
- Tunes suggestion timing/content based on belief confidence

### Task 7.6: Self-Reflection Engine
- **Files:** `src/homie_core/feedback/reflection.py`, `tests/unit/test_feedback/test_reflection.py`
- Periodic meta-analysis via LLM, generates internal memos

---

## Phase 8: Behavioral Intelligence

### Task 8.1: Observer Base Class
- **Files:** `src/homie_core/behavioral/base.py`, `tests/unit/test_behavioral/test_base.py`
- Abstract base: `on_tick()`, `get_observations()`, `get_profile_updates()`

### Task 8.2: Media Observer
- **Files:** `src/homie_core/behavioral/media_observer.py`
- Windows media transport APIs, tracks currently playing, skip events

### Task 8.3: Work Observer
- **Files:** `src/homie_core/behavioral/work_observer.py`
- Tracks coding languages, git activity, IDE usage

### Task 8.4: Browsing Observer
- **Files:** `src/homie_core/behavioral/browsing_observer.py`
- Reads browser history SQLite DB (Chrome/Firefox/Edge)

### Task 8.5: Routine Observer
- **Files:** `src/homie_core/behavioral/routine_observer.py`
- Detects daily patterns: wake/sleep, breaks, weekly cycles

### Task 8.6: Emotional Observer
- **Files:** `src/homie_core/behavioral/emotional_observer.py`
- Typing speed, app switching frequency, message tone analysis

### Task 8.7: Social Observer
- **Files:** `src/homie_core/behavioral/social_observer.py`
- Aggregates communication patterns from plugins

### Task 8.8: Habit Detector
- **Files:** `src/homie_core/behavioral/habit_detector.py`, `tests/unit/test_behavioral/test_habit_detector.py`
- Action chain detection, trigger-response pairs, absence detection

### Task 8.9: Profile Synthesizer
- **Files:** `src/homie_core/behavioral/profile_synthesizer.py`, `tests/unit/test_behavioral/test_profile_synthesizer.py`
- Aggregates all observer data into structured user profile domains

---

## Phase 9: Voice Pipeline

### Task 9.1: Audio I/O
- **Files:** `src/homie_core/voice/audio_io.py`, `tests/unit/test_voice/test_audio_io.py`
- PyAudio mic input stream, sounddevice speaker output

### Task 9.2: Wake Word Engine
- **Files:** `src/homie_core/voice/wakeword.py`, `tests/unit/test_voice/test_wakeword.py`
- OpenWakeWord with "Hey Homie" custom wake phrase

### Task 9.3: STT (Speech to Text)
- **Files:** `src/homie_core/voice/stt.py`, `tests/unit/test_voice/test_stt.py`
- faster-whisper with GPU acceleration, VAD for silence detection

### Task 9.4: TTS (Text to Speech)
- **Files:** `src/homie_core/voice/tts.py`, `tests/unit/test_voice/test_tts.py`
- Piper TTS, non-blocking playback, interruption support

### Task 9.5: VAD (Voice Activity Detection)
- **Files:** `src/homie_core/voice/vad.py`, `tests/unit/test_voice/test_vad.py`
- Silence detection to know when user stops speaking

---

## Phase 10: Plugin System

### Task 10.1: Plugin Base Class
- **Files:** `src/homie_core/plugins/base.py`, `tests/unit/test_plugins/test_base.py`
- `HomiePlugin` ABC with `on_activate`, `on_context`, `on_query`, `on_action`, `on_deactivate`

### Task 10.2: Plugin Manager
- **Files:** `src/homie_core/plugins/manager.py`, `tests/unit/test_plugins/test_manager.py`
- Discovery, loading, lifecycle, hot-reload

### Task 10.3: Plugin Permissions
- **Files:** `src/homie_core/plugins/permissions.py`, `tests/unit/test_plugins/test_permissions.py`
- Permission declaration, user approval, enforcement

### Task 10.4: Plugin Sandbox
- **Files:** `src/homie_core/plugins/sandbox.py`, `tests/unit/test_plugins/test_sandbox.py`
- Thread isolation, crash protection, timeout enforcement

### Task 10.5: MCP Host
- **Files:** `src/homie_core/plugins/mcp_host.py`, `tests/unit/test_plugins/test_mcp_host.py`
- MCP server discovery, tool listing, invocation bridge

---

## Phase 11: Application Layer

### Task 11.1: CLI Entry Point
- **Files:** `src/homie_app/cli.py`, `tests/unit/test_app/test_cli.py`
- `homie` command: `homie start`, `homie init`, `homie model *`, `homie plugin *`, `homie backup/restore`

### Task 11.2: Init / Setup Flow
- **Files:** `src/homie_app/init.py`, `tests/unit/test_app/test_init.py`
- `homie init` — hardware detection, model download, config generation
- `homie init --auto` — fully automatic

### Task 11.3: System Tray
- **Files:** `src/homie_app/tray/app.py`
- pystray tray icon, menu (start/stop, voice toggle, settings, quit)
- Global hotkey registration (F9 for voice toggle)

### Task 11.4: Local Dashboard
- **Files:** `src/homie_app/tray/dashboard.py`
- FastAPI: `/api/memory`, `/api/profile`, `/api/beliefs`, `/api/plugins`, `/api/settings`

### Task 11.5: Setup Wizard GUI
- **Files:** `src/homie_app/wizard/gui.py`, `src/homie_app/wizard/steps.py`
- tkinter wizard: hardware results, model download, mic test, plugin selection, onboarding

### Task 11.6: System Prompts
- **Files:** `src/homie_app/prompts/system.py`, `src/homie_app/prompts/onboarding.py`, `src/homie_app/prompts/reflection.py`
- Default personality, onboarding interview questions, reflection prompts

---

## Phase 12: Built-in Plugins (Essential Set)

### Task 12.1: System Plugin
- **Files:** `src/homie_app/plugins/system_plugin.py`
- Running apps, CPU/RAM, notifications

### Task 12.2: Clipboard Plugin
- **Files:** `src/homie_app/plugins/clipboard_plugin.py`
- Clipboard history, search

### Task 12.3: Browser Plugin
- **Files:** `src/homie_app/plugins/browser_plugin.py`
- Chrome/Firefox/Edge history, open tabs

### Task 12.4: IDE Plugin
- **Files:** `src/homie_app/plugins/ide_plugin.py`
- VS Code / JetBrains active project, open files, git status

### Task 12.5: Git Plugin
- **Files:** `src/homie_app/plugins/git_plugin.py`
- Commit history, branch status, diff summaries

### Task 12.6: Email Plugin
- **Files:** `src/homie_app/plugins/email_plugin.py`
- Local IMAP, inbox summary, draft replies

### Task 12.7: Calendar Plugin
- **Files:** `src/homie_app/plugins/calendar_plugin.py`
- CalDAV / ICS, upcoming events, conflicts

### Task 12.8: Notes Plugin
- **Files:** `src/homie_app/plugins/notes_plugin.py`
- Obsidian vault / markdown folder search and summarize

### Task 12.9: Health Plugin
- **Files:** `src/homie_app/plugins/health_plugin.py`
- Break reminders, screen time tracking

### Task 12.10: Music Plugin
- **Files:** `src/homie_app/plugins/music_plugin.py`
- Windows media transport, now playing, play/pause/skip

### Task 12.11: Shortcuts Plugin
- **Files:** `src/homie_app/plugins/shortcuts_plugin.py`
- User-defined trigger → action macros

### Task 12.12: Workflows Plugin
- **Files:** `src/homie_app/plugins/workflows_plugin.py`
- Multi-step automations from YAML definitions

---

## Dependency Graph

```
Phase 1 (Scaffold + Config)
    ↓
Phase 2 (Storage: SQLite + ChromaDB + Backup)
    ↓
Phase 3 (Hardware + Model Engine)
    ↓
Phase 4 (Memory: Working + Episodic + Semantic + Consolidator)
    ↓
Phase 5 (Brain: Orchestrator + Planner + Suggestions)
    ↓
Phase 6 (Context: Screen + Apps + Files + Clipboard)  ← can parallel with Phase 7
Phase 7 (Feedback: 5 channels + intelligent loop)     ← can parallel with Phase 6
    ↓
Phase 8 (Behavioral Intelligence: 6 observers + habits + profile)
    ↓
Phase 9 (Voice Pipeline)  ← can parallel with Phase 10
Phase 10 (Plugin System)  ← can parallel with Phase 9
    ↓
Phase 11 (App: CLI + Tray + Dashboard + Wizard)
    ↓
Phase 12 (Built-in Plugins)
```

---

## Estimated Scope

| Phase | Tasks | Estimated Files |
|-------|-------|----------------|
| 1. Scaffold & Config | 3 | 6 |
| 2. Storage | 3 | 8 |
| 3. Hardware & Model | 4 | 10 |
| 4. Memory | 4 | 10 |
| 5. Brain | 3 | 6 |
| 6. Context | 5 | 10 |
| 7. Feedback | 6 | 12 |
| 8. Behavioral | 9 | 18 |
| 9. Voice | 5 | 10 |
| 10. Plugin System | 5 | 10 |
| 11. App Layer | 6 | 10 |
| 12. Built-in Plugins | 12 | 12 |
| **Total** | **65** | **~122** |
