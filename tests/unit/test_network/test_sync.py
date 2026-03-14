"""Tests for the memory sync engine."""
import pytest
from homie_core.network.sync import SyncEngine, SyncEntry, SyncResult


def test_sync_engine_init():
    engine = SyncEngine(device_id="desktop-1")
    assert engine.device_id == "desktop-1"
    assert engine.get_sync_version("conversation") == 0


def test_conversation_merge_append_only():
    engine = SyncEngine(device_id="desktop-1")
    local = [
        SyncEntry(id="msg-1", device_id="desktop-1", timestamp=100, data={"content": "hello"}),
        SyncEntry(id="msg-2", device_id="desktop-1", timestamp=200, data={"content": "world"}),
    ]
    remote = [
        SyncEntry(id="msg-3", device_id="phone-1", timestamp=150, data={"content": "hi"}),
    ]
    result = engine.merge_conversations(local, remote)
    assert len(result.merged) == 3
    # Should be sorted by timestamp
    assert result.merged[0].id == "msg-1"
    assert result.merged[1].id == "msg-3"
    assert result.merged[2].id == "msg-2"
    assert result.new_entries == 1


def test_conversation_merge_deduplicates():
    engine = SyncEngine(device_id="desktop-1")
    entry = SyncEntry(id="msg-1", device_id="desktop-1", timestamp=100, data={"content": "hello"})
    result = engine.merge_conversations([entry], [entry])
    assert len(result.merged) == 1


def test_episodic_merge_dedup_by_hash():
    engine = SyncEngine(device_id="desktop-1")
    local = [
        SyncEntry(id="ep-1", device_id="desktop-1", timestamp=100, data={"content": "memory A"}, content_hash="hash-a"),
    ]
    remote = [
        SyncEntry(id="ep-2", device_id="phone-1", timestamp=200, data={"content": "memory A"}, content_hash="hash-a"),
        SyncEntry(id="ep-3", device_id="phone-1", timestamp=300, data={"content": "memory B"}, content_hash="hash-b"),
    ]
    result = engine.merge_episodic(local, remote)
    assert len(result.merged) == 2  # hash-a deduped, hash-b added
    assert result.new_entries == 1


def test_settings_merge_lamport_wins():
    engine = SyncEngine(device_id="desktop-1")
    local = {"theme": {"value": "dark", "lamport": 5, "device_id": "desktop-1"}}
    remote = {"theme": {"value": "light", "lamport": 3, "device_id": "phone-1"}}
    result = engine.merge_settings(local, remote)
    assert result["theme"]["value"] == "dark"  # Higher lamport wins


def test_settings_merge_lamport_tie_breaks_by_device_id():
    engine = SyncEngine(device_id="desktop-1")
    local = {"theme": {"value": "dark", "lamport": 5, "device_id": "desktop-1"}}
    remote = {"theme": {"value": "light", "lamport": 5, "device_id": "phone-1"}}
    result = engine.merge_settings(local, remote)
    # Lexicographic tie break: "phone-1" > "desktop-1"
    assert result["theme"]["value"] == "light"


def test_semantic_merge_with_tombstone():
    engine = SyncEngine(device_id="desktop-1")
    local = [
        SyncEntry(id="sem-1", device_id="desktop-1", timestamp=100, data={"embedding": [0.1, 0.2]}),
    ]
    remote = [
        SyncEntry(id="sem-1", device_id="phone-1", timestamp=200, data=None, tombstone=True, lamport=3),
    ]
    result = engine.merge_semantic(local, remote)
    assert len(result.merged) == 0  # Tombstoned
    assert result.deleted == 1
