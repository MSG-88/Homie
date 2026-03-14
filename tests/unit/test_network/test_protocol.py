"""Tests for HomieSync protocol message types."""
import json
import pytest
from homie_core.network.protocol import (
    ProtocolMessage, InferenceRequest, InferenceResponse,
    StatusMessage, HelloMessage, MemorySyncMessage,
    ConversationSyncMessage, FileTransferInit,
    PROTOCOL_VERSION,
)


def test_hello_message_serialization():
    msg = HelloMessage(device_id="desktop-1", device_name="My PC")
    data = msg.to_json()
    parsed = json.loads(data)
    assert parsed["type"] == "hello"
    assert parsed["protocol_version"] == PROTOCOL_VERSION
    assert parsed["payload"]["device_id"] == "desktop-1"


def test_inference_request_roundtrip():
    req = InferenceRequest(request_id="req-001", prompt="Hello", max_tokens=100, temperature=0.7)
    data = req.to_json()
    parsed = ProtocolMessage.from_json(data)
    assert isinstance(parsed, InferenceRequest)
    assert parsed.prompt == "Hello"
    assert parsed.request_id == "req-001"


def test_inference_response_roundtrip():
    resp = InferenceResponse(request_id="req-001", content="Hi there!", source="local")
    data = resp.to_json()
    parsed = ProtocolMessage.from_json(data)
    assert isinstance(parsed, InferenceResponse)
    assert parsed.content == "Hi there!"


def test_status_message():
    msg = StatusMessage(device_id="phone-1", model_loaded=True, model_name="Qwen-1.5B", daemon_running=True, battery_level=85)
    data = msg.to_json()
    parsed = ProtocolMessage.from_json(data)
    assert isinstance(parsed, StatusMessage)
    assert parsed.battery_level == 85


def test_memory_sync_message():
    msg = MemorySyncMessage(device_id="d1", sync_type="incremental", data_type="episodic", entries=[{"id": "1"}], sync_version=5)
    data = msg.to_json()
    parsed = ProtocolMessage.from_json(data)
    assert isinstance(parsed, MemorySyncMessage)
    assert parsed.sync_version == 5


def test_conversation_sync_message():
    msg = ConversationSyncMessage(device_id="d1", messages=[{"id": "m1", "content": "hi"}], sync_version=3)
    data = msg.to_json()
    parsed = ProtocolMessage.from_json(data)
    assert isinstance(parsed, ConversationSyncMessage)
    assert len(parsed.messages) == 1


def test_file_transfer_init():
    msg = FileTransferInit(filename="model.gguf", size=800_000_000, sha256="abc123", chunk_size=1048576, transfer_port=9000)
    data = msg.to_json()
    parsed = ProtocolMessage.from_json(data)
    assert isinstance(parsed, FileTransferInit)
    assert parsed.size == 800_000_000


def test_unknown_message_type():
    data = json.dumps({"type": "unknown_type", "protocol_version": "1.0.0", "payload": {}})
    with pytest.raises(ValueError, match="Unknown message type"):
        ProtocolMessage.from_json(data)


def test_major_version_mismatch():
    data = json.dumps({"type": "hello", "protocol_version": "99.0.0", "payload": {"device_id": "x", "device_name": "y"}})
    with pytest.raises(ValueError, match="Incompatible protocol version"):
        ProtocolMessage.from_json(data)
