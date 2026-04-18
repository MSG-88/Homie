import json, time
from homie_core.mesh.transport import sign_message, verify_signature, MeshMessage, MeshMessageType

def test_sign_message():
    sig = sign_message(b"secret", "hello world")
    assert isinstance(sig, str)
    assert len(sig) == 64

def test_verify_valid_signature():
    secret = b"my-secret-key"
    sig = sign_message(secret, "test payload")
    assert verify_signature(secret, "test payload", sig) is True

def test_verify_invalid_signature():
    assert verify_signature(b"my-secret-key", "data", "bad_signature") is False

def test_verify_wrong_secret():
    sig = sign_message(b"key-a", "data")
    assert verify_signature(b"key-b", "data", sig) is False

def test_mesh_message_types():
    assert MeshMessageType.SYNC_REQUEST == "sync_request"
    assert MeshMessageType.SYNC_RESPONSE == "sync_response"
    assert MeshMessageType.EVENT_PUSH == "event_push"
    assert MeshMessageType.HEARTBEAT == "heartbeat"
    assert MeshMessageType.INFERENCE_REQUEST == "inference_request"
    assert MeshMessageType.INFERENCE_RESPONSE == "inference_response"

def test_mesh_message_round_trip():
    secret = b"test-key"
    msg = MeshMessage.create(msg_type=MeshMessageType.HEARTBEAT, node_id="node-1",
                             payload={"status": "online"}, secret=secret)
    raw = msg.to_json()
    parsed = MeshMessage.from_json(raw)
    assert parsed.msg_type == MeshMessageType.HEARTBEAT
    assert parsed.node_id == "node-1"
    assert parsed.payload == {"status": "online"}
    assert parsed.verify(secret) is True

def test_mesh_message_rejects_tampered():
    secret = b"key"
    msg = MeshMessage.create(msg_type=MeshMessageType.HEARTBEAT, node_id="n1",
                             payload={"ok": True}, secret=secret)
    d = json.loads(msg.to_json())
    d["payload"]["ok"] = False
    parsed = MeshMessage.from_json(json.dumps(d))
    assert parsed.verify(secret) is False

def test_mesh_message_rejects_expired():
    secret = b"key"
    msg = MeshMessage.create(msg_type=MeshMessageType.HEARTBEAT, node_id="n1",
                             payload={}, secret=secret)
    msg.timestamp = time.time() - 120
    parsed = MeshMessage.from_json(msg.to_json())
    assert parsed.verify(secret, max_age_seconds=60) is False
