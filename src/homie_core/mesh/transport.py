"""HMAC-authenticated message transport for mesh communication."""
from __future__ import annotations
import hashlib, hmac, json, time
from dataclasses import dataclass

def sign_message(secret: bytes, payload: str) -> str:
    return hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()

def verify_signature(secret: bytes, payload: str, signature: str) -> bool:
    expected = sign_message(secret, payload)
    return hmac.compare_digest(expected, signature)

class MeshMessageType:
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    EVENT_PUSH = "event_push"
    HEARTBEAT = "heartbeat"
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"

@dataclass
class MeshMessage:
    msg_type: str
    node_id: str
    payload: dict
    timestamp: float
    signature: str = ""

    @classmethod
    def create(cls, msg_type: str, node_id: str, payload: dict, secret: bytes) -> MeshMessage:
        ts = time.time()
        msg = cls(msg_type=msg_type, node_id=node_id, payload=payload, timestamp=ts)
        msg.signature = sign_message(secret, msg._signable_string())
        return msg

    def verify(self, secret: bytes, max_age_seconds: float = 60.0) -> bool:
        if max_age_seconds > 0 and (time.time() - self.timestamp) > max_age_seconds:
            return False
        return verify_signature(secret, self._signable_string(), self.signature)

    def _signable_string(self) -> str:
        payload_str = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
        return f"{self.msg_type}:{self.node_id}:{payload_str}:{self.timestamp}"

    def to_json(self) -> str:
        return json.dumps({"msg_type": self.msg_type, "node_id": self.node_id,
                           "payload": self.payload, "timestamp": self.timestamp,
                           "signature": self.signature})

    @classmethod
    def from_json(cls, raw: str) -> MeshMessage:
        d = json.loads(raw)
        return cls(msg_type=d["msg_type"], node_id=d["node_id"], payload=d["payload"],
                   timestamp=d["timestamp"], signature=d.get("signature", ""))
