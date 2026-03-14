"""HomieSync protocol — message types for LAN device communication."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

PROTOCOL_VERSION = "1.0.0"


def _major(version: str) -> int:
    return int(version.split(".")[0])


@dataclass
class ProtocolMessage:
    """Base class for all protocol messages."""
    type: str = ""

    def to_json(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "type"}
        return json.dumps({
            "type": self.type,
            "protocol_version": PROTOCOL_VERSION,
            "payload": payload,
        })

    @staticmethod
    def from_json(data: str) -> "ProtocolMessage":
        obj = json.loads(data)
        version = obj.get("protocol_version", "1.0.0")
        if _major(version) != _major(PROTOCOL_VERSION):
            raise ValueError(f"Incompatible protocol version: {version} (expected {PROTOCOL_VERSION} major)")
        msg_type = obj["type"]
        payload = obj.get("payload", {})
        registry = {
            "hello": HelloMessage,
            "inference_request": InferenceRequest,
            "inference_response": InferenceResponse,
            "status": StatusMessage,
            "memory_sync": MemorySyncMessage,
            "conversation_sync": ConversationSyncMessage,
            "command": CommandMessage,
            "command_result": CommandResultMessage,
            "file_transfer_init": FileTransferInit,
            "file_transfer_ack": FileTransferAck,
            "file_transfer_cancel": FileTransferCancel,
            "unpair": UnpairMessage,
        }
        cls = registry.get(msg_type)
        if cls is None:
            raise ValueError(f"Unknown message type: {msg_type}")
        return cls(**payload)


@dataclass
class HelloMessage(ProtocolMessage):
    device_id: str = ""
    device_name: str = ""
    type: str = field(default="hello", init=False)


@dataclass
class InferenceRequest(ProtocolMessage):
    request_id: str = ""
    prompt: str = ""
    max_tokens: int = 1024
    temperature: float = 0.7
    stop: Optional[list[str]] = None
    type: str = field(default="inference_request", init=False)


@dataclass
class InferenceResponse(ProtocolMessage):
    request_id: str = ""
    content: str = ""
    source: str = ""
    error: str = ""
    type: str = field(default="inference_response", init=False)


@dataclass
class StatusMessage(ProtocolMessage):
    device_id: str = ""
    model_loaded: bool = False
    model_name: str = ""
    daemon_running: bool = False
    battery_level: Optional[int] = None
    type: str = field(default="status", init=False)


@dataclass
class MemorySyncMessage(ProtocolMessage):
    device_id: str = ""
    sync_type: str = ""
    data_type: str = ""
    entries: list = field(default_factory=list)
    sync_version: int = 0
    type: str = field(default="memory_sync", init=False)


@dataclass
class ConversationSyncMessage(ProtocolMessage):
    device_id: str = ""
    messages: list = field(default_factory=list)
    sync_version: int = 0
    type: str = field(default="conversation_sync", init=False)


@dataclass
class CommandMessage(ProtocolMessage):
    command_id: str = ""
    command: str = ""
    args: dict = field(default_factory=dict)
    type: str = field(default="command", init=False)


@dataclass
class CommandResultMessage(ProtocolMessage):
    command_id: str = ""
    result: str = ""
    success: bool = True
    type: str = field(default="command_result", init=False)


@dataclass
class FileTransferInit(ProtocolMessage):
    filename: str = ""
    size: int = 0
    sha256: str = ""
    chunk_size: int = 1048576
    transfer_port: int = 0
    type: str = field(default="file_transfer_init", init=False)


@dataclass
class FileTransferAck(ProtocolMessage):
    bytes_received: int = 0
    type: str = field(default="file_transfer_ack", init=False)


@dataclass
class FileTransferCancel(ProtocolMessage):
    reason: str = ""
    type: str = field(default="file_transfer_cancel", init=False)


@dataclass
class UnpairMessage(ProtocolMessage):
    device_id: str = ""
    type: str = field(default="unpair", init=False)
