"""WebSocket sync server for Homie LAN communication."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import string
from pathlib import Path
from typing import Optional

from homie_core.network.protocol import (
    ProtocolMessage, HelloMessage, InferenceRequest, InferenceResponse,
    StatusMessage, UnpairMessage, PROTOCOL_VERSION,
)

logger = logging.getLogger(__name__)


class SyncServer:
    """WebSocket server for LAN device sync."""

    def __init__(
        self,
        device_id: str,
        device_name: str,
        port: int = 8765,
        data_dir: Path | str | None = None,
    ):
        self.device_id = device_id
        self.device_name = device_name
        self.port = port
        self._data_dir = Path(data_dir) if data_dir else Path.home() / ".homie"
        self._devices_file = self._data_dir / "paired_devices.json"
        self._pairing_code: str = ""
        self._paired_devices: dict[str, dict] = {}
        self._connected_clients: dict[str, object] = {}
        self._inference_handler = None
        self._load_paired_devices()

    @property
    def paired_devices(self) -> dict[str, dict]:
        return dict(self._paired_devices)

    def set_inference_handler(self, handler) -> None:
        self._inference_handler = handler

    def generate_pairing_code(self) -> str:
        self._pairing_code = "".join(random.choices(string.digits, k=6))
        return self._pairing_code

    def verify_pairing_code(self, code: str) -> bool:
        return code == self._pairing_code and self._pairing_code != ""

    def add_paired_device(self, device_id: str, name: str, public_key: str) -> None:
        self._paired_devices[device_id] = {"name": name, "public_key": public_key}
        self._save_paired_devices()

    def remove_paired_device(self, device_id: str) -> None:
        self._paired_devices.pop(device_id, None)
        self._save_paired_devices()

    def _load_paired_devices(self) -> None:
        if self._devices_file.exists():
            try:
                self._paired_devices = json.loads(self._devices_file.read_text())
            except Exception:
                self._paired_devices = {}

    def _save_paired_devices(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._devices_file.write_text(json.dumps(self._paired_devices, indent=2))

    async def handle_connection(self, websocket) -> None:
        device_id = None
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=10)
            msg = ProtocolMessage.from_json(raw)
            if not isinstance(msg, HelloMessage):
                await websocket.close(1002, "Expected hello message")
                return
            device_id = msg.device_id
            if device_id not in self._paired_devices:
                await websocket.close(1008, "Device not paired")
                return
            self._connected_clients[device_id] = websocket
            logger.info("Device connected: %s", device_id)
            hello = HelloMessage(device_id=self.device_id, device_name=self.device_name)
            await websocket.send(hello.to_json())
            async for raw in websocket:
                try:
                    msg = ProtocolMessage.from_json(raw)
                    await self._handle_message(msg, websocket)
                except Exception as e:
                    logger.error("Error handling message: %s", e)
        except Exception as e:
            logger.error("Connection error: %s", e)
        finally:
            if device_id:
                self._connected_clients.pop(device_id, None)
                logger.info("Device disconnected: %s", device_id)

    async def _handle_message(self, msg: ProtocolMessage, websocket) -> None:
        if isinstance(msg, InferenceRequest):
            await self._handle_inference(msg, websocket)
        elif isinstance(msg, StatusMessage):
            logger.info(
                "Status from %s: model=%s, battery=%s",
                msg.device_id, msg.model_name, msg.battery_level,
            )
        elif isinstance(msg, UnpairMessage):
            self.remove_paired_device(msg.device_id)
            await websocket.close(1000, "Unpaired")

    async def _handle_inference(self, req: InferenceRequest, websocket) -> None:
        if not self._inference_handler:
            resp = InferenceResponse(request_id=req.request_id, error="No inference handler available")
        else:
            try:
                result = self._inference_handler.generate(
                    req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    stop=req.stop,
                )
                resp = InferenceResponse(request_id=req.request_id, content=result, source="lan")
            except Exception as e:
                resp = InferenceResponse(request_id=req.request_id, error=str(e))
        await websocket.send(resp.to_json())

    async def start(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed — LAN sync server disabled")
            return
        logger.info("Starting sync server on port %d", self.port)
        async with websockets.serve(self.handle_connection, "0.0.0.0", self.port):
            await asyncio.Future()
