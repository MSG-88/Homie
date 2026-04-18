from __future__ import annotations

from typing import Optional

from homie_core.mesh.unified_user_model import UnifiedUserModel


class CrossDevicePerceiver:
    """Brain PERCEIVE integration — surfaces cross-device context and handoff messages."""

    def __init__(self, unified_model: Optional[UnifiedUserModel] = None):
        self._model = unified_model
        self._pending_handoff: Optional[dict] = None

    def set_pending_handoff(self, handoff: dict) -> None:
        self._pending_handoff = handoff

    def get_context_block(self) -> str:
        parts = []
        if self._model:
            block = self._model.to_context_block()
            if block:
                parts.append(block)
        if self._pending_handoff:
            msg = self._pending_handoff.get("message", "")
            if msg:
                parts.append(f"\n[DEVICE SWITCH]\n{msg}")
            self._pending_handoff = None
        return "\n".join(parts)
