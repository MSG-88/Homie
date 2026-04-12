from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional

import psutil

from homie_core.utils import utc_now


@dataclass
class NodeContext:
    node_id: str
    node_name: str
    active_window: str = ""
    active_process: str = ""
    activity_type: str = "idle"
    activity_confidence: float = 0.0
    minutes_active: float = 0.0
    idle_minutes: float = 0.0
    flow_score: float = 0.5
    session_start: str = ""
    cpu_usage: float = 0.0
    gpu_usage: Optional[float] = None
    ram_usage_gb: float = 0.0
    battery_pct: Optional[float] = None
    last_updated: str = field(default_factory=lambda: utc_now().isoformat())

    @property
    def is_idle(self) -> bool:
        return self.idle_minutes >= 5.0 and self.minutes_active == 0.0

    def summary(self) -> str:
        parts = [f"{self.node_name}:"]
        if self.is_idle:
            parts.append("idle")
        else:
            if self.activity_type != "idle":
                parts.append(self.activity_type)
            if self.active_window:
                parts.append(f"in {self.active_window}")
            if self.minutes_active > 1:
                parts.append(f"({int(self.minutes_active)}min)")
            if self.flow_score > 0.7:
                parts.append("[focused]")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "active_window": self.active_window,
            "active_process": self.active_process,
            "activity_type": self.activity_type,
            "activity_confidence": self.activity_confidence,
            "minutes_active": self.minutes_active,
            "idle_minutes": self.idle_minutes,
            "flow_score": self.flow_score,
            "session_start": self.session_start,
            "cpu_usage": self.cpu_usage,
            "gpu_usage": self.gpu_usage,
            "ram_usage_gb": self.ram_usage_gb,
            "battery_pct": self.battery_pct,
            "last_updated": self.last_updated,
        }

    def to_event_payload(self) -> dict:
        return self.to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> NodeContext:
        return cls(
            node_id=d["node_id"],
            node_name=d["node_name"],
            active_window=d.get("active_window", ""),
            active_process=d.get("active_process", ""),
            activity_type=d.get("activity_type", "idle"),
            activity_confidence=d.get("activity_confidence", 0.0),
            minutes_active=d.get("minutes_active", 0.0),
            idle_minutes=d.get("idle_minutes", 0.0),
            flow_score=d.get("flow_score", 0.5),
            session_start=d.get("session_start", ""),
            cpu_usage=d.get("cpu_usage", 0.0),
            gpu_usage=d.get("gpu_usage"),
            ram_usage_gb=d.get("ram_usage_gb", 0.0),
            battery_pct=d.get("battery_pct"),
            last_updated=d.get("last_updated", ""),
        )


def collect_local_context(node_id: str, node_name: str) -> NodeContext:
    mem = psutil.virtual_memory()
    return NodeContext(
        node_id=node_id,
        node_name=node_name,
        cpu_usage=psutil.cpu_percent(interval=0.1),
        ram_usage_gb=round(mem.used / (1024 ** 3), 1),
        battery_pct=_get_battery(),
    )


def _get_battery() -> Optional[float]:
    try:
        bat = psutil.sensors_battery()
        return bat.percent if bat else None
    except Exception:
        return None
