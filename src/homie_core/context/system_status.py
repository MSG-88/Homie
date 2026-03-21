from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemStatus:
    disk_used_gb: float
    disk_total_gb: float
    disk_pct: float
    battery_pct: Optional[float]
    battery_plugged: Optional[bool]
    net_sent_mb: float
    net_recv_mb: float
    cpu_pct: float
    ram_pct: float


def get_system_status(watch_path: str = "/") -> SystemStatus:
    try:
        import psutil
    except ImportError:
        return SystemStatus(0, 0, 0, None, None, 0, 0, 0, 0)
    disk = psutil.disk_usage(watch_path)
    batt = psutil.sensors_battery()
    net = psutil.net_io_counters()
    mem = psutil.virtual_memory()
    return SystemStatus(
        disk_used_gb=disk.used / (1024**3),
        disk_total_gb=disk.total / (1024**3),
        disk_pct=disk.percent,
        battery_pct=batt.percent if batt else None,
        battery_plugged=batt.power_plugged if batt else None,
        net_sent_mb=net.bytes_sent / (1024**2),
        net_recv_mb=net.bytes_recv / (1024**2),
        cpu_pct=psutil.cpu_percent(interval=None),
        ram_pct=mem.percent,
    )
