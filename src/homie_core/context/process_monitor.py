from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProcessSnapshot:
    pid: int
    name: str
    cpu_pct: float
    mem_mb: float
    status: str


def get_top_processes(n: int = 15) -> list[ProcessSnapshot]:
    """Return top N processes by CPU usage."""
    try:
        import psutil
    except ImportError:
        return []
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'status']):
        try:
            cpu = p.cpu_percent(interval=None)
            mem = p.memory_info().rss / (1024**2)
            procs.append(ProcessSnapshot(p.pid, p.name(), cpu, mem, p.status()))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(procs, key=lambda x: x.cpu_pct, reverse=True)[:n]
