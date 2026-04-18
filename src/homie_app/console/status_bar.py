"""Enhanced status bar — shows model, memory, mesh, system, and time."""
from __future__ import annotations

import os
import sys
from datetime import datetime

from rich.rule import Rule


def _get_gpu_short() -> str:
    """Quick GPU usage string."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            used, total = r.stdout.strip().split(", ")
            return f"GPU:{used}/{total}MB"
    except Exception:
        pass
    return ""


def _get_ram_short() -> str:
    """Quick RAM usage string."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return f"RAM:{mem.used // (1024 ** 3)}/{mem.total // (1024 ** 3)}G"
    except Exception:
        return ""


def print_status_bar(
    rc,
    model_name: str = "",
    memory_count: int = 0,
    project: str = "",
    mesh_ctx=None,
) -> None:
    """Print an enhanced status bar with system info."""
    now = datetime.now().strftime("%H:%M")
    parts: list[str] = []

    # Model
    if model_name:
        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        parts.append(f"[homie.brand]{short_name}[/]")
    elif mesh_ctx and hasattr(mesh_ctx, "identity"):
        parts.append(f"[homie.brand]Homie[/]")

    # Mesh node
    if mesh_ctx:
        role = mesh_ctx.identity.role if hasattr(mesh_ctx, "identity") else "?"
        parts.append(f"[homie.tool]{role}[/]")

    # Memory
    if memory_count:
        parts.append(f"[homie.memory]{memory_count} facts[/]")

    # System resources (compact)
    ram = _get_ram_short()
    if ram:
        parts.append(f"[homie.dim]{ram}[/]")
    gpu = _get_gpu_short()
    if gpu:
        parts.append(f"[homie.dim]{gpu}[/]")

    # Project
    if project:
        parts.append(f"[homie.tool]{project}[/]")

    # Time
    parts.append(f"[homie.dim]{now}[/]")

    rc.print(Rule(" | ".join(parts), style="homie.dim", align="left"))
