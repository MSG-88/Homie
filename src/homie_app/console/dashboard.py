"""Homie system dashboard — rich terminal UI showing system state.

Displays a live overview panel with:
- Node identity and mesh status
- System resources (CPU, RAM, GPU, disk)
- Model info and inference status
- Network peers and connectivity
- Recent events and feedback stats
- Training readiness
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from typing import Optional

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _safe(fn, default="N/A"):
    """Run fn, return default on any error."""
    try:
        return fn()
    except Exception:
        return default


def build_system_panel() -> Panel:
    """Build the system resources panel."""
    import psutil
    import shutil

    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    disk_path = "C:" + os.sep if sys.platform == "win32" else "/"
    disk = shutil.disk_usage(disk_path)

    # GPU info
    gpu_name = "N/A"
    gpu_mem = ""
    gpu_temp = ""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.free,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(", ")
            gpu_name = parts[0].strip()
            used_mb = int(parts[1].strip())
            free_mb = int(parts[2].strip())
            temp = parts[3].strip()
            gpu_mem = f"{used_mb}MB / {used_mb + free_mb}MB"
            gpu_temp = f"{temp}C"
    except Exception:
        pass

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold cyan", width=10)
    table.add_column(style="white")

    table.add_row("CPU", f"{cpu}% ({psutil.cpu_count()} cores)")
    table.add_row("RAM", f"{mem.used / (1024 ** 3):.1f} / {mem.total / (1024 ** 3):.1f} GB ({mem.percent}%)")
    table.add_row("Disk", f"{disk.free / (1024 ** 3):.0f} GB free")
    if gpu_name != "N/A":
        table.add_row("GPU", gpu_name)
        table.add_row("VRAM", gpu_mem)
        table.add_row("Temp", gpu_temp)

    return Panel(table, title="[bold]System[/bold]", border_style="cyan", width=45)


def build_node_panel(mesh_ctx=None) -> Panel:
    """Build the node identity and mesh panel."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold green", width=10)
    table.add_column(style="white")

    if mesh_ctx:
        table.add_row("Node", mesh_ctx.node_name)
        table.add_row("ID", mesh_ctx.node_id[:12] + "...")
        table.add_row("Role", mesh_ctx.identity.role)
        table.add_row("Score", f"{mesh_ctx.capabilities.capability_score():.0f}")
        table.add_row("Mesh", mesh_ctx.identity.mesh_id or "standalone")

        # Health
        from homie_core.mesh.health import MeshHealthChecker
        health = MeshHealthChecker(mesh_context=mesh_ctx).run_all()
        table.add_row("Health", f"[bold green]{health.status}[/]" if health.healthy else f"[bold red]{health.status}[/]")
    else:
        table.add_row("Mesh", "[dim]not initialized[/dim]")

    return Panel(table, title="[bold]Node[/bold]", border_style="green", width=45)


def build_model_panel(engine=None, config=None) -> Panel:
    """Build the model info panel."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold yellow", width=10)
    table.add_column(style="white")

    if config:
        table.add_row("Backend", config.llm.backend)
        model_name = os.path.basename(config.llm.model_path)
        table.add_row("Model", model_name)
    if engine and engine.is_loaded:
        table.add_row("Status", "[bold green]loaded[/]")
        if hasattr(engine, "_current_model") and engine._current_model:
            table.add_row("Params", engine._current_model.params or "?")
    else:
        table.add_row("Status", "[bold red]not loaded[/]")

    return Panel(table, title="[bold]Model[/bold]", border_style="yellow", width=45)


def build_network_panel() -> Panel:
    """Build the network peers panel."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold magenta", width=18)
    table.add_column(style="white")

    try:
        r = subprocess.run(["tailscale", "status"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 2:
                ip = parts[0]
                name = parts[1]
                if "active" in line:
                    table.add_row(f"[green]{name}[/]", f"{ip} [green]active[/]")
                elif "offline" in line:
                    table.add_row(f"[dim]{name}[/]", f"[dim]{ip} offline[/]")
                else:
                    table.add_row(name, ip)
    except Exception:
        table.add_row("[dim]Tailscale[/]", "[dim]not available[/]")

    return Panel(table, title="[bold]Network[/bold]", border_style="magenta", width=45)


def build_stats_panel(mesh_ctx=None) -> Panel:
    """Build events and training stats panel."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold blue", width=10)
    table.add_column(style="white")

    if mesh_ctx:
        table.add_row("Events", str(mesh_ctx.mesh_manager.event_count()))
        table.add_row("Feedback", str(mesh_ctx.feedback_store.total_count()))
        summary = mesh_ctx.training_trigger.get_summary()
        table.add_row("SFT", str(summary["sft_pairs"]))
        table.add_row("DPO", str(summary["dpo_pairs"]))
        ready = summary["ready"]
        table.add_row("Train", "[bold green]READY[/]" if ready else f"need {500 - summary['total_signals']} more")
    else:
        table.add_row("Stats", "[dim]unavailable[/]")

    return Panel(table, title="[bold]Training[/bold]", border_style="blue", width=45)


def show_dashboard(rc, engine=None, config=None, mesh_ctx=None) -> None:
    """Display the full system dashboard."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rc.print()
    rc.print(Panel(
        Text(f"Homie Dashboard - {now}", justify="center"),
        style="bold cyan", width=92,
    ))

    # Row 1: System + Node
    try:
        sys_panel = build_system_panel()
        node_panel = build_node_panel(mesh_ctx)
        rc.print(Columns([sys_panel, node_panel], padding=1))
    except Exception:
        rc.print("[dim]Dashboard panels unavailable[/]")

    # Row 2: Model + Network
    try:
        model_panel = build_model_panel(engine, config)
        net_panel = build_network_panel()
        rc.print(Columns([model_panel, net_panel], padding=1))
    except Exception:
        pass

    # Row 3: Stats
    try:
        stats_panel = build_stats_panel(mesh_ctx)
        rc.print(stats_panel)
    except Exception:
        pass

    rc.print()
