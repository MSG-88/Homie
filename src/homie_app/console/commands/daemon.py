"""Handler for /daemon slash command — manage background service."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter

_PID_FILE_REL = ".homie/daemon.pid"
_LOG_FILE_REL = ".homie/daemon.log"


def _pid_file():
    from pathlib import Path
    return Path.home() / _PID_FILE_REL


def _log_file():
    from pathlib import Path
    return Path.home() / _LOG_FILE_REL


def _is_running() -> tuple[bool, int | None]:
    """Check if daemon is running. Returns (is_running, pid)."""
    pid_file = _pid_file()
    if not pid_file.exists():
        return False, None
    try:
        pid = int(pid_file.read_text().strip())
        import psutil
        proc = psutil.Process(pid)
        if proc.is_running() and "python" in proc.name().lower():
            return True, pid
        # Stale
        pid_file.unlink(missing_ok=True)
        return False, None
    except Exception:
        return False, None


def _handle_daemon_start(args: str, **ctx) -> str:
    running, pid = _is_running()
    if running:
        return f"Daemon already running (PID: {pid}). Use '/daemon stop' first."
    try:
        import subprocess
        import sys
        config_path = ctx.get("config_path", "")
        cmd = [sys.executable, "-m", "homie_app.cli", "daemon", "--headless"]
        if config_path:
            cmd.extend(["--config", config_path])

        # Launch detached — CREATE_NO_WINDOW on Windows
        kwargs = {"start_new_session": True}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        proc = subprocess.Popen(cmd, **kwargs)

        # Wait briefly to check if it started OK
        import time
        time.sleep(2)
        running, new_pid = _is_running()
        if running:
            return f"Daemon started (PID: {new_pid}). Tray icon should appear."
        return f"Daemon launched (PID: {proc.pid}). Starting up..."
    except Exception as e:
        return f"Could not start daemon: {e}"


def _handle_daemon_stop(args: str, **ctx) -> str:
    running, pid = _is_running()
    if not running:
        # Clean up stale PID file
        _pid_file().unlink(missing_ok=True)
        return "Daemon is not running."
    try:
        import os, signal
        os.kill(pid, signal.SIGTERM)
        # Wait for clean shutdown
        import time
        for _ in range(10):
            time.sleep(0.5)
            r, _ = _is_running()
            if not r:
                return f"Daemon stopped (PID: {pid})."
        return f"Sent stop signal to PID {pid}. It may take a moment to shut down."
    except Exception as e:
        return f"Could not stop daemon: {e}"


def _handle_daemon_status(args: str, **ctx) -> str:
    running, pid = _is_running()
    if not running:
        return "Daemon: not running"
    try:
        import psutil, time
        proc = psutil.Process(pid)
        uptime = time.time() - proc.create_time()
        hours = int(uptime // 3600)
        mins = int((uptime % 3600) // 60)
        mem = proc.memory_info().rss / (1024 * 1024)
        return (
            f"Daemon: running\n"
            f"  PID: {pid}\n"
            f"  Uptime: {hours}h {mins}m\n"
            f"  Memory: {mem:.0f} MB"
        )
    except Exception:
        return f"Daemon: running (PID: {pid})"


def _handle_daemon_install(args: str, **ctx) -> str:
    """Register Homie to start on Windows login via Task Scheduler."""
    try:
        from homie_app.service.scheduler_task import ServiceManager
        mgr = ServiceManager()
        if mgr.register():
            return "Homie registered to start on login (Windows Task Scheduler)."
        return "Failed to register. Try running as administrator."
    except Exception as e:
        return f"Install failed: {e}"


def _handle_daemon_uninstall(args: str, **ctx) -> str:
    """Remove Homie from Windows startup."""
    try:
        from homie_app.service.scheduler_task import ServiceManager
        mgr = ServiceManager()
        if mgr.unregister():
            return "Homie removed from startup."
        return "Failed to unregister."
    except Exception as e:
        return f"Uninstall failed: {e}"


def _handle_daemon_log(args: str, **ctx) -> str:
    """Show recent daemon log entries."""
    log_file = _log_file()
    if not log_file.exists():
        return "No daemon log file found. Start the daemon first."
    try:
        lines = log_file.read_text(encoding="utf-8").splitlines()
        n = 30
        if args.strip().isdigit():
            n = int(args.strip())
        tail = lines[-n:] if len(lines) > n else lines
        return "\n".join(tail) if tail else "(log is empty)"
    except Exception as e:
        return f"Could not read log: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="daemon",
        description="Manage background service (start, stop, status, install, log)",
        args_spec="start|stop|status|install|uninstall|log",
        subcommands={
            "start": SlashCommand(name="start", description="Start background daemon", handler_fn=_handle_daemon_start),
            "stop": SlashCommand(name="stop", description="Stop daemon", handler_fn=_handle_daemon_stop),
            "status": SlashCommand(name="status", description="Show daemon status", handler_fn=_handle_daemon_status),
            "install": SlashCommand(name="install", description="Register to start on login", handler_fn=_handle_daemon_install),
            "uninstall": SlashCommand(name="uninstall", description="Remove from startup", handler_fn=_handle_daemon_uninstall),
            "log": SlashCommand(name="log", description="Show recent daemon logs", args_spec="[lines]", handler_fn=_handle_daemon_log),
        },
    ))
