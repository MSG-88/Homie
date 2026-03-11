"""Built-in tools for Homie's agentic loop.

These give Homie real capabilities:
- Memory: teach facts, recall knowledge, search episodes
- System: time, date, system info, running processes
- Files: search files, read file contents
- Plugins: query any enabled plugin
"""
from __future__ import annotations

import datetime
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from homie_core.brain.tool_registry import Tool, ToolParam, ToolRegistry
from homie_core.memory.working import WorkingMemory


def register_builtin_tools(
    registry: ToolRegistry,
    working_memory: WorkingMemory,
    semantic_memory=None,
    episodic_memory=None,
    plugin_manager=None,
    storage_path: Optional[str] = None,
) -> None:
    """Register all built-in tools with the registry."""

    # ===================================================================
    # MEMORY TOOLS
    # ===================================================================

    def tool_remember(fact: str, confidence: float = 0.7, tags: str = "") -> str:
        """Store a fact in long-term semantic memory."""
        if not semantic_memory:
            return "Memory system not available."
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        fid = semantic_memory.learn(fact, confidence=float(confidence), tags=tag_list)
        return f"Remembered: \"{fact}\" (id={fid}, confidence={confidence})"

    registry.register(Tool(
        name="remember",
        description="Store a fact about the user in long-term memory. Use when the user shares preferences, habits, or important information.",
        params=[
            ToolParam(name="fact", description="The fact to remember", type="string"),
            ToolParam(name="confidence", description="How confident (0.0-1.0)", type="float", required=False, default=0.7),
            ToolParam(name="tags", description="Comma-separated tags", type="string", required=False, default=""),
        ],
        execute=tool_remember,
        category="memory",
    ))

    def tool_recall(query: str, limit: int = 5) -> str:
        """Search semantic memory for relevant facts."""
        if not semantic_memory:
            return "No facts stored yet."
        facts = semantic_memory.get_facts(min_confidence=0.3)
        if not facts:
            return "No facts stored yet."

        # Simple keyword matching + confidence ranking
        query_words = set(query.lower().split())
        scored = []
        for f in facts:
            fact_words = set(f["fact"].lower().split())
            overlap = len(query_words & fact_words)
            score = overlap * 0.5 + f["confidence"] * 0.5
            if overlap > 0 or f["confidence"] > 0.7:
                scored.append((f, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:int(limit)]

        if not top:
            return f"No facts matching '{query}' found."

        lines = [f"Found {len(top)} relevant facts:"]
        for f, score in top:
            lines.append(f"  - {f['fact']} (confidence: {f['confidence']:.0%})")
        return "\n".join(lines)

    registry.register(Tool(
        name="recall",
        description="Search your memory for facts about the user. Use before answering questions about user preferences or history.",
        params=[
            ToolParam(name="query", description="What to search for", type="string"),
            ToolParam(name="limit", description="Max results", type="int", required=False, default=5),
        ],
        execute=tool_recall,
        category="memory",
    ))

    def tool_recall_episodes(query: str, limit: int = 3) -> str:
        """Search episodic memory for past sessions/events."""
        if not episodic_memory:
            return "No episodes recorded yet."
        try:
            episodes = episodic_memory.recall(query, n=int(limit))
            if not episodes:
                return f"No episodes matching '{query}' found."
            lines = [f"Found {len(episodes)} related episodes:"]
            for ep in episodes:
                mood = f", mood: {ep['mood']}" if ep.get("mood") else ""
                outcome = f", outcome: {ep['outcome']}" if ep.get("outcome") else ""
                lines.append(f"  - {ep['summary']}{mood}{outcome}")
            return "\n".join(lines)
        except Exception as e:
            return f"Episode search failed: {e}"

    registry.register(Tool(
        name="recall_episodes",
        description="Search past sessions and events. Use when the user asks about what they did before.",
        params=[
            ToolParam(name="query", description="What to search for", type="string"),
            ToolParam(name="limit", description="Max results", type="int", required=False, default=3),
        ],
        execute=tool_recall_episodes,
        category="memory",
    ))

    def tool_forget(fact: str) -> str:
        """Remove a fact from memory."""
        if not semantic_memory:
            return "Memory system not available."
        facts = semantic_memory.get_facts()
        for f in facts:
            if fact.lower() in f["fact"].lower():
                semantic_memory.forget_fact(f["id"])
                return f"Forgot: \"{f['fact']}\""
        return f"No fact matching '{fact}' found."

    registry.register(Tool(
        name="forget",
        description="Remove a fact from memory when the user says to forget something or corrects old information.",
        params=[
            ToolParam(name="fact", description="The fact to forget (partial match)", type="string"),
        ],
        execute=tool_forget,
        category="memory",
    ))

    # ===================================================================
    # SYSTEM TOOLS
    # ===================================================================

    def tool_current_time() -> str:
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S (%A)")

    registry.register(Tool(
        name="current_time",
        description="Get the current date and time.",
        params=[],
        execute=tool_current_time,
        category="system",
    ))

    def tool_system_info() -> str:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.5)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            return (
                f"OS: {platform.system()} {platform.release()}\n"
                f"CPU: {cpu}% used\n"
                f"RAM: {ram.percent}% used ({ram.used // (1024**3)}GB / {ram.total // (1024**3)}GB)\n"
                f"Disk: {disk.percent}% used ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)"
            )
        except ImportError:
            return f"OS: {platform.system()} {platform.release()}, Python: {platform.python_version()}"

    registry.register(Tool(
        name="system_info",
        description="Get system information: CPU, RAM, disk usage.",
        params=[],
        execute=tool_system_info,
        category="system",
    ))

    def tool_running_apps() -> str:
        """List currently running applications from app tracker."""
        apps = working_memory.get("tracked_apps", {})
        if not apps:
            active = working_memory.get("active_window", "")
            process = working_memory.get("active_process", "")
            if active:
                return f"Currently active: {process} — {active}"
            return "No application data available."
        lines = ["Running applications:"]
        for app, info in sorted(apps.items()):
            if isinstance(info, dict):
                lines.append(f"  - {app}: {info.get('duration', 'active')}")
            else:
                lines.append(f"  - {app}")
        return "\n".join(lines)

    registry.register(Tool(
        name="running_apps",
        description="List currently running applications and which one is active.",
        params=[],
        execute=tool_running_apps,
        category="system",
    ))

    # ===================================================================
    # FILE TOOLS
    # ===================================================================

    def tool_search_files(pattern: str, directory: str = "") -> str:
        """Search for files matching a pattern."""
        search_dir = Path(directory) if directory else Path.home()
        if not search_dir.exists():
            return f"Directory not found: {search_dir}"
        try:
            matches = list(search_dir.rglob(pattern))[:20]
            if not matches:
                return f"No files matching '{pattern}' in {search_dir}"
            lines = [f"Found {len(matches)} files:"]
            for m in matches:
                size = m.stat().st_size if m.is_file() else 0
                size_str = f"{size // 1024}KB" if size > 1024 else f"{size}B"
                lines.append(f"  {m.relative_to(search_dir)} ({size_str})")
            return "\n".join(lines)
        except PermissionError:
            return f"Permission denied searching {search_dir}"
        except Exception as e:
            return f"Search error: {e}"

    registry.register(Tool(
        name="search_files",
        description="Search for files by name pattern (glob). Use when the user asks to find files.",
        params=[
            ToolParam(name="pattern", description="Glob pattern (e.g. '*.py', '*.txt')", type="string"),
            ToolParam(name="directory", description="Directory to search in", type="string", required=False, default=""),
        ],
        execute=tool_search_files,
        category="files",
    ))

    def tool_read_file(path: str, max_lines: int = 50) -> str:
        """Read contents of a text file."""
        file_path = Path(path)
        if not file_path.exists():
            return f"File not found: {path}"
        if not file_path.is_file():
            return f"Not a file: {path}"
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if len(lines) > int(max_lines):
                shown = "\n".join(lines[:int(max_lines)])
                return f"{shown}\n\n... ({len(lines) - int(max_lines)} more lines)"
            return text
        except Exception as e:
            return f"Error reading file: {e}"

    registry.register(Tool(
        name="read_file",
        description="Read the contents of a text file. Use when the user asks about file contents.",
        params=[
            ToolParam(name="path", description="Absolute path to the file", type="string"),
            ToolParam(name="max_lines", description="Max lines to read", type="int", required=False, default=50),
        ],
        execute=tool_read_file,
        category="files",
    ))

    # ===================================================================
    # CONTEXT TOOLS
    # ===================================================================

    def tool_user_context() -> str:
        """Get full user context snapshot."""
        snapshot = working_memory.snapshot()
        parts = []

        active = snapshot.get("active_window", "")
        if active:
            parts.append(f"Active window: {active}")
        process = snapshot.get("active_process", "")
        if process:
            parts.append(f"Active app: {process}")
        activity = snapshot.get("activity_type", "")
        if activity and activity != "unknown":
            parts.append(f"Activity: {activity}")
        flow = snapshot.get("flow_score", 0.5)
        parts.append(f"Focus level: {flow:.0%}")
        sentiment = snapshot.get("sentiment", "neutral")
        arousal = snapshot.get("arousal", "calm")
        if sentiment != "neutral" or arousal != "calm":
            parts.append(f"Mood: {sentiment}, {arousal}")
        deep = snapshot.get("is_deep_work", False)
        if deep:
            parts.append("In deep work mode")
        switches = snapshot.get("switch_count_30m", 0)
        if switches > 0:
            parts.append(f"Context switches (30m): {switches}")

        return "\n".join(parts) if parts else "No context data available."

    registry.register(Tool(
        name="user_context",
        description="Get the user's current context: what app they're using, their focus level, mood, and activity type.",
        params=[],
        execute=tool_user_context,
        category="context",
    ))

    # ===================================================================
    # PLUGIN BRIDGE
    # ===================================================================

    if plugin_manager:
        def tool_plugin(plugin: str, intent: str, params: str = "") -> str:
            """Query an enabled plugin."""
            param_dict = {}
            if params:
                try:
                    param_dict = dict(
                        pair.split("=", 1) for pair in params.split(",") if "=" in pair
                    )
                except Exception:
                    pass

            enabled = plugin_manager.list_enabled()
            if plugin not in enabled:
                available = ", ".join(enabled) if enabled else "none"
                return f"Plugin '{plugin}' not enabled. Available: {available}"

            result = plugin_manager.query_plugin(plugin, intent, param_dict)
            if result.success:
                return str(result.data) if result.data else "OK"
            return f"Plugin error: {result.error}"

        registry.register(Tool(
            name="plugin",
            description="Query an enabled plugin. Available plugins depend on what the user has enabled.",
            params=[
                ToolParam(name="plugin", description="Plugin name (e.g. 'system', 'clipboard')", type="string"),
                ToolParam(name="intent", description="What to ask the plugin (e.g. 'status', 'history')", type="string"),
                ToolParam(name="params", description="Key=value pairs separated by commas", type="string", required=False, default=""),
            ],
            execute=tool_plugin,
            category="plugins",
        ))
