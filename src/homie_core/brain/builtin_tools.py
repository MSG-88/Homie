"""Built-in tools for Homie's agentic loop.

These give Homie real capabilities:
- Memory: teach facts, recall knowledge, search episodes
- System: time, date, system info, running processes, terminal commands
- Files: search files, read file contents
- Git: status, log, diff, branches
- Clipboard: read/write clipboard
- Notes: save, list, read markdown notes
- Web: search via DuckDuckGo
- Plugins: query any enabled plugin
"""
from __future__ import annotations

import datetime
import json
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
    rag_pipeline=None,
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

    def tool_search_files(pattern: str, directory: str = "", backend=None) -> str:
        """Search for files matching a pattern."""
        if backend:
            try:
                matches = backend.glob(pattern, base=directory or None)
                if not matches:
                    return f"No files matching '{pattern}'"
                lines = [f"Found {len(matches)} files:"]
                for m in matches:
                    lines.append(f"  {m}")
                return "\n".join(lines)
            except Exception as e:
                return f"Search error: {e}"
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

    def tool_read_file(path: str, max_lines: int = 50, backend=None) -> str:
        """Read contents of a text file."""
        if backend:
            try:
                text = backend.read(path)
                lines = text.splitlines()
                if len(lines) > int(max_lines):
                    shown = "\n".join(lines[:int(max_lines)])
                    return f"{shown}\n\n... ({len(lines) - int(max_lines)} more lines)"
                return text
            except Exception as e:
                return f"Error reading file: {e}"
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
    # RAG / DOCUMENT SEARCH TOOLS
    # ===================================================================

    if rag_pipeline:
        def tool_search_docs(query: str, top_k: int = 5, file_filter: str = "") -> str:
            """Search indexed documents using hybrid BM25 + vector search."""
            try:
                contexts = rag_pipeline.retrieve(
                    query, top_k=int(top_k), max_chars=3000,
                    file_filter=file_filter if file_filter else None,
                )
                if not contexts:
                    return f"No documents matching '{query}' found. Try indexing a directory first."
                lines = [f"Found {len(contexts)} relevant sections:"]
                for ctx in contexts:
                    lines.append(ctx.to_attributed_text())
                    lines.append("")
                return "\n".join(lines)
            except Exception as e:
                return f"Document search error: {e}"

        registry.register(Tool(
            name="search_docs",
            description="Search indexed documents and code using semantic + keyword search. More powerful than search_files — finds content by meaning, not just filename.",
            params=[
                ToolParam(name="query", description="What to search for", type="string"),
                ToolParam(name="top_k", description="Max results", type="int", required=False, default=5),
                ToolParam(name="file_filter", description="Glob filter for file paths (e.g. '*.py')", type="string", required=False, default=""),
            ],
            execute=tool_search_docs,
            category="documents",
        ))

        def tool_index_directory(directory: str) -> str:
            """Index a directory for document search."""
            dir_path = Path(directory)
            if not dir_path.exists():
                return f"Directory not found: {directory}"
            if not dir_path.is_dir():
                return f"Not a directory: {directory}"
            try:
                count = rag_pipeline.index_directory(dir_path)
                stats = rag_pipeline.get_stats()
                return (
                    f"Indexed {count} new chunks from {directory}\n"
                    f"Total: {stats['total_chunks']} chunks across {stats['indexed_files']} files"
                )
            except Exception as e:
                return f"Indexing error: {e}"

        registry.register(Tool(
            name="index_directory",
            description="Index a directory so its files become searchable via search_docs. Use when the user wants you to learn about their codebase or documents.",
            params=[
                ToolParam(name="directory", description="Path to directory to index", type="string"),
            ],
            execute=tool_index_directory,
            category="documents",
        ))

        def tool_index_stats() -> str:
            """Get RAG indexing statistics."""
            stats = rag_pipeline.get_stats()
            return (
                f"Indexed files: {stats['indexed_files']}\n"
                f"Total chunks: {stats['total_chunks']}\n"
                f"Indexed directories: {', '.join(stats['indexed_dirs']) or 'none'}"
            )

        registry.register(Tool(
            name="index_stats",
            description="Show statistics about indexed documents — how many files and chunks are searchable.",
            params=[],
            execute=tool_index_stats,
            category="documents",
        ))

    # ===================================================================
    # GIT TOOLS
    # ===================================================================

    def tool_git_status() -> str:
        """Run git status and return parsed output."""
        try:
            result = subprocess.run(
                ["git", "status", "--short", "--branch"],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                return f"Git error: {result.stderr.strip()}"
            return output if output else "Working tree clean, nothing to commit."
        except FileNotFoundError:
            return "Git is not installed or not in PATH."
        except subprocess.TimeoutExpired:
            return "Git status timed out."
        except Exception as e:
            return f"Git error: {e}"

    registry.register(Tool(
        name="git_status",
        description="Run git status to see current changes, staged files, and branch info.",
        params=[],
        execute=tool_git_status,
        category="git",
    ))

    def tool_git_log(count: int = 5) -> str:
        """Show recent git commits."""
        try:
            result = subprocess.run(
                ["git", "log", f"--oneline", f"-{int(count)}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return f"Git error: {result.stderr.strip()}"
            output = result.stdout.strip()
            return output if output else "No commits found."
        except FileNotFoundError:
            return "Git is not installed or not in PATH."
        except subprocess.TimeoutExpired:
            return "Git log timed out."
        except Exception as e:
            return f"Git error: {e}"

    registry.register(Tool(
        name="git_log",
        description="Show recent git commits (one line each).",
        params=[
            ToolParam(name="count", description="Number of commits to show", type="int", required=False, default=5),
        ],
        execute=tool_git_log,
        category="git",
    ))

    def tool_git_diff(staged: bool = False) -> str:
        """Show current git diff."""
        try:
            cmd = ["git", "diff"]
            if staged:
                cmd.append("--cached")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return f"Git error: {result.stderr.strip()}"
            output = result.stdout.strip()
            if len(output) > 5000:
                output = output[:5000] + "\n... truncated"
            return output if output else "No changes."
        except FileNotFoundError:
            return "Git is not installed or not in PATH."
        except subprocess.TimeoutExpired:
            return "Git diff timed out."
        except Exception as e:
            return f"Git error: {e}"

    registry.register(Tool(
        name="git_diff",
        description="Show the current git diff (unstaged changes by default, or staged with staged=true).",
        params=[
            ToolParam(name="staged", description="Show staged changes instead", type="bool", required=False, default=False),
        ],
        execute=tool_git_diff,
        category="git",
    ))

    def tool_git_branch() -> str:
        """List git branches and show current branch."""
        try:
            result = subprocess.run(
                ["git", "branch", "--list"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return f"Git error: {result.stderr.strip()}"
            output = result.stdout.strip()
            return output if output else "No branches found."
        except FileNotFoundError:
            return "Git is not installed or not in PATH."
        except subprocess.TimeoutExpired:
            return "Git branch timed out."
        except Exception as e:
            return f"Git error: {e}"

    registry.register(Tool(
        name="git_branch",
        description="List all git branches and highlight the current one.",
        params=[],
        execute=tool_git_branch,
        category="git",
    ))

    # ===================================================================
    # TERMINAL TOOL
    # ===================================================================

    BLOCKED_COMMANDS = {
        "rm -rf /", "rm -rf ~", "format", "del /f /s /q", "mkfs",
        "dd if=", ":()", "chmod -r 777 /", "shutdown", "reboot",
    }

    def tool_run_command(command: str, backend=None) -> str:
        """Execute a shell command with safety checks."""
        cmd_lower = command.lower().strip()
        for blocked in BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return f"Blocked: command contains dangerous pattern '{blocked}'."

        if backend:
            try:
                return backend.execute(command)
            except Exception as e:
                return f"Command error: {e}"

        try:
            result = subprocess.run(
                command, shell=True,
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            output = output.strip()
            if len(output) > 5000:
                output = output[:5000] + "\n... truncated"
            if not output:
                return f"Command completed with exit code {result.returncode}."
            return output
        except subprocess.TimeoutExpired:
            return "Command timed out (10s limit)."
        except Exception as e:
            return f"Command error: {e}"

    registry.register(Tool(
        name="run_command",
        description="Execute a shell command and return its output. Dangerous commands are blocked for safety.",
        params=[
            ToolParam(name="command", description="Shell command to execute", type="string"),
        ],
        execute=tool_run_command,
        category="system",
    ))

    # ===================================================================
    # CLIPBOARD TOOLS
    # ===================================================================

    def tool_clipboard_read() -> str:
        """Read current clipboard content."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["powershell", "-command", "Get-Clipboard"],
                    capture_output=True, text=True, timeout=5,
                )
                return result.stdout.strip() if result.stdout.strip() else "Clipboard is empty."
            elif platform.system() == "Darwin":
                result = subprocess.run(
                    ["pbpaste"], capture_output=True, text=True, timeout=5,
                )
                return result.stdout.strip() if result.stdout.strip() else "Clipboard is empty."
            else:
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=5,
                )
                return result.stdout.strip() if result.stdout.strip() else "Clipboard is empty."
        except FileNotFoundError:
            return "Clipboard tool not available on this system."
        except subprocess.TimeoutExpired:
            return "Clipboard read timed out."
        except Exception as e:
            return f"Clipboard error: {e}"

    registry.register(Tool(
        name="clipboard_read",
        description="Read the current clipboard content.",
        params=[],
        execute=tool_clipboard_read,
        category="clipboard",
    ))

    def tool_clipboard_write(text: str) -> str:
        """Write text to clipboard."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["powershell", "-command", f"Set-Clipboard -Value '{text}'"],
                    capture_output=True, text=True, timeout=5,
                )
            elif platform.system() == "Darwin":
                result = subprocess.run(
                    ["pbcopy"], input=text, capture_output=True, text=True, timeout=5,
                )
            else:
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text, capture_output=True, text=True, timeout=5,
                )
            if result.returncode != 0:
                return f"Clipboard error: {result.stderr.strip()}"
            return f"Copied to clipboard ({len(text)} chars)."
        except FileNotFoundError:
            return "Clipboard tool not available on this system."
        except subprocess.TimeoutExpired:
            return "Clipboard write timed out."
        except Exception as e:
            return f"Clipboard error: {e}"

    registry.register(Tool(
        name="clipboard_write",
        description="Write text to the system clipboard.",
        params=[
            ToolParam(name="text", description="Text to copy to clipboard", type="string"),
        ],
        execute=tool_clipboard_write,
        category="clipboard",
    ))

    # ===================================================================
    # NOTES TOOLS
    # ===================================================================

    notes_dir = Path(storage_path or "~/.homie").expanduser() / "notes"

    def tool_save_note(title: str, content: str) -> str:
        """Save a note as a markdown file."""
        try:
            notes_dir.mkdir(parents=True, exist_ok=True)
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
            safe_title = safe_title.strip().replace(" ", "_")
            if not safe_title:
                return "Invalid note title."
            filepath = notes_dir / f"{safe_title}.md"
            filepath.write_text(f"# {title}\n\n{content}", encoding="utf-8")
            return f"Note saved: {filepath.name}"
        except Exception as e:
            return f"Error saving note: {e}"

    registry.register(Tool(
        name="save_note",
        description="Save a note with a title. Notes are stored as markdown files.",
        params=[
            ToolParam(name="title", description="Title of the note", type="string"),
            ToolParam(name="content", description="Note content (supports markdown)", type="string"),
        ],
        execute=tool_save_note,
        category="notes",
    ))

    def tool_list_notes() -> str:
        """List all saved notes."""
        try:
            if not notes_dir.exists():
                return "No notes saved yet."
            notes = sorted(notes_dir.glob("*.md"))
            if not notes:
                return "No notes saved yet."
            lines = [f"Found {len(notes)} notes:"]
            for note in notes:
                size = note.stat().st_size
                lines.append(f"  - {note.stem} ({size}B)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing notes: {e}"

    registry.register(Tool(
        name="list_notes",
        description="List all saved notes.",
        params=[],
        execute=tool_list_notes,
        category="notes",
    ))

    def tool_read_note(title: str) -> str:
        """Read a specific note by title."""
        try:
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
            safe_title = safe_title.strip().replace(" ", "_")
            filepath = notes_dir / f"{safe_title}.md"
            if not filepath.exists():
                # Try case-insensitive search
                for note in notes_dir.glob("*.md"):
                    if note.stem.lower() == safe_title.lower():
                        filepath = note
                        break
                else:
                    return f"Note '{title}' not found."
            return filepath.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading note: {e}"

    registry.register(Tool(
        name="read_note",
        description="Read a specific note by its title.",
        params=[
            ToolParam(name="title", description="Title of the note to read", type="string"),
        ],
        execute=tool_read_note,
        category="notes",
    ))

    # ===================================================================
    # WEB SEARCH TOOL
    # ===================================================================

    def tool_web_search(query: str) -> str:
        """Search the web using DuckDuckGo instant answers API."""
        try:
            import requests
        except ImportError:
            return "Web search unavailable: requests library not installed."

        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            parts = []
            # Abstract (main answer)
            if data.get("Abstract"):
                parts.append(f"**{data.get('Heading', 'Answer')}**: {data['Abstract']}")
                if data.get("AbstractURL"):
                    parts.append(f"Source: {data['AbstractURL']}")

            # Answer box
            if data.get("Answer"):
                parts.append(f"Answer: {data['Answer']}")

            # Related topics
            related = data.get("RelatedTopics", [])
            if related:
                parts.append("\nRelated:")
                for topic in related[:5]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        text = topic["Text"][:200]
                        parts.append(f"  - {text}")

            if not parts:
                return f"No instant results for '{query}'. Try a more specific query."

            return "\n".join(parts)
        except Exception as e:
            return f"Web search error: {e}"

    registry.register(Tool(
        name="web_search",
        description="Search the web using DuckDuckGo. Returns instant answers and related topics.",
        params=[
            ToolParam(name="query", description="Search query", type="string"),
        ],
        execute=tool_web_search,
        category="web",
    ))

    # ===================================================================
    # PLUGIN BRIDGE
    # ===================================================================

    # ===================================================================
    # INTELLIGENCE TOOLS (weather, news)
    # ===================================================================
    # These are registered separately via register_intelligence_tools()
    # so they can be called with config/vault from the console bootstrap.

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


def _get_vault_key(vault, provider: str) -> str:
    """Retrieve an API key from the vault for a given provider."""
    if not vault:
        return ""
    try:
        cred = vault.get_credential(provider, "default")
        return cred.access_token if cred else ""
    except Exception:
        return ""


def register_intelligence_tools(
    registry: ToolRegistry,
    config=None,
    vault=None,
) -> None:
    """Register weather and news tools for conversational Brain access."""

    def get_weather(location: str = "", **kwargs) -> str:
        city = location or ""
        if not city and config:
            loc = getattr(config, "location", None)
            if loc:
                city = loc.city
        if not city:
            return "No location provided or configured. Use /location set <city> first."
        api_key = _get_vault_key(vault, "weather")
        from homie_core.intelligence.weather import WeatherService
        svc = WeatherService(api_key=api_key)
        data = svc.get_current(city)
        return svc.format_current(data)

    registry.register(Tool(
        name="get_weather",
        description="Get current weather for a location. Uses configured location if none specified.",
        params=[
            ToolParam(name="location", description="City name (optional, defaults to configured location)", type="string", required=False, default=""),
        ],
        execute=get_weather,
        category="intelligence",
    ))

    def get_news(query: str = "", **kwargs) -> str:
        api_key = _get_vault_key(vault, "news")
        country = "us"
        if config:
            loc = getattr(config, "location", None)
            if loc and loc.country:
                country = loc.country.lower()
        from homie_core.intelligence.news import NewsService
        svc = NewsService(api_key=api_key)
        data = svc.get_headlines(country=country, query=query)
        return svc.format_headlines(data)

    registry.register(Tool(
        name="get_news",
        description="Get top news headlines, optionally filtered by topic. Uses configured country.",
        params=[
            ToolParam(name="query", description="Topic to search for (optional)", type="string", required=False, default=""),
        ],
        execute=get_news,
        category="intelligence",
    ))
