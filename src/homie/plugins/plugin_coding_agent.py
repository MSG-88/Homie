"""Homie Plugin: Coding Agent Framework

Adapts coding-agent architecture patterns (plan→act→observe→reflect loop,
tool registry, context management) into Homie's local plugin system.

Inspired by: https://magazine.sebastianraschka.com/p/components-of-a-coding-agent

This plugin provides a local coding-agent that can read files, propose edits,
and run shell commands on managed machines through Homie's orchestrator,
following a structured agent loop with safety guardrails.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from homie.config import HomieConfig, cfg_get
from homie.llm_ollama import LLMError, ollama_generate
from homie.safety import validate_plan

logger = logging.getLogger(__name__)

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

class AgentPhase(Enum):
    """Phases in the plan-act-observe-reflect agent loop."""
    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    REFLECT = "reflect"
    DONE = "done"
    ERROR = "error"


@dataclass
class ToolCall:
    """A single tool invocation requested by the agent."""
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Result returned after executing a tool."""
    name: str
    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class AgentMessage:
    """One entry in the agent's conversation/context window."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None


@dataclass
class AgentState:
    """Mutable state carried across loop iterations."""
    phase: AgentPhase = AgentPhase.PLAN
    messages: List[AgentMessage] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 10
    goal: str = ""
    final_answer: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class AgentTool(ABC):
    """Base class for tools available to the coding agent."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description shown to the LLM."""

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON-schema-style parameter description for the LLM."""
        return {}

    @abstractmethod
    def execute(self, arguments: Dict[str, Any], cfg: HomieConfig) -> ToolResult:
        """Run the tool locally. Must not make network calls unless opt-in via config."""


class ReadFileTool(AgentTool):
    """Read a local file's contents."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {"path": {"type": "string", "description": "Absolute file path to read."}}

    def execute(self, arguments: Dict[str, Any], cfg: HomieConfig) -> ToolResult:
        import pathlib
        path = pathlib.Path(arguments.get("path", ""))
        if not path.is_absolute():
            return ToolResult(name=self.name, success=False, output="", error="Path must be absolute.")
        if not path.exists():
            return ToolResult(name=self.name, success=False, output="", error=f"File not found: {path}")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:50_000]
            return ToolResult(name=self.name, success=True, output=content)
        except OSError as exc:
            return ToolResult(name=self.name, success=False, output="", error=str(exc))


class ListDirectoryTool(AgentTool):
    """List files in a directory."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and subdirectories at the given path."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {"path": {"type": "string", "description": "Absolute directory path."}}

    def execute(self, arguments: Dict[str, Any], cfg: HomieConfig) -> ToolResult:
        import pathlib
        path = pathlib.Path(arguments.get("path", ""))
        if not path.is_dir():
            return ToolResult(name=self.name, success=False, output="", error=f"Not a directory: {path}")
        try:
            entries = sorted(str(p.name) + ("/" if p.is_dir() else "") for p in path.iterdir())
            return ToolResult(name=self.name, success=True, output="\n".join(entries[:500]))
        except OSError as exc:
            return ToolResult(name=self.name, success=False, output="", error=str(exc))


class SearchFilesTool(AgentTool):
    """Search for a text pattern in files under a directory."""

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return "Search for a text pattern in files under a directory (case-insensitive, max 50 matches)."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "path": {"type": "string", "description": "Absolute directory path to search."},
            "pattern": {"type": "string", "description": "Text pattern to search for."},
        }

    def execute(self, arguments: Dict[str, Any], cfg: HomieConfig) -> ToolResult:
        import pathlib
        import re
        path = pathlib.Path(arguments.get("path", ""))
        pattern = arguments.get("pattern", "")
        if not path.is_dir():
            return ToolResult(name=self.name, success=False, output="", error=f"Not a directory: {path}")
        if not pattern:
            return ToolResult(name=self.name, success=False, output="", error="Empty search pattern.")
        try:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
        except re.error as exc:
            return ToolResult(name=self.name, success=False, output="", error=f"Bad pattern: {exc}")
        matches: List[str] = []
        for file_path in path.rglob("*"):
            if len(matches) >= 50:
                break
            if not file_path.is_file() or file_path.stat().st_size > 1_000_000:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{file_path}:{i}: {line.strip()[:200]}")
                        if len(matches) >= 50:
                            break
            except OSError:
                continue
        return ToolResult(name=self.name, success=True, output="\n".join(matches) or "No matches found.")


class ProposeEditTool(AgentTool):
    """Propose a file edit (does NOT write; returns a diff-like preview)."""

    @property
    def name(self) -> str:
        return "propose_edit"

    @property
    def description(self) -> str:
        return "Propose an edit to a file. Returns a preview; does not write unless approved."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "path": {"type": "string", "description": "Absolute file path."},
            "old_text": {"type": "string", "description": "Exact text to replace."},
            "new_text": {"type": "string", "description": "Replacement text."},
        }

    def execute(self, arguments: Dict[str, Any], cfg: HomieConfig) -> ToolResult:
        import pathlib
        path = pathlib.Path(arguments.get("path", ""))
        old_text = arguments.get("old_text", "")
        new_text = arguments.get("new_text", "")
        if not path.exists():
            return ToolResult(name=self.name, success=False, output="", error=f"File not found: {path}")
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            return ToolResult(name=self.name, success=False, output="", error=str(exc))
        if old_text not in content:
            return ToolResult(name=self.name, success=False, output="", error="old_text not found in file.")
        preview = (
            f"--- {path}\n"
            f"+++ {path} (proposed)\n"
            f"-{old_text}\n"
            f"+{new_text}\n"
        )
        return ToolResult(name=self.name, success=True, output=preview)


class FinishTool(AgentTool):
    """Signal that the agent has completed its task."""

    @property
    def name(self) -> str:
        return "finish"

    @property
    def description(self) -> str:
        return "Call when the task is complete. Provide a summary of what was done."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {"summary": {"type": "string", "description": "Summary of work done."}}

    def execute(self, arguments: Dict[str, Any], cfg: HomieConfig) -> ToolResult:
        summary = arguments.get("summary", "Task complete.")
        return ToolResult(name=self.name, success=True, output=summary)


class ToolRegistry:
    """Registry of tools available to the coding agent."""

    def __init__(self) -> None:
        self._tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[AgentTool]:
        return self._tools.get(name)

    def all_tools(self) -> Sequence[AgentTool]:
        return list(self._tools.values())

    def schema_for_llm(self) -> str:
        """Return a text description of all tools for inclusion in the LLM prompt."""
        lines: List[str] = []
        for tool in self._tools.values():
            params = json.dumps(tool.parameters_schema) if tool.parameters_schema else "{}"
            lines.append(f"- {tool.name}: {tool.description}  params={params}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context manager (keeps message history within token budget)
# ---------------------------------------------------------------------------

class ContextManager:
    """Manages the agent's conversation context within a character budget."""

    def __init__(self, max_chars: int = 30_000) -> None:
        self.max_chars = max_chars

    def trim(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        """Keep the system message and most recent messages within budget."""
        if not messages:
            return messages
        total = sum(len(m.content) for m in messages)
        if total <= self.max_chars:
            return messages
        # Always keep the first (system) message
        system = [messages[0]] if messages[0].role == "system" else []
        rest = messages[1:] if system else messages[:]
        # Drop oldest non-system messages until within budget
        system_chars = sum(len(m.content) for m in system)
        trimmed: List[AgentMessage] = []
        running = 0
        for msg in reversed(rest):
            running += len(msg.content)
            if system_chars + running > self.max_chars:
                break
            trimmed.insert(0, msg)
        return system + trimmed


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _build_system_prompt(tools: ToolRegistry, goal: str) -> str:
    return (
        "You are a coding agent running inside HOMIE, a local AI assistant.\n"
        "You operate in a plan-act-observe-reflect loop.\n\n"
        f"GOAL: {goal}\n\n"
        "AVAILABLE TOOLS:\n"
        f"{tools.schema_for_llm()}\n\n"
        "RESPONSE FORMAT: Return ONLY a JSON object, no markdown fences.\n"
        '{"thought": "your reasoning", "tool": "tool_name", "arguments": {...}}\n'
        'When done, use the "finish" tool.\n'
        "Rules:\n"
        "- Never delete or overwrite files without proposing the edit first.\n"
        "- Stay within the project directory.\n"
        "- If stuck after 3 attempts, use finish to report what you found.\n"
    )


def _parse_agent_response(raw: str) -> Optional[ToolCall]:
    """Extract a tool call from the LLM's JSON response."""
    text = raw.strip().strip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    tool_name = data.get("tool")
    arguments = data.get("arguments", {})
    if not tool_name:
        return None
    return ToolCall(name=tool_name, arguments=arguments)


def _messages_to_prompt(messages: List[AgentMessage]) -> str:
    """Flatten message history into a single prompt string for Ollama."""
    parts: List[str] = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"[SYSTEM]\n{msg.content}")
        elif msg.role == "user":
            parts.append(f"[USER]\n{msg.content}")
        elif msg.role == "assistant":
            parts.append(f"[ASSISTANT]\n{msg.content}")
        elif msg.role == "tool":
            parts.append(f"[TOOL RESULT]\n{msg.content}")
    parts.append("[ASSISTANT]")
    return "\n\n".join(parts)


class CodingAgent:
    """A local coding agent that follows the plan-act-observe-reflect loop.

    Integrates with Homie's LLM backend (Ollama) and respects safety
    validation before any actions reach the orchestrator.
    """

    def __init__(
        self,
        cfg: HomieConfig,
        tools: Optional[ToolRegistry] = None,
        max_iterations: int = 10,
        context_budget: int = 30_000,
    ) -> None:
        self.cfg = cfg
        self.tools = tools or self._default_tools()
        self.max_iterations = max_iterations
        self.context_mgr = ContextManager(max_chars=context_budget)

    @staticmethod
    def _default_tools() -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(ListDirectoryTool())
        registry.register(SearchFilesTool())
        registry.register(ProposeEditTool())
        registry.register(FinishTool())
        return registry

    def run(self, goal: str) -> AgentState:
        """Execute the agent loop for the given goal. Returns final state."""
        state = AgentState(
            goal=goal,
            max_iterations=self.max_iterations,
        )
        system_prompt = _build_system_prompt(self.tools, goal)
        state.messages.append(AgentMessage(role="system", content=system_prompt))
        state.messages.append(AgentMessage(role="user", content=goal))

        while state.iteration < state.max_iterations and state.phase != AgentPhase.DONE:
            state.iteration += 1
            state.phase = AgentPhase.PLAN
            logger.info("Agent iteration %d/%d", state.iteration, state.max_iterations)

            # --- PLAN + ACT: ask LLM for next tool call ---
            trimmed = self.context_mgr.trim(state.messages)
            prompt = _messages_to_prompt(trimmed)
            try:
                raw_response = ollama_generate(self.cfg, prompt)
            except LLMError as exc:
                logger.error("LLM error: %s", exc)
                state.phase = AgentPhase.ERROR
                state.final_answer = f"LLM error: {exc}"
                break

            state.messages.append(AgentMessage(role="assistant", content=raw_response))
            tool_call = _parse_agent_response(raw_response)

            if tool_call is None:
                state.messages.append(
                    AgentMessage(role="user", content="Could not parse your response. Reply with valid JSON.")
                )
                continue

            # --- ACT: execute tool ---
            state.phase = AgentPhase.ACT
            tool = self.tools.get(tool_call.name)
            if tool is None:
                result = ToolResult(
                    name=tool_call.name, success=False, output="",
                    error=f"Unknown tool '{tool_call.name}'.",
                )
            else:
                result = tool.execute(tool_call.arguments, self.cfg)

            # --- OBSERVE: feed result back ---
            state.phase = AgentPhase.OBSERVE
            result_text = result.output if result.success else f"ERROR: {result.error}"
            state.messages.append(AgentMessage(role="tool", content=result_text, tool_result=result))

            # --- REFLECT: check if done ---
            state.phase = AgentPhase.REFLECT
            if tool_call.name == "finish":
                state.phase = AgentPhase.DONE
                state.final_answer = result.output
                break

        if state.phase != AgentPhase.DONE:
            state.phase = AgentPhase.ERROR
            state.final_answer = state.final_answer or "Max iterations reached without completion."

        logger.info("Agent finished: phase=%s iterations=%d", state.phase.value, state.iteration)
        return state


# ---------------------------------------------------------------------------
# Plugin lifecycle
# ---------------------------------------------------------------------------

_agent_instance: Optional[CodingAgent] = None


def activate(cfg: HomieConfig) -> CodingAgent:
    """Activate the coding-agent plugin and return the agent instance."""
    global _agent_instance  # noqa: PLW0603
    max_iter = cfg_get(cfg, "plugins", "coding_agent", "max_iterations", default=10)
    budget = cfg_get(cfg, "plugins", "coding_agent", "context_budget", default=30_000)
    _agent_instance = CodingAgent(cfg, max_iterations=max_iter, context_budget=budget)
    logger.info("Coding agent plugin activated (max_iter=%d, budget=%d)", max_iter, budget)
    return _agent_instance


def deactivate() -> None:
    """Deactivate the coding-agent plugin and release resources."""
    global _agent_instance  # noqa: PLW0603
    _agent_instance = None
    logger.info("Coding agent plugin deactivated.")


def get_agent() -> Optional[CodingAgent]:
    """Return the current agent instance, or None if not activated."""
    return _agent_instance
