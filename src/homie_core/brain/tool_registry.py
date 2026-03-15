"""Tool Registry — defines callable tools for the agentic loop.

Each tool has a name, description, parameter schema, and execute function.
The registry generates tool descriptions for the model prompt and
dispatches tool calls parsed from model output.
"""
from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolParam:
    """A single parameter for a tool."""
    name: str
    description: str
    type: str = "string"
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    """A callable tool the model can invoke."""
    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)
    execute: Callable[..., str] = field(default=lambda **kw: "")
    category: str = "general"


@dataclass
class ToolCall:
    """A parsed tool call from model output."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None


# Pattern to detect tool calls in model output
# Format: <tool>tool_name(param1="value1", param2="value2")</tool>
_TOOL_CALL_PATTERN = re.compile(
    r"<tool>\s*(\w+)\s*\(([^)]*)\)\s*</tool>",
    re.DOTALL,
)

# Alternative JSON format: {"tool": "name", "args": {...}}
_JSON_TOOL_PATTERN = re.compile(
    r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*(\{[^}]*\})\s*\}',
    re.DOTALL,
)

# Action/Observation format (used by many LLMs):
# Action: tool_name(param="value")
_ACTION_TOOL_PATTERN = re.compile(
    r'Action:\s*(\w+)\s*\(([^)]*)\)',
    re.IGNORECASE,
)

# Markdown code block tool format:
# ```tool\n{"name": "...", "arguments": {...}}\n```
_MARKDOWN_TOOL_PATTERN = re.compile(
    r'```(?:tool|tool_code|json)?\s*\n\s*\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*?\})\s*\}\s*\n```',
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from model output.

    Supports multiple formats for maximum LLM compatibility:
    1. <tool>name(key="value", ...)</tool>
    2. {"tool": "name", "args": {"key": "value"}}
    3. Action: name(key="value")
    4. ```tool\n{"name": "...", "arguments": {...}}\n```
    """
    calls = []

    # Try XML-style format first (highest priority — our native format)
    for match in _TOOL_CALL_PATTERN.finditer(text):
        name = match.group(1)
        args_str = match.group(2).strip()
        args = _parse_kwargs(args_str)
        calls.append(ToolCall(name=name, arguments=args))

    # Try JSON format
    if not calls:
        for match in _JSON_TOOL_PATTERN.finditer(text):
            name = match.group(1)
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(name=name, arguments=args))

    # Try markdown code block format
    if not calls:
        for match in _MARKDOWN_TOOL_PATTERN.finditer(text):
            name = match.group(1)
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(name=name, arguments=args))

    # Try Action: format (fallback)
    if not calls:
        for match in _ACTION_TOOL_PATTERN.finditer(text):
            name = match.group(1)
            args_str = match.group(2).strip()
            args = _parse_kwargs(args_str)
            calls.append(ToolCall(name=name, arguments=args))

    return calls


def _parse_kwargs(args_str: str) -> dict[str, Any]:
    """Parse key=value pairs from a tool call argument string."""
    if not args_str:
        return {}

    args = {}
    # Match key="value" or key=value patterns
    pattern = re.compile(r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\S+))')
    for m in pattern.finditer(args_str):
        key = m.group(1)
        value = m.group(2) if m.group(2) is not None else (
            m.group(3) if m.group(3) is not None else m.group(4)
        )
        # Try to parse as number/bool
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:
                value = int(value)
            except (ValueError, TypeError):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
        args[key] = value
    return args


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,      # deletion
                curr[j] + 1,           # insertion
                prev[j] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


class ToolRegistry:
    """Registry of callable tools available to the model.

    Includes fuzzy name matching — when the model hallucinates a
    close-but-wrong tool name, the registry auto-corrects to the
    nearest match if edit distance ≤ 2.
    """

    # Maximum edit distance for fuzzy matching
    _FUZZY_THRESHOLD = 2

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._context: dict[str, Any] = {}

    def set_context(self, context: dict[str, Any]) -> None:
        """Set execution context (e.g. backend) injected into tools that accept it."""
        self._context = context

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def _fuzzy_match(self, name: str) -> Optional[Tool]:
        """Find the closest tool name within edit distance threshold."""
        best_tool: Optional[Tool] = None
        best_dist = self._FUZZY_THRESHOLD + 1
        for tool_name, tool in self._tools.items():
            dist = _levenshtein(name.lower(), tool_name.lower())
            if dist < best_dist:
                best_dist = dist
                best_tool = tool
        return best_tool if best_dist <= self._FUZZY_THRESHOLD else None

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a parsed tool call with fuzzy name matching."""
        tool = self._tools.get(call.name)
        if not tool:
            # Try fuzzy match before giving up
            tool = self._fuzzy_match(call.name)
            if tool:
                resolved_name = tool.name
            else:
                return ToolResult(
                    tool_name=call.name, success=False, output="",
                    error=f"Unknown tool: {call.name}",
                )
        else:
            resolved_name = call.name

        try:
            kwargs = dict(call.arguments)
            # Inject backend if the tool function accepts it
            try:
                sig = inspect.signature(tool.execute)
                if "backend" in sig.parameters:
                    kwargs.setdefault("backend", self._context.get("backend"))
            except (ValueError, TypeError):
                pass
            output = tool.execute(**kwargs)
            return ToolResult(tool_name=resolved_name, success=True, output=str(output))
        except TypeError as e:
            return ToolResult(
                tool_name=resolved_name, success=False, output="",
                error=f"Invalid arguments: {e}",
            )
        except Exception as e:
            return ToolResult(
                tool_name=resolved_name, success=False, output="",
                error=f"Tool error: {e}",
            )

    def generate_tool_prompt(self) -> str:
        """Generate a prompt section describing available tools.

        This goes into the system prompt so the model knows what tools exist
        and how to call them.
        """
        if not self._tools:
            return ""

        lines = ["[TOOLS]", "You can use these tools by writing: <tool>name(param=\"value\")</tool>", ""]

        for tool in self._tools.values():
            params_desc = ""
            if tool.params:
                param_parts = []
                for p in tool.params:
                    req = " (required)" if p.required else f" (optional, default={p.default})"
                    param_parts.append(f"    {p.name}: {p.type} — {p.description}{req}")
                params_desc = "\n" + "\n".join(param_parts)

            lines.append(f"- {tool.name}: {tool.description}{params_desc}")

        lines.append("")
        lines.append("Only call tools when needed. You can call multiple tools in one response.")
        lines.append("After a tool result, continue your response naturally.")
        return "\n".join(lines)
