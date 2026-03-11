"""Tests for the tool registry, agentic loop, and built-in tools."""
import pytest
from unittest.mock import MagicMock

from homie_core.brain.tool_registry import (
    Tool, ToolParam, ToolCall, ToolResult, ToolRegistry,
    parse_tool_calls, _parse_kwargs,
)
from homie_core.brain.agentic_loop import AgenticLoop, _strip_tool_markers
from homie_core.brain.builtin_tools import register_builtin_tools
from homie_core.memory.working import WorkingMemory


# -----------------------------------------------------------------------
# Tool call parsing
# -----------------------------------------------------------------------

class TestParseToolCalls:
    def test_xml_style_simple(self):
        text = 'Let me check. <tool>current_time()</tool>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "current_time"
        assert calls[0].arguments == {}

    def test_xml_style_with_args(self):
        text = '<tool>remember(fact="user likes Python", confidence=0.9)</tool>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "remember"
        assert calls[0].arguments["fact"] == "user likes Python"
        assert calls[0].arguments["confidence"] == 0.9

    def test_json_style(self):
        text = '{"tool": "recall", "args": {"query": "preferences"}}'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "recall"
        assert calls[0].arguments["query"] == "preferences"

    def test_multiple_calls(self):
        text = '<tool>current_time()</tool> Let me also check <tool>system_info()</tool>'
        calls = parse_tool_calls(text)
        assert len(calls) == 2

    def test_no_tool_calls(self):
        text = "Just a normal response without any tool calls."
        calls = parse_tool_calls(text)
        assert len(calls) == 0

    def test_boolean_args(self):
        args = _parse_kwargs('flag=true other=false')
        assert args["flag"] is True
        assert args["other"] is False

    def test_integer_args(self):
        args = _parse_kwargs('limit=5 offset=10')
        assert args["limit"] == 5
        assert args["offset"] == 10


# -----------------------------------------------------------------------
# Tool Registry
# -----------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = Tool(name="test", description="A test tool", execute=lambda: "ok")
        registry.register(tool)
        assert registry.get("test") is tool

    def test_execute_tool(self):
        registry = ToolRegistry()
        tool = Tool(
            name="greet", description="Say hello",
            params=[ToolParam(name="name", description="Name to greet")],
            execute=lambda name="World": f"Hello, {name}!",
        )
        registry.register(tool)

        result = registry.execute(ToolCall(name="greet", arguments={"name": "Alice"}))
        assert result.success is True
        assert result.output == "Hello, Alice!"

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute(ToolCall(name="nonexistent"))
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_execute_with_invalid_args(self):
        registry = ToolRegistry()
        tool = Tool(
            name="strict", description="Strict tool",
            execute=lambda required_arg: required_arg,
        )
        registry.register(tool)
        result = registry.execute(ToolCall(name="strict", arguments={}))
        assert result.success is False

    def test_generate_tool_prompt(self):
        registry = ToolRegistry()
        registry.register(Tool(
            name="search", description="Search for files",
            params=[ToolParam(name="query", description="Search query")],
        ))
        prompt = registry.generate_tool_prompt()
        assert "[TOOLS]" in prompt
        assert "search" in prompt
        assert "Search for files" in prompt

    def test_empty_registry_prompt(self):
        registry = ToolRegistry()
        assert registry.generate_tool_prompt() == ""

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(Tool(name="a", description="Tool A"))
        registry.register(Tool(name="b", description="Tool B"))
        assert len(registry.list_tools()) == 2


# -----------------------------------------------------------------------
# Strip tool markers
# -----------------------------------------------------------------------

class TestStripToolMarkers:
    def test_strips_xml_markers(self):
        text = "Here's the time: <tool>current_time()</tool> Let me know!"
        clean = _strip_tool_markers(text)
        assert "<tool>" not in clean
        assert "current_time" not in clean

    def test_preserves_normal_text(self):
        text = "Just a normal response."
        assert _strip_tool_markers(text) == text


# -----------------------------------------------------------------------
# Agentic Loop
# -----------------------------------------------------------------------

class TestAgenticLoop:
    def test_no_tool_calls_returns_directly(self):
        engine = MagicMock()
        engine.generate.return_value = "Hello! How can I help?"
        registry = ToolRegistry()

        loop = AgenticLoop(engine, registry)
        result = loop.process("test prompt")
        assert result == "Hello! How can I help?"
        engine.generate.assert_called_once()

    def test_tool_call_triggers_re_generation(self):
        engine = MagicMock()
        engine.generate.side_effect = [
            '<tool>current_time()</tool>',
            "The current time is 2:30 PM.",
        ]

        registry = ToolRegistry()
        registry.register(Tool(
            name="current_time", description="Get time",
            execute=lambda: "2024-01-01 14:30:00",
        ))

        loop = AgenticLoop(engine, registry)
        result = loop.process("what time is it?")
        assert "2:30 PM" in result
        assert engine.generate.call_count == 2

    def test_max_iterations_prevents_infinite_loop(self):
        engine = MagicMock()
        # Always returns a tool call
        engine.generate.return_value = '<tool>current_time()</tool>'

        registry = ToolRegistry()
        registry.register(Tool(
            name="current_time", description="Get time",
            execute=lambda: "now",
        ))

        loop = AgenticLoop(engine, registry, max_iterations=3)
        result = loop.process("test")
        assert engine.generate.call_count == 3


# -----------------------------------------------------------------------
# Built-in Tools
# -----------------------------------------------------------------------

class TestBuiltinTools:
    def setup_method(self):
        self.registry = ToolRegistry()
        self.wm = WorkingMemory()
        self.sm = MagicMock()
        self.em = MagicMock()
        register_builtin_tools(
            registry=self.registry,
            working_memory=self.wm,
            semantic_memory=self.sm,
            episodic_memory=self.em,
        )

    def test_current_time_registered(self):
        tool = self.registry.get("current_time")
        assert tool is not None
        result = self.registry.execute(ToolCall(name="current_time"))
        assert result.success is True
        assert len(result.output) > 0

    def test_remember_tool(self):
        self.sm.learn.return_value = 1
        result = self.registry.execute(ToolCall(
            name="remember",
            arguments={"fact": "user likes dark mode", "confidence": 0.8},
        ))
        assert result.success is True
        assert "Remembered" in result.output
        self.sm.learn.assert_called_once()

    def test_recall_tool_no_facts(self):
        self.sm.get_facts.return_value = []
        result = self.registry.execute(ToolCall(
            name="recall", arguments={"query": "preferences"},
        ))
        assert result.success is True
        assert "No facts" in result.output

    def test_recall_tool_with_facts(self):
        self.sm.get_facts.return_value = [
            {"fact": "user likes dark mode", "confidence": 0.9},
            {"fact": "user prefers Python", "confidence": 0.8},
        ]
        result = self.registry.execute(ToolCall(
            name="recall", arguments={"query": "dark mode"},
        ))
        assert result.success is True
        assert "dark mode" in result.output

    def test_forget_tool(self):
        self.sm.get_facts.return_value = [
            {"id": 1, "fact": "user likes vim"},
        ]
        result = self.registry.execute(ToolCall(
            name="forget", arguments={"fact": "vim"},
        ))
        assert result.success is True
        assert "Forgot" in result.output

    def test_recall_episodes_tool(self):
        self.em.recall.return_value = [
            {"summary": "Debugged auth module", "mood": "focused", "outcome": "success"},
        ]
        result = self.registry.execute(ToolCall(
            name="recall_episodes", arguments={"query": "debugging"},
        ))
        assert result.success is True
        assert "auth" in result.output

    def test_system_info_tool(self):
        result = self.registry.execute(ToolCall(name="system_info"))
        assert result.success is True
        assert "OS:" in result.output or "Python:" in result.output

    def test_user_context_tool(self):
        self.wm.update("active_window", "VS Code")
        self.wm.update("activity_type", "coding")
        self.wm.update("flow_score", 0.85)
        result = self.registry.execute(ToolCall(name="user_context"))
        assert result.success is True
        assert "VS Code" in result.output

    def test_read_file_not_found(self):
        result = self.registry.execute(ToolCall(
            name="read_file",
            arguments={"path": "/nonexistent/file.txt"},
        ))
        assert result.success is True  # tool ran, returned error message
        assert "not found" in result.output.lower()

    def test_all_expected_tools_registered(self):
        expected = ["remember", "recall", "recall_episodes", "forget",
                    "current_time", "system_info", "running_apps",
                    "search_files", "read_file", "user_context"]
        for name in expected:
            assert self.registry.get(name) is not None, f"Tool '{name}' not registered"
