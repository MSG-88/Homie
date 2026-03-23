"""Unit tests for CodeAgent."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from homie_core.neural.agents.code_agent import CodeAgent
from homie_core.neural.communication.agent_bus import AgentBus, AgentMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def bus():
    b = AgentBus()
    yield b
    b.shutdown()


def _inference_fn(response):
    def fn(prompt: str, **kw) -> str:
        return response
    return fn


def _json_inference(obj):
    return _inference_fn(json.dumps(obj))


_SAMPLE_CODE = """\
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnalyzeCode:
    def test_analyze_returns_structured_result(self, bus):
        resp = {
            "language": "python",
            "summary": "Recursive Fibonacci implementation.",
            "structure": {"classes": [], "functions": ["fibonacci"], "imports": []},
            "issues": ["Exponential time complexity"],
            "suggestions": ["Use memoization"],
            "complexity": "medium",
        }
        agent = CodeAgent(bus, _json_inference(resp))
        result = agent.analyze_code("<inline>", code=_SAMPLE_CODE)
        assert result["language"] == "python"
        assert "fibonacci" in result["structure"]["functions"]
        assert len(result["issues"]) > 0

    def test_analyze_reads_from_file(self, bus):
        resp = {"language": "python", "summary": "Hello world."}
        agent = CodeAgent(bus, _json_inference(resp))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')")
            f.flush()
            result = agent.analyze_code(f.name)
        assert result["summary"] == "Hello world."

    def test_analyze_no_code_no_file(self, bus):
        agent = CodeAgent(bus, _json_inference({}))
        result = agent.analyze_code("")
        assert "error" in result

    def test_analyze_bad_file_path(self, bus):
        agent = CodeAgent(bus, _json_inference({}))
        result = agent.analyze_code("/nonexistent/path/foo.py")
        assert "error" in result

    def test_analyze_llm_failure(self, bus):
        agent = CodeAgent(bus, _inference_fn("not json"))
        result = agent.analyze_code("", code=_SAMPLE_CODE)
        assert result["summary"] == "Analysis failed."


class TestGenerateCode:
    def test_generates_code(self, bus):
        agent = CodeAgent(bus, _inference_fn("def add(a, b):\n    return a + b"))
        code = agent.generate_code("Write a function that adds two numbers")
        assert "def add" in code

    def test_strips_markdown_fences(self, bus):
        agent = CodeAgent(bus, _inference_fn("```python\ndef add(a, b):\n    return a + b\n```"))
        code = agent.generate_code("Write a function that adds two numbers")
        assert not code.startswith("```")
        assert "def add" in code

    def test_empty_specification(self, bus):
        agent = CodeAgent(bus, _inference_fn("should not be called"))
        assert agent.generate_code("") == ""

    def test_generate_handles_failure(self, bus):
        def failing(prompt, **kw):
            raise RuntimeError("LLM down")
        agent = CodeAgent(bus, failing)
        result = agent.generate_code("something")
        assert "failed" in result.lower()


class TestReviewCode:
    def test_review_returns_structured_result(self, bus):
        resp = {
            "quality_score": 0.7,
            "issues": [{"severity": "warning", "line": "4", "message": "No memoization", "suggestion": "Use lru_cache"}],
            "strengths": ["Clear naming"],
            "summary": "Functional but could be optimized.",
        }
        agent = CodeAgent(bus, _json_inference(resp))
        result = agent.review_code(_SAMPLE_CODE)
        assert result["quality_score"] == 0.7
        assert len(result["issues"]) == 1

    def test_review_empty_code(self, bus):
        agent = CodeAgent(bus, _json_inference({}))
        result = agent.review_code("")
        assert result["quality_score"] == 0.0
        assert result["summary"] == "No code provided."

    def test_review_clamps_quality_score(self, bus):
        resp = {"quality_score": 1.5, "issues": [], "strengths": [], "summary": "great"}
        agent = CodeAgent(bus, _json_inference(resp))
        result = agent.review_code(_SAMPLE_CODE)
        assert result["quality_score"] == 1.0

    def test_review_with_context(self, bus):
        resp = {"quality_score": 0.8, "issues": [], "strengths": [], "summary": "ok"}
        agent = CodeAgent(bus, _json_inference(resp))
        result = agent.review_code(_SAMPLE_CODE, context="Part of math utils module")
        assert result["quality_score"] == 0.8


class TestRefactor:
    def test_refactor_returns_modified_code(self, bus):
        refactored = "from functools import lru_cache\n\n@lru_cache\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)"
        agent = CodeAgent(bus, _inference_fn(refactored))
        result = agent.refactor(_SAMPLE_CODE, "Add memoization")
        assert "lru_cache" in result

    def test_refactor_empty_code(self, bus):
        agent = CodeAgent(bus, _inference_fn("should not be called"))
        assert agent.refactor("", "do something") == ""

    def test_refactor_no_instruction_returns_original(self, bus):
        agent = CodeAgent(bus, _inference_fn("should not be called"))
        assert agent.refactor(_SAMPLE_CODE, "") == _SAMPLE_CODE

    def test_refactor_handles_failure(self, bus):
        def failing(prompt, **kw):
            raise RuntimeError("LLM down")
        agent = CodeAgent(bus, failing)
        result = agent.refactor(_SAMPLE_CODE, "optimize")
        assert result == _SAMPLE_CODE  # Returns original on failure


class TestProcessMessage:
    def test_process_routes_analyze(self, bus):
        resp = {"language": "python", "summary": "test"}
        agent = CodeAgent(bus, _json_inference(resp))
        msg = AgentMessage(
            from_agent="meta", to_agent="code",
            message_type="goal",
            content={"action": "analyze_code", "code": _SAMPLE_CODE},
        )
        result = asyncio.get_event_loop().run_until_complete(agent.process(msg))
        assert result.content["language"] == "python"

    def test_process_routes_generate(self, bus):
        agent = CodeAgent(bus, _inference_fn("def foo(): pass"))
        msg = AgentMessage(
            from_agent="meta", to_agent="code",
            message_type="goal",
            content={"action": "generate_code", "specification": "A foo function"},
        )
        result = asyncio.get_event_loop().run_until_complete(agent.process(msg))
        assert "def foo" in result.content["code"]

    def test_process_routes_unknown(self, bus):
        agent = CodeAgent(bus, _json_inference({}))
        msg = AgentMessage(
            from_agent="meta", to_agent="code",
            message_type="goal",
            content={"action": "dance"},
        )
        result = asyncio.get_event_loop().run_until_complete(agent.process(msg))
        assert "error" in result.content
