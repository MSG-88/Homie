import pytest
from unittest.mock import MagicMock
from homie_core.brain.orchestrator import BrainOrchestrator
from homie_core.memory.working import WorkingMemory


@pytest.fixture
def brain():
    engine = MagicMock()
    engine.generate.return_value = "Hello! I'm Homie, your AI assistant."
    wm = WorkingMemory()
    return BrainOrchestrator(model_engine=engine, working_memory=wm), engine, wm


def test_process_returns_response(brain):
    br, engine, wm = brain
    response = br.process("Hello")
    assert response == "Hello! I'm Homie, your AI assistant."
    engine.generate.assert_called_once()


def test_process_adds_to_conversation(brain):
    br, _, wm = brain
    br.process("Hello")
    msgs = wm.get_conversation()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_prompt_includes_facts(brain):
    br, _, _ = brain
    sm = MagicMock()
    sm.get_facts.return_value = [{"fact": "user likes Python", "confidence": 0.9}]
    br._sm = sm
    prompt = br._build_optimized_prompt("test")
    assert "user likes Python" in prompt


def test_prompt_includes_episodes(brain):
    br, _, _ = brain
    em = MagicMock()
    em.recall.return_value = [{"summary": "Debugged auth module", "mood": "focused"}]
    br._em = em
    prompt = br._build_optimized_prompt("debugging")
    assert "Debugged auth module" in prompt


def test_process_stream_yields_tokens(brain):
    br, engine, wm = brain
    engine.stream.return_value = iter(["Hello", " world", "!"])
    tokens = list(br.process_stream("Hi"))
    assert tokens == ["Hello", " world", "!"]
    msgs = wm.get_conversation()
    assert msgs[-1]["content"] == "Hello world!"


def test_optimized_prompt_respects_budget(brain):
    br, _, _ = brain
    br.set_system_prompt("A" * 2500)
    prompt = br._build_optimized_prompt("test query")
    # Should still include system prompt and user query even with tight budget
    assert "A" * 2500 in prompt
    assert "test query" in prompt


def test_set_system_prompt(brain):
    br, _, _ = brain
    br.set_system_prompt("You are a coding assistant.")
    assert br._system_prompt == "You are a coding assistant."
