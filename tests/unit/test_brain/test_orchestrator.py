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


def test_context_includes_facts(brain):
    br, _, _ = brain
    sm = MagicMock()
    sm.get_facts.return_value = [{"fact": "user likes Python", "confidence": 0.9}]
    sm.get_all_profiles.return_value = {}
    br._sm = sm
    context = br._build_context("test")
    assert "user likes Python" in context["known_facts"]


def test_context_includes_episodes(brain):
    br, _, _ = brain
    em = MagicMock()
    em.recall.return_value = [{"summary": "Debugged auth module", "mood": "focused"}]
    br._em = em
    context = br._build_context("debugging")
    assert "Debugged auth module" in context["relevant_episodes"]


def test_set_system_prompt(brain):
    br, _, _ = brain
    br.set_system_prompt("You are a coding assistant.")
    assert br._system_prompt == "You are a coding assistant."
