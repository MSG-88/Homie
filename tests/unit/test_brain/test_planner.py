import pytest
from unittest.mock import MagicMock
from homie_core.brain.planner import ActionPlanner, PlannedAction


@pytest.fixture
def planner():
    engine = MagicMock()
    return ActionPlanner(model_engine=engine), engine


def test_plan_returns_action(planner):
    pl, engine = planner
    engine.generate.return_value = '{"action": "respond", "target": "", "params": {}, "reason": "greeting", "confidence": 0.9}'
    action = pl.plan("Hello!")
    assert isinstance(action, PlannedAction)
    assert action.action == "respond"


def test_plan_teach_action(planner):
    pl, engine = planner
    engine.generate.return_value = '{"action": "teach", "target": "", "params": {"fact": "I am allergic to peanuts"}, "reason": "user teaching", "confidence": 0.95}'
    action = pl.plan("Remember that I am allergic to peanuts")
    assert action.action == "teach"
    assert action.params["fact"] == "I am allergic to peanuts"


def test_plan_plugin_action(planner):
    pl, engine = planner
    engine.generate.return_value = '{"action": "run_plugin", "target": "email", "params": {"action": "summarize_inbox"}, "reason": "user wants email summary", "confidence": 0.8}'
    action = pl.plan("Summarize my emails")
    assert action.action == "run_plugin"
    assert action.target == "email"


def test_plan_handles_invalid_json(planner):
    pl, engine = planner
    engine.generate.return_value = "This is not JSON at all"
    action = pl.plan("Hello")
    assert action.action == "respond"  # fallback


def test_plan_handles_invalid_action(planner):
    pl, engine = planner
    engine.generate.return_value = '{"action": "fly_to_moon", "target": ""}'
    action = pl.plan("do something impossible")
    assert action.action == "respond"  # invalid action falls back
