import pytest
from homie_core.brain.suggestion_engine import SuggestionEngine, Suggestion


def test_generate_break_suggestion():
    engine = SuggestionEngine(threshold=0.5)
    suggestions = engine.generate_suggestions({"is_deep_work": True})
    assert len(suggestions) >= 1
    assert any(s.category == "health" for s in suggestions)


def test_no_suggestion_when_not_deep_work():
    engine = SuggestionEngine(threshold=0.5)
    suggestions = engine.generate_suggestions({"is_deep_work": False})
    health_suggestions = [s for s in suggestions if s.category == "health"]
    assert len(health_suggestions) == 0


def test_meeting_reminder():
    engine = SuggestionEngine(threshold=0.5)
    suggestions = engine.generate_suggestions({"upcoming_meeting": 10})
    assert any(s.category == "calendar" for s in suggestions)


def test_record_response_updates_rates():
    engine = SuggestionEngine()
    engine.record_response("health", accepted=True)
    engine.record_response("health", accepted=True)
    engine.record_response("health", accepted=False)
    rates = engine.get_acceptance_rates()
    assert abs(rates["health"] - 0.667) < 0.01


def test_suppress_suggestions():
    engine = SuggestionEngine(threshold=0.5)
    engine.suppress(minutes=30)
    suggestions = engine.generate_suggestions({"is_deep_work": True})
    health_suggestions = [s for s in suggestions if s.category == "health"]
    assert len(health_suggestions) == 0


def test_filter_low_acceptance_category():
    engine = SuggestionEngine(threshold=0.5)
    for _ in range(10):
        engine.record_response("health", accepted=False)
    suggestions = engine.generate_suggestions({"is_deep_work": True})
    health_suggestions = [s for s in suggestions if s.category == "health"]
    assert len(health_suggestions) == 0
