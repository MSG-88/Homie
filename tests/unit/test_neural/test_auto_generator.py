"""Tests for the AutoGenerator module."""

import pytest

from homie_core.neural.proactive.auto_generator import AutoGenerator
from homie_core.neural.proactive.trigger_engine import ProactiveTask


def _make_inference(response: str):
    def inference_fn(prompt: str) -> str:
        return response
    return inference_fn


def _task(**kwargs):
    return ProactiveTask(
        id=kwargs.get("id", "test_task"),
        trigger_type=kwargs.get("trigger_type", "schedule"),
        trigger_config=kwargs.get("trigger_config", {}),
        action=kwargs.get("action", "generate_report"),
        domain=kwargs.get("domain", "finance"),
        priority=kwargs.get("priority", 3),
    )


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_returns_report_string(self):
        gen = AutoGenerator(_make_inference("# Monthly Report\nAll is well."))
        result = gen.generate_report(_task(), {"revenue": 10000})
        assert "Monthly Report" in result

    def test_prompt_includes_domain(self):
        calls = []
        def mock(prompt):
            calls.append(prompt)
            return "report"
        gen = AutoGenerator(mock)
        gen.generate_report(_task(domain="accounting"), {})
        assert "accounting" in calls[0]

    def test_prompt_includes_action(self):
        calls = []
        def mock(prompt):
            calls.append(prompt)
            return "report"
        gen = AutoGenerator(mock)
        gen.generate_report(_task(action="generate_financial_summary"), {})
        assert "generate_financial_summary" in calls[0]

    def test_data_included_in_prompt(self):
        calls = []
        def mock(prompt):
            calls.append(prompt)
            return "report"
        gen = AutoGenerator(mock)
        gen.generate_report(_task(), {"total_expenses": 5000})
        assert "total_expenses" in calls[0]


# ---------------------------------------------------------------------------
# generate_alert
# ---------------------------------------------------------------------------

class TestGenerateAlert:
    def test_returns_alert_string(self):
        gen = AutoGenerator(_make_inference("CRITICAL: Unusual transaction detected."))
        result = gen.generate_alert(_task(priority=1), {"amount": 50000})
        assert "CRITICAL" in result

    def test_prompt_includes_priority(self):
        calls = []
        def mock(prompt):
            calls.append(prompt)
            return "alert"
        gen = AutoGenerator(mock)
        gen.generate_alert(_task(priority=1), {})
        assert "1" in calls[0]

    def test_prompt_includes_trigger_type(self):
        calls = []
        def mock(prompt):
            calls.append(prompt)
            return "alert"
        gen = AutoGenerator(mock)
        gen.generate_alert(_task(trigger_type="threshold"), {})
        assert "threshold" in calls[0]


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------

class TestGenerateSummary:
    def test_returns_summary_string(self):
        gen = AutoGenerator(_make_inference("Weekly summary: productivity was high."))
        result = gen.generate_summary("weekly_work", {"tasks_completed": 15})
        assert "Weekly summary" in result

    def test_prompt_includes_topic(self):
        calls = []
        def mock(prompt):
            calls.append(prompt)
            return "summary"
        gen = AutoGenerator(mock)
        gen.generate_summary("quarterly_review", {"data": "test"})
        assert "quarterly_review" in calls[0]

    def test_returns_string_type(self):
        gen = AutoGenerator(_make_inference("A summary."))
        result = gen.generate_summary("test", {})
        assert isinstance(result, str)
