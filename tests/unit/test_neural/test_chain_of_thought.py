"""Tests for the ChainOfThought module."""

import json
import pytest

from homie_core.neural.reasoning.chain_of_thought import ChainOfThought
from homie_core.neural.reasoning.jurisdiction import JurisdictionContext


def _make_inference(response: str):
    def inference_fn(prompt: str) -> str:
        return response
    return inference_fn


# ---------------------------------------------------------------------------
# reason
# ---------------------------------------------------------------------------

class TestReason:
    def test_returns_structured_result(self):
        result_json = {
            "steps": [
                {"step_number": 1, "reasoning": "Analyze revenue", "conclusion": "Revenue is $1M"},
                {"step_number": 2, "reasoning": "Analyze costs", "conclusion": "Costs are $800K"},
            ],
            "final_answer": "Profit margin is 20%",
            "confidence": 0.9,
            "assumptions": ["Using provided data only"],
        }
        cot = ChainOfThought(_make_inference(json.dumps(result_json)))
        result = cot.reason("What is the profit margin?", {"revenue": 1000000, "costs": 800000})
        assert result["final_answer"] == "Profit margin is 20%"
        assert len(result["steps"]) == 2
        assert result["confidence"] == 0.9

    def test_empty_question_returns_empty(self):
        cot = ChainOfThought(_make_inference("should not be called"))
        result = cot.reason("")
        assert result["final_answer"] == ""
        assert result["confidence"] == 0.0

    def test_whitespace_question_returns_empty(self):
        cot = ChainOfThought(_make_inference("should not be called"))
        result = cot.reason("   ")
        assert result["steps"] == []

    def test_no_context(self):
        result_json = {
            "steps": [{"step_number": 1, "reasoning": "think", "conclusion": "done"}],
            "final_answer": "42",
            "confidence": 0.7,
            "assumptions": [],
        }
        cot = ChainOfThought(_make_inference(json.dumps(result_json)))
        result = cot.reason("What is the meaning of life?")
        assert result["final_answer"] == "42"

    def test_fallback_on_invalid_json(self):
        cot = ChainOfThought(_make_inference("The answer is clearly 42."))
        result = cot.reason("What is the meaning of life?")
        assert result["final_answer"] == "The answer is clearly 42."
        assert result["confidence"] == 0.5

    def test_inference_receives_context(self):
        calls = []
        def mock(prompt):
            calls.append(prompt)
            return json.dumps({"steps": [], "final_answer": "ok", "confidence": 1.0, "assumptions": []})
        cot = ChainOfThought(mock)
        cot.reason("test?", {"key": "value"})
        assert "key" in calls[0]
        assert "value" in calls[0]


# ---------------------------------------------------------------------------
# analyze_with_domain
# ---------------------------------------------------------------------------

class TestAnalyzeWithDomain:
    def test_returns_domain_analysis(self):
        analysis = {
            "domain": "tax",
            "jurisdiction": {"country": "IN"},
            "reasoning_steps": [{"step": 1, "analysis": "Check GST", "finding": "Compliant"}],
            "summary": "Tax analysis complete",
            "key_insights": ["GST compliant"],
            "risk_assessment": {"level": "low", "factors": []},
            "recommendations": ["File on time"],
            "confidence": 0.88,
        }
        cot = ChainOfThought(_make_inference(json.dumps(analysis)))
        jur = JurisdictionContext(country="IN", tax_regime="GST", currency="INR")
        result = cot.analyze_with_domain("Tax return data...", "tax", jur)
        assert result["domain"] == "tax"
        assert result["confidence"] == 0.88

    def test_empty_content_returns_empty(self):
        cot = ChainOfThought(_make_inference("should not be called"))
        result = cot.analyze_with_domain("", "finance")
        assert result["summary"] == ""
        assert result["confidence"] == 0.0

    def test_none_jurisdiction_uses_default(self):
        analysis = {
            "domain": "finance",
            "jurisdiction": {"country": "Unknown"},
            "reasoning_steps": [],
            "summary": "Analysis",
            "key_insights": [],
            "risk_assessment": {"level": "low", "factors": []},
            "recommendations": [],
            "confidence": 0.6,
        }
        calls = []
        def mock(prompt):
            calls.append(prompt)
            return json.dumps(analysis)
        cot = ChainOfThought(mock)
        cot.analyze_with_domain("some content", "finance", None)
        assert "Unknown" in calls[0]

    def test_fallback_on_invalid_json(self):
        cot = ChainOfThought(_make_inference("Everything looks good."))
        result = cot.analyze_with_domain("doc content", "legal")
        assert result["summary"] == "Everything looks good."
        assert result["domain"] == "legal"
