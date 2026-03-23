"""Tests for the DomainExpert module."""

import json
import pytest

from homie_core.neural.reasoning.domain_expert import DomainExpert, VALID_DOMAINS
from homie_core.neural.reasoning.jurisdiction import JurisdictionContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_inference(response: str):
    """Return a mock inference_fn that always returns *response*."""
    def inference_fn(prompt: str) -> str:
        return response
    return inference_fn


@pytest.fixture
def expert():
    return DomainExpert(_make_inference("accounting"))


# ---------------------------------------------------------------------------
# classify_domain
# ---------------------------------------------------------------------------

class TestClassifyDomain:
    def test_returns_valid_domain(self):
        expert = DomainExpert(_make_inference("finance"))
        assert expert.classify_domain("budget analysis quarterly") == "finance"

    def test_returns_accounting(self):
        expert = DomainExpert(_make_inference("accounting"))
        assert expert.classify_domain("ledger reconciliation") == "accounting"

    def test_returns_legal(self):
        expert = DomainExpert(_make_inference("legal"))
        assert expert.classify_domain("contract clause review") == "legal"

    def test_returns_tax(self):
        expert = DomainExpert(_make_inference("The domain is tax."))
        assert expert.classify_domain("tax deduction filing") == "tax"

    def test_defaults_to_general_on_unknown_response(self):
        expert = DomainExpert(_make_inference("something random"))
        assert expert.classify_domain("hello world") == "general"

    def test_empty_text_returns_general(self):
        expert = DomainExpert(_make_inference("finance"))
        assert expert.classify_domain("") == "general"

    def test_whitespace_only_returns_general(self):
        expert = DomainExpert(_make_inference("finance"))
        assert expert.classify_domain("   ") == "general"

    def test_inference_called_with_text(self):
        calls = []
        def mock(prompt):
            calls.append(prompt)
            return "legal"
        expert = DomainExpert(mock)
        expert.classify_domain("contract review")
        assert len(calls) == 1
        assert "contract review" in calls[0]


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------

class TestExtractEntities:
    def test_returns_parsed_entities(self):
        entities = [
            {"type": "amount", "value": "$5,000", "context": "invoice total"},
            {"type": "date", "value": "2026-03-15", "context": "due date"},
        ]
        expert = DomainExpert(_make_inference(json.dumps(entities)))
        result = expert.extract_entities("Invoice total $5,000 due 2026-03-15", "accounting")
        assert len(result) == 2
        assert result[0]["type"] == "amount"

    def test_handles_wrapped_json(self):
        entities = [{"type": "party", "value": "Acme Corp", "context": "vendor"}]
        expert = DomainExpert(_make_inference(f"Here are the entities: {json.dumps(entities)}"))
        result = expert.extract_entities("Acme Corp invoice", "accounting")
        assert len(result) == 1

    def test_empty_text_returns_empty(self):
        expert = DomainExpert(_make_inference("[]"))
        assert expert.extract_entities("", "finance") == []

    def test_invalid_json_returns_empty(self):
        expert = DomainExpert(_make_inference("not json at all"))
        assert expert.extract_entities("some text", "legal") == []

    def test_invalid_domain_defaults_to_general(self):
        calls = []
        def mock(prompt):
            calls.append(prompt)
            return "[]"
        expert = DomainExpert(mock)
        expert.extract_entities("text", "nonexistent")
        assert "general" in calls[0]

    def test_returns_list_type(self):
        expert = DomainExpert(_make_inference("[]"))
        result = expert.extract_entities("no entities here", "tax")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# analyze_document
# ---------------------------------------------------------------------------

class TestAnalyzeDocument:
    def test_returns_structured_analysis(self):
        analysis = {
            "summary": "This is a quarterly P&L.",
            "key_findings": ["Revenue up 10%"],
            "risk_flags": ["High overhead ratio"],
            "action_items": ["Review overhead costs"],
            "confidence": 0.85,
        }
        expert = DomainExpert(_make_inference(json.dumps(analysis)))
        result = expert.analyze_document("Quarterly P&L statement...", "finance")
        assert result["summary"] == "This is a quarterly P&L."
        assert result["confidence"] == 0.85

    def test_empty_content_returns_empty_analysis(self):
        expert = DomainExpert(_make_inference("should not be called"))
        result = expert.analyze_document("", "finance")
        assert result["summary"] == ""
        assert result["confidence"] == 0.0

    def test_invalid_json_returns_fallback(self):
        expert = DomainExpert(_make_inference("The document looks fine."))
        result = expert.analyze_document("some doc", "legal")
        assert "summary" in result
        assert result["summary"] == "The document looks fine."


# ---------------------------------------------------------------------------
# apply_rules
# ---------------------------------------------------------------------------

class TestApplyRules:
    def test_applies_rules_with_jurisdiction_context(self):
        rules_result = {
            "applicable_rules": ["GST applies at 18%"],
            "compliance_status": "compliant",
            "findings": ["All invoices have GSTIN"],
            "recommendations": [],
            "risk_level": "low",
        }
        expert = DomainExpert(_make_inference(json.dumps(rules_result)))
        jur = JurisdictionContext(country="IN", tax_regime="GST", currency="INR")
        entities = [{"type": "amount", "value": "10000", "context": "sale"}]
        result = expert.apply_rules(entities, "tax", jur)
        assert result["compliance_status"] == "compliant"

    def test_empty_entities_returns_needs_review(self):
        expert = DomainExpert(_make_inference("should not be called"))
        result = expert.apply_rules([], "tax")
        assert result["compliance_status"] == "needs_review"

    def test_accepts_dict_jurisdiction(self):
        rules_result = {
            "applicable_rules": [],
            "compliance_status": "needs_review",
            "findings": [],
            "recommendations": [],
            "risk_level": "low",
        }
        expert = DomainExpert(_make_inference(json.dumps(rules_result)))
        result = expert.apply_rules(
            [{"type": "amount", "value": "500"}],
            "finance",
            {"country": "US", "tax_regime": "IRS"},
        )
        assert "compliance_status" in result

    def test_none_jurisdiction(self):
        rules_result = {
            "applicable_rules": [],
            "compliance_status": "needs_review",
            "findings": [],
            "recommendations": [],
            "risk_level": "low",
        }
        expert = DomainExpert(_make_inference(json.dumps(rules_result)))
        result = expert.apply_rules(
            [{"type": "clause", "value": "indemnity"}],
            "legal",
            None,
        )
        assert "risk_level" in result
