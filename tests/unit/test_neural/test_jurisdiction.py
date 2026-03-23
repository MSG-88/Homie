"""Tests for the JurisdictionEngine module."""

import pytest

from homie_core.neural.reasoning.jurisdiction import (
    JurisdictionContext,
    JurisdictionEngine,
)


@pytest.fixture
def engine():
    return JurisdictionEngine()


# ---------------------------------------------------------------------------
# JurisdictionContext dataclass
# ---------------------------------------------------------------------------

class TestJurisdictionContext:
    def test_defaults(self):
        ctx = JurisdictionContext(country="US")
        assert ctx.country == "US"
        assert ctx.state_province == ""
        assert ctx.fiscal_year_start == "January"

    def test_full_init(self):
        ctx = JurisdictionContext(
            country="IN",
            state_province="Karnataka",
            tax_regime="GST",
            currency="INR",
            fiscal_year_start="April",
            legal_framework="Indian Common Law",
        )
        assert ctx.currency == "INR"
        assert ctx.fiscal_year_start == "April"


# ---------------------------------------------------------------------------
# detect_from_config
# ---------------------------------------------------------------------------

class TestDetectFromConfig:
    def test_us_eastern_timezone(self, engine):
        ctx = engine.detect_from_config({"timezone": "America/New_York"})
        assert ctx.country == "US"
        assert ctx.state_province == "New York"
        assert ctx.tax_regime == "IRS"
        assert ctx.currency == "USD"

    def test_india_timezone(self, engine):
        ctx = engine.detect_from_config({"timezone": "Asia/Kolkata"})
        assert ctx.country == "IN"
        assert ctx.tax_regime == "GST"
        assert ctx.currency == "INR"
        assert ctx.fiscal_year_start == "April"

    def test_australia_timezone(self, engine):
        ctx = engine.detect_from_config({"timezone": "Australia/Sydney"})
        assert ctx.country == "AU"
        assert ctx.tax_regime == "ATO"
        assert ctx.fiscal_year_start == "July"

    def test_uk_timezone(self, engine):
        ctx = engine.detect_from_config({"timezone": "Europe/London"})
        assert ctx.country == "GB"
        assert ctx.tax_regime == "HMRC"

    def test_unknown_timezone_uses_region_fallback(self, engine):
        ctx = engine.detect_from_config({"timezone": "America/Phoenix"})
        assert ctx.country == "US"
        assert ctx.tax_regime == "IRS"

    def test_config_country_override(self, engine):
        ctx = engine.detect_from_config({
            "timezone": "America/New_York",
            "country": "CA",
        })
        assert ctx.country == "CA"  # overridden

    def test_none_config(self, engine):
        ctx = engine.detect_from_config(None)
        assert ctx.country == "Unknown"

    def test_empty_config(self, engine):
        ctx = engine.detect_from_config({})
        assert ctx.country == "Unknown"


# ---------------------------------------------------------------------------
# get_tax_rules
# ---------------------------------------------------------------------------

class TestGetTaxRules:
    def test_irs_rules(self, engine):
        ctx = JurisdictionContext(country="US", tax_regime="IRS")
        rules = engine.get_tax_rules(ctx)
        assert rules["regime"] == "IRS"
        assert "filing_deadline" in rules

    def test_gst_rules(self, engine):
        ctx = JurisdictionContext(country="IN", tax_regime="GST")
        rules = engine.get_tax_rules(ctx)
        assert "GST" in rules["regime"]

    def test_unknown_regime(self, engine):
        ctx = JurisdictionContext(country="XX", tax_regime="UNKNOWN")
        rules = engine.get_tax_rules(ctx)
        assert "note" in rules


# ---------------------------------------------------------------------------
# get_legal_framework
# ---------------------------------------------------------------------------

class TestGetLegalFramework:
    def test_us_federal(self, engine):
        ctx = JurisdictionContext(country="US", legal_framework="US Federal + State")
        fw = engine.get_legal_framework(ctx)
        assert "United States" in fw

    def test_uk_common_law(self, engine):
        ctx = JurisdictionContext(country="GB", legal_framework="UK Common Law")
        fw = engine.get_legal_framework(ctx)
        assert "English" in fw

    def test_unknown_framework(self, engine):
        ctx = JurisdictionContext(country="XX", legal_framework="Martian Law")
        fw = engine.get_legal_framework(ctx)
        assert "Martian Law" in fw
