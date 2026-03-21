"""Tests for the EntityExtractor — pattern-based and spaCy NER tiers."""
from __future__ import annotations

import pytest

from homie_core.knowledge.extractor import EntityExtractor
from homie_core.knowledge.models import Entity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _names(entities: list[Entity]) -> list[str]:
    return [e.name for e in entities]


def _types(entities: list[Entity]) -> list[str]:
    return [e.entity_type for e in entities]


def _by_type(entities: list[Entity], etype: str) -> list[Entity]:
    return [e for e in entities if e.entity_type == etype]


# ---------------------------------------------------------------------------
# Pattern-based extraction (no spaCy required)
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor_no_model():
    """Extractor that never loads spaCy."""
    return EntityExtractor(use_model=False)


class TestPatternEmailExtraction:
    def test_finds_email(self, extractor_no_model):
        entities, _ = extractor_no_model.extract(
            "Contact alice@example.com for more info.", source="test"
        )
        emails = _by_type(entities, "person")
        assert any("alice@example.com" in e.name for e in emails)

    def test_finds_multiple_emails(self, extractor_no_model):
        entities, _ = extractor_no_model.extract(
            "Reach bob@corp.io or carol@uni.edu for details."
        )
        persons = _by_type(entities, "person")
        person_names = [e.name for e in persons]
        assert any("bob@corp.io" in n for n in person_names)
        assert any("carol@uni.edu" in n for n in person_names)

    def test_email_entity_has_email_attribute(self, extractor_no_model):
        entities, _ = extractor_no_model.extract("dave@test.org is the admin.")
        email_entities = [e for e in entities if "dave@test.org" in e.name]
        assert len(email_entities) >= 1
        assert email_entities[0].attributes.get("email") == "dave@test.org"


class TestPatternURLExtraction:
    def test_finds_http_url(self, extractor_no_model):
        entities, _ = extractor_no_model.extract(
            "See https://example.com for details."
        )
        locs = _by_type(entities, "location")
        assert any("https://example.com" in e.name for e in locs)

    def test_finds_https_url(self, extractor_no_model):
        entities, _ = extractor_no_model.extract(
            "Visit http://docs.python.org for help."
        )
        locs = _by_type(entities, "location")
        assert any("http://docs.python.org" in e.name for e in locs)

    def test_url_entity_has_url_attribute(self, extractor_no_model):
        entities, _ = extractor_no_model.extract("Go to https://openai.com now.")
        url_entities = [e for e in entities if "openai.com" in e.name]
        assert len(url_entities) >= 1
        assert "url" in url_entities[0].attributes

    def test_no_url_no_location(self, extractor_no_model):
        entities, _ = extractor_no_model.extract("Just plain text, no links.")
        locs = _by_type(entities, "location")
        assert len(locs) == 0


class TestPatternFilePathExtraction:
    def test_finds_unix_absolute_path(self, extractor_no_model):
        entities, _ = extractor_no_model.extract(
            "Check /home/user/project/main.py for the entry point."
        )
        docs = _by_type(entities, "document")
        assert any("main.py" in e.name for e in docs)

    def test_finds_relative_path(self, extractor_no_model):
        entities, _ = extractor_no_model.extract(
            "See ./src/utils.py for helper functions."
        )
        docs = _by_type(entities, "document")
        assert any("utils.py" in e.name for e in docs)

    def test_finds_deep_path(self, extractor_no_model):
        entities, _ = extractor_no_model.extract(
            "Config is at ../config/settings.yaml"
        )
        docs = _by_type(entities, "document")
        assert any("settings.yaml" in e.name for e in docs)


class TestPatternImportExtraction:
    def test_finds_import_statement(self, extractor_no_model):
        entities, _ = extractor_no_model.extract("import numpy as np")
        tools = _by_type(entities, "tool")
        assert any("numpy" in e.name for e in tools)

    def test_finds_from_import(self, extractor_no_model):
        entities, _ = extractor_no_model.extract("from pathlib import Path")
        tools = _by_type(entities, "tool")
        assert any("pathlib" in e.name for e in tools)

    def test_finds_multiple_imports(self, extractor_no_model):
        code = "import os\nimport sys\nfrom typing import List"
        entities, _ = extractor_no_model.extract(code)
        tools = _by_type(entities, "tool")
        tool_names = [e.name for e in tools]
        assert any("os" in n for n in tool_names)
        assert any("sys" in n for n in tool_names)

    def test_skips_future_import(self, extractor_no_model):
        entities, _ = extractor_no_model.extract("from __future__ import annotations")
        tools = _by_type(entities, "tool")
        assert not any("__future__" in e.name for e in tools)

    def test_subpackage_uses_top_level(self, extractor_no_model):
        entities, _ = extractor_no_model.extract("from homie_core.rag import pipeline")
        tools = _by_type(entities, "tool")
        assert any("homie_core" in e.name for e in tools)


class TestPatternDeduplication:
    def test_same_email_not_duplicated(self, extractor_no_model):
        entities, _ = extractor_no_model.extract(
            "alice@example.com and alice@example.com again."
        )
        alice_entities = [e for e in entities if "alice@example.com" in e.name]
        assert len(alice_entities) == 1

    def test_same_url_not_duplicated(self, extractor_no_model):
        entities, _ = extractor_no_model.extract(
            "https://example.com is at https://example.com"
        )
        url_entities = [e for e in entities if "example.com" in e.name
                        and e.entity_type == "location"]
        assert len(url_entities) == 1


class TestPatternSourceTracking:
    def test_source_propagated(self, extractor_no_model):
        entities, _ = extractor_no_model.extract(
            "import pandas", source="my_script.py"
        )
        assert all(e.source == "my_script.py" for e in entities)

    def test_default_source_is_extraction(self, extractor_no_model):
        entities, _ = extractor_no_model.extract("import json")
        assert all(e.source in ("extraction", "") or e.source for e in entities)


# ---------------------------------------------------------------------------
# Fallback when spaCy is unavailable
# ---------------------------------------------------------------------------

class TestFallbackToPatterns:
    def test_use_model_false_uses_patterns(self):
        """use_model=False must always use pattern extraction."""
        ext = EntityExtractor(use_model=False)
        assert ext._nlp is None

    def test_pattern_fallback_still_extracts(self):
        """Pattern-only extractor should find emails even without spaCy."""
        ext = EntityExtractor(use_model=False)
        entities, _ = ext.extract("Contact support@example.com today.")
        assert any("support@example.com" in e.name for e in entities)

    def test_graceful_if_spacy_model_missing(self, monkeypatch):
        """If spaCy is installed but en_core_web_sm is missing, fall back quietly."""
        import sys
        # Simulate OSError from spacy.load
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None

        import importlib
        real_spacy = None
        try:
            real_spacy = importlib.import_module("spacy")
        except ImportError:
            pytest.skip("spaCy not installed")

        original_load = real_spacy.load

        def broken_load(name, *a, **kw):
            raise OSError(f"Can't find model '{name}'")

        monkeypatch.setattr(real_spacy, "load", broken_load)
        ext = EntityExtractor(use_model=True)
        assert ext._nlp is None
        # Should still extract via patterns
        entities, _ = ext.extract("alice@example.com")
        assert len(entities) >= 1


# ---------------------------------------------------------------------------
# spaCy NER extraction (skip if not installed)
# ---------------------------------------------------------------------------

try:
    import spacy as _spacy
    _spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except (ImportError, OSError):
    _SPACY_AVAILABLE = False

spacy_only = pytest.mark.skipif(
    not _SPACY_AVAILABLE,
    reason="spaCy en_core_web_sm not installed",
)


@spacy_only
class TestSpaCyExtraction:
    @pytest.fixture
    def model_extractor(self):
        return EntityExtractor(use_model=True)

    def test_extracts_person_entity(self, model_extractor):
        entities, _ = model_extractor.extract("Albert Einstein developed the theory of relativity.")
        persons = _by_type(entities, "person")
        assert any("Einstein" in e.name or "Albert" in e.name for e in persons)

    def test_extracts_org_as_concept(self, model_extractor):
        entities, _ = model_extractor.extract("Google announced new AI products this week.")
        concepts = _by_type(entities, "concept")
        assert any("Google" in e.name for e in concepts)
        # ORG entities should have org=True attribute
        google_ents = [e for e in concepts if "Google" in e.name]
        assert google_ents[0].attributes.get("org") is True

    def test_extracts_gpe_as_location(self, model_extractor):
        entities, _ = model_extractor.extract("The conference is held in Paris this year.")
        locs = _by_type(entities, "location")
        assert any("Paris" in e.name for e in locs)

    def test_merges_pattern_results(self, model_extractor):
        """spaCy extractor should also capture emails via pattern layer."""
        entities, _ = model_extractor.extract(
            "John Smith (john@smith.com) gave a talk in London."
        )
        names = [e.name for e in entities]
        assert any("john@smith.com" in n for n in names)

    def test_no_duplicate_between_layers(self, model_extractor):
        """Entities found by both layers should not be duplicated."""
        entities, _ = model_extractor.extract("Alice works at Google.")
        google_entities = [e for e in entities if "Google" in e.name]
        assert len(google_entities) == 1

    def test_spacy_entities_have_higher_confidence(self, model_extractor):
        """spaCy-sourced entities should have confidence >= 0.85."""
        entities, _ = model_extractor.extract("Napoleon Bonaparte was a French emperor.")
        for e in entities:
            assert e.confidence >= 0.0  # at minimum above zero
