"""Tests for MLModelRegistry — model tracking and lifecycle."""

import pytest

from homie_core.ml.registry import MLModelRegistry
from homie_core.ml.classifier import TextClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return MLModelRegistry()


@pytest.fixture
def model():
    return TextClassifier("test_model", classes=["a", "b"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_returns_name(self, registry, model):
        name = registry.register(model)
        assert name == "test_model"

    def test_registered_model_retrievable(self, registry, model):
        registry.register(model)
        assert registry.get("test_model") is model

    def test_has(self, registry, model):
        registry.register(model)
        assert registry.has("test_model") is True
        assert registry.has("nonexistent") is False


class TestGet:
    def test_get_missing_returns_none(self, registry):
        assert registry.get("nope") is None


class TestListActive:
    def test_list_active_includes_registered(self, registry, model):
        registry.register(model)
        assert "test_model" in registry.list_active()

    def test_list_active_excludes_archived(self, registry, model):
        registry.register(model)
        registry.archive("test_model")
        assert "test_model" not in registry.list_active()

    def test_list_all_includes_archived(self, registry, model):
        registry.register(model)
        registry.archive("test_model")
        assert "test_model" in registry.list_all()


class TestMetrics:
    def test_get_metrics_empty_initially(self, registry, model):
        registry.register(model)
        assert registry.get_metrics("test_model") == {}

    def test_update_metrics(self, registry, model):
        registry.register(model)
        registry.update_metrics("test_model", {"accuracy": 0.95})
        assert registry.get_metrics("test_model") == {"accuracy": 0.95}

    def test_get_metrics_unknown_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get_metrics("ghost")


class TestStatus:
    def test_default_status_active(self, registry, model):
        registry.register(model)
        assert registry.get_status("test_model") == "active"

    def test_set_deployed(self, registry, model):
        registry.register(model)
        registry.set_status("test_model", "deployed")
        assert registry.get_status("test_model") == "deployed"

    def test_invalid_status_raises(self, registry, model):
        registry.register(model)
        with pytest.raises(ValueError, match="Invalid status"):
            registry.set_status("test_model", "banana")


class TestRemove:
    def test_remove_existing(self, registry, model):
        registry.register(model)
        assert registry.remove("test_model") is True
        assert registry.get("test_model") is None

    def test_remove_nonexistent(self, registry):
        assert registry.remove("nope") is False


class TestSummary:
    def test_summary(self, registry, model):
        registry.register(model)
        summaries = registry.summary()
        assert len(summaries) == 1
        assert summaries[0]["name"] == "test_model"
        assert summaries[0]["status"] == "active"
