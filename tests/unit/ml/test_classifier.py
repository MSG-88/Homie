"""Tests for TextClassifier — TF-IDF + LogReg (sklearn) or Naive Bayes fallback."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from homie_core.ml.classifier import TextClassifier, _NaiveBayesFallback, _tokenize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INTENT_TEXTS = [
    "turn on the lights",
    "switch on the lamp",
    "lights on please",
    "turn off the lights",
    "switch off the lamp",
    "lights off please",
    "what is the weather today",
    "tell me the forecast",
    "will it rain tomorrow",
    "play some music",
    "play a song",
    "put on some tunes",
]
INTENT_LABELS = [
    "lights_on", "lights_on", "lights_on",
    "lights_off", "lights_off", "lights_off",
    "weather", "weather", "weather",
    "music", "music", "music",
]


@pytest.fixture
def clf():
    return TextClassifier("intent", classes=["lights_on", "lights_off", "weather", "music"])


@pytest.fixture
def trained_clf(clf):
    clf.train(INTENT_TEXTS, INTENT_LABELS)
    return clf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTextClassifierInit:
    def test_creates_with_name_and_classes(self, clf):
        assert clf.name == "intent"
        assert clf.model_type == "classifier"
        assert set(clf.classes) == {"lights_on", "lights_off", "weather", "music"}

    def test_is_not_trained_initially(self, clf):
        assert clf.is_trained is False

    def test_repr(self, clf):
        r = repr(clf)
        assert "intent" in r
        assert "untrained" in r


class TestTextClassifierTrain:
    def test_train_returns_metrics(self, clf):
        metrics = clf.train(INTENT_TEXTS, INTENT_LABELS)
        assert "accuracy" in metrics
        assert "n_samples" in metrics
        assert metrics["n_samples"] == len(INTENT_TEXTS)

    def test_is_trained_after_training(self, clf):
        clf.train(INTENT_TEXTS, INTENT_LABELS)
        assert clf.is_trained is True

    def test_train_empty_raises(self, clf):
        with pytest.raises(ValueError, match="empty"):
            clf.train([], [])

    def test_train_mismatched_lengths_raises(self, clf):
        with pytest.raises(ValueError, match="same length"):
            clf.train(["hello"], ["a", "b"])

    def test_train_discovers_new_classes(self):
        clf = TextClassifier("test", classes=["a"])
        clf.train(["hello world", "foo bar"], ["a", "b"])
        assert "b" in clf.classes


class TestTextClassifierPredict:
    def test_predict_returns_list(self, trained_clf):
        preds = trained_clf.predict(["turn on the lights"])
        assert isinstance(preds, list)
        assert len(preds) == 1

    def test_predict_correct_label(self, trained_clf):
        preds = trained_clf.predict(["turn on the lights", "play a song"])
        assert preds[0] == "lights_on"
        assert preds[1] == "music"

    def test_predict_before_training_raises(self, clf):
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict(["hello"])

    def test_predict_proba_returns_dicts(self, trained_clf):
        proba = trained_clf.predict_proba(["turn on the lights"])
        assert isinstance(proba, list)
        assert isinstance(proba[0], dict)
        assert set(proba[0].keys()).issubset(set(trained_clf.classes))
        # probabilities should sum to ~1
        total = sum(proba[0].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_proba_before_training_raises(self, clf):
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict_proba(["hello"])


class TestTextClassifierPersistence:
    def test_save_and_load_json(self, trained_clf, tmp_path):
        path = tmp_path / "model.json"
        trained_clf.save(path)
        assert path.exists()

        loaded = TextClassifier("loaded", classes=[])
        loaded.load(path)
        assert loaded.is_trained
        assert loaded.name == trained_clf.name

    def test_loaded_model_predicts(self, trained_clf, tmp_path):
        path = tmp_path / "model.json"
        trained_clf.save(path)

        loaded = TextClassifier("loaded", classes=[])
        loaded.load(path)
        preds = loaded.predict(["turn on the lights"])
        assert preds[0] == "lights_on"

    def test_load_nonexistent_raises(self, clf, tmp_path):
        with pytest.raises(FileNotFoundError):
            clf.load(tmp_path / "nonexistent.json")


class TestNaiveBayesFallback:
    def test_fit_and_predict(self):
        nb = _NaiveBayesFallback(["pos", "neg"])
        nb.fit(["great awesome good", "terrible bad awful"], ["pos", "neg"])
        preds = nb.predict(["awesome great"])
        assert preds[0] == "pos"

    def test_to_dict_roundtrip(self):
        nb = _NaiveBayesFallback(["a", "b"])
        nb.fit(["hello world", "foo bar"], ["a", "b"])
        data = nb.to_dict()
        restored = _NaiveBayesFallback.from_dict(data)
        assert restored.predict(["hello"]) == nb.predict(["hello"])


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello, World! 123")
        assert tokens == ["hello", "world", "123"]

    def test_empty(self):
        assert _tokenize("") == []


class TestForceFallbackBackend:
    """Ensure the naive Bayes fallback works even when sklearn is importable."""

    def test_naive_bayes_backend(self):
        clf = TextClassifier("test_nb", classes=["pos", "neg"])
        clf._backend = "naive_bayes"  # force fallback
        clf.train(
            ["good great awesome", "bad terrible awful", "nice wonderful", "horrible nasty"],
            ["pos", "neg", "pos", "neg"],
        )
        assert clf.is_trained
        preds = clf.predict(["great wonderful"])
        assert preds[0] == "pos"
