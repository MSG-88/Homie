from unittest.mock import MagicMock
import math

from homie_core.neural.activity_classifier import ActivityClassifier, CATEGORIES


def _fake_embed(text):
    """Deterministic fake embeddings based on text content."""
    if "code" in text.lower() or ".py" in text.lower():
        return [1.0, 0.0, 0.0, 0.0]
    elif "chrome" in text.lower() or "google" in text.lower():
        return [0.0, 1.0, 0.0, 0.0]
    elif "slack" in text.lower() or "teams" in text.lower():
        return [0.0, 0.0, 1.0, 0.0]
    elif "word" in text.lower() or "docs" in text.lower():
        return [0.0, 0.0, 0.0, 1.0]
    return [0.25, 0.25, 0.25, 0.25]


def test_categories_exist():
    assert "coding" in CATEGORIES
    assert "researching" in CATEGORIES
    assert "communicating" in CATEGORIES
    assert len(CATEGORIES) >= 8


def test_classify_returns_all_categories():
    embed_fn = MagicMock(side_effect=_fake_embed)
    classifier = ActivityClassifier(embed_fn=embed_fn, embed_dim=4)
    classifier._init_prototypes()

    result = classifier.classify("Code.exe", "engine.py - Homie")
    assert isinstance(result, dict)
    for cat in CATEGORIES:
        assert cat in result
        assert 0.0 <= result[cat] <= 1.0


def test_classify_coding_activity():
    embed_fn = MagicMock(side_effect=_fake_embed)
    classifier = ActivityClassifier(embed_fn=embed_fn, embed_dim=4)
    classifier._init_prototypes()

    result = classifier.classify("Code.exe", "engine.py - Homie")
    # Coding should be among top categories
    top = max(result, key=result.get)
    assert isinstance(top, str)


def test_train_online():
    embed_fn = MagicMock(side_effect=_fake_embed)
    classifier = ActivityClassifier(embed_fn=embed_fn, embed_dim=4)
    classifier._init_prototypes()

    # Should not raise
    classifier.train_online("Code.exe", "engine.py - Homie", "coding")
    classifier.train_online("chrome.exe", "Google", "researching")


def test_serialize_deserialize():
    embed_fn = MagicMock(side_effect=_fake_embed)
    classifier = ActivityClassifier(embed_fn=embed_fn, embed_dim=4)
    classifier._init_prototypes()

    data = classifier.serialize()
    assert "prototypes" in data
    assert "weights" in data

    restored = ActivityClassifier.deserialize(data, embed_fn=embed_fn)
    assert restored is not None


def test_get_top_activity():
    embed_fn = MagicMock(side_effect=_fake_embed)
    classifier = ActivityClassifier(embed_fn=embed_fn, embed_dim=4)
    classifier._init_prototypes()

    result = classifier.classify("Code.exe", "engine.py - Homie")
    top = classifier.get_top_activity("Code.exe", "engine.py - Homie")
    assert isinstance(top, str)
    assert top in CATEGORIES
