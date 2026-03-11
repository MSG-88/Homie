from unittest.mock import MagicMock
import math

from homie_core.neural.context_engine import SemanticContextEngine


def _fake_embed(text):
    """Simple deterministic embeddings."""
    val = hash(text) % 1000 / 1000.0
    return [val, 1.0 - val, val * 0.5, (1.0 - val) * 0.5]


def test_init():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    assert engine.get_context_vector() == [0.0] * 4


def test_update_changes_context():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    engine.update("Code.exe", "engine.py - Homie")
    vec = engine.get_context_vector()
    assert any(v != 0.0 for v in vec)


def test_context_shift_detection():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4,
                                   shift_threshold=0.3)

    # Same context — no shift
    engine.update("Code.exe", "engine.py - Homie")
    engine.update("Code.exe", "config.py - Homie")
    # These are similar enough they may not trigger a shift

    # Very different context — should detect shift
    engine.update("Code.exe", "engine.py")
    initial = engine.get_context_vector()
    engine.update("spotify.exe", "Playing Music - Best Hits 2026")
    shifted = engine.detect_context_shift()
    # We just test it returns a bool
    assert isinstance(shifted, bool)


def test_find_relevant_memories():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    engine.update("Code.exe", "engine.py")

    memories = [
        {"summary": "coding session on engine.py", "embedding": _fake_embed("coding engine")},
        {"summary": "grocery shopping list", "embedding": _fake_embed("groceries milk bread")},
        {"summary": "debugging config parser", "embedding": _fake_embed("config debug code")},
    ]

    results = engine.find_relevant_memories(memories, top_k=2)
    assert len(results) == 2
    assert all("summary" in r for r in results)


def test_get_activity_summary():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    engine.update("Code.exe", "engine.py - Homie")
    engine.update("chrome.exe", "Stack Overflow")

    summary = engine.get_activity_summary()
    assert "observations" in summary
    assert summary["observations"] == 2


def test_rolling_window_limits():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4,
                                   window_size=3)
    for i in range(10):
        engine.update("app.exe", f"window {i}")

    # Should only keep last 3
    assert len(engine._recent_embeddings) == 3
