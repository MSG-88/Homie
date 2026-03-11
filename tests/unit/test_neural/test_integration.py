from unittest.mock import MagicMock
from datetime import datetime, timezone

from homie_core.neural.context_engine import SemanticContextEngine
from homie_core.neural.activity_classifier import ActivityClassifier
from homie_core.neural.sentiment import SentimentAnalyzer
from homie_core.neural.intent_inferencer import IntentInferencer
from homie_core.intelligence.observer_loop import ObserverLoop
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.context.screen_monitor import WindowInfo
from homie_core.memory.working import WorkingMemory


def _fake_embed(text):
    val = hash(text) % 1000 / 1000.0
    return [val, 1.0 - val, val * 0.5, (1.0 - val) * 0.5]


def test_observer_with_neural_context():
    wm = WorkingMemory()
    tg = TaskGraph()
    context_engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    classifier = ActivityClassifier(embed_fn=_fake_embed, embed_dim=4)
    classifier._init_prototypes()

    loop = ObserverLoop(
        working_memory=wm,
        task_graph=tg,
        context_engine=context_engine,
        activity_classifier=classifier,
    )

    window = WindowInfo(
        title="engine.py - Homie",
        process_name="Code.exe",
        pid=1234,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    loop._handle_window_change(window)

    # Context engine should have been updated
    vec = context_engine.get_context_vector()
    assert any(v != 0.0 for v in vec)

    # Working memory should have activity classification
    activity = wm.get("activity_type")
    assert activity is not None


def test_observer_without_neural_still_works():
    """Backward compatibility — observer works without neural components."""
    wm = WorkingMemory()
    tg = TaskGraph()
    loop = ObserverLoop(working_memory=wm, task_graph=tg)

    window = WindowInfo(
        title="engine.py - Homie",
        process_name="Code.exe",
        pid=1234,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    loop._handle_window_change(window)

    assert wm.get("active_window") == "engine.py - Homie"
    assert len(tg.get_tasks()) == 1
