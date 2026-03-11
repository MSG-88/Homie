from unittest.mock import MagicMock
from datetime import datetime, timezone

from homie_core.neural.context_engine import SemanticContextEngine
from homie_core.neural.activity_classifier import ActivityClassifier
from homie_core.neural.sentiment import SentimentAnalyzer
from homie_core.neural.intent_inferencer import IntentInferencer
from homie_core.neural.rhythm_model import CircadianRhythmModel
from homie_core.neural.behavioral_profile import BehavioralProfile
from homie_core.neural.preference_engine import PreferenceEngine
from homie_core.intelligence.observer_loop import ObserverLoop
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.intelligence.workflow_predictor import WorkflowPredictor
from homie_core.intelligence.flow_detector import FlowDetector
from homie_core.intelligence.planner import HTNPlanner, DecompositionRule
from homie_core.intelligence.action_selector import MCTSActionSelector
from homie_core.intelligence.self_reflection import SelfReflection
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


def test_full_neural_pipeline():
    """End-to-end: Phase 1 + 2 + 3 components work together via ObserverLoop."""
    wm = WorkingMemory()
    tg = TaskGraph()
    context_engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    classifier = ActivityClassifier(embed_fn=_fake_embed, embed_dim=4)
    classifier._init_prototypes()
    rhythm = CircadianRhythmModel()
    profile = BehavioralProfile(embed_dim=4)
    prefs = PreferenceEngine()
    workflow = WorkflowPredictor()
    flow = FlowDetector(window_size=10)

    loop = ObserverLoop(
        working_memory=wm,
        task_graph=tg,
        context_engine=context_engine,
        activity_classifier=classifier,
        rhythm_model=rhythm,
        behavioral_profile=profile,
        preference_engine=prefs,
        workflow_predictor=workflow,
        flow_detector=flow,
    )

    window = WindowInfo(
        title="engine.py - Homie",
        process_name="Code.exe",
        pid=1234,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    loop._handle_window_change(window)

    # Phase 1: context + classification
    assert any(v != 0.0 for v in context_engine.get_context_vector())
    assert wm.get("activity_type") is not None

    # Phase 2: rhythm, profile, preferences
    assert rhythm._hourly_buckets
    assert profile.sample_count > 0
    assert prefs.get_preferences("activity")

    # Phase 3: workflow + flow
    assert flow._window
    assert wm.get("flow_score") is not None


def test_phase4_autonomous_pipeline():
    """Phase 4 components work together for decision-making."""
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="help_user",
        subtasks=["assess", "suggest", "verify"],
    ))
    for p in ["assess", "suggest", "verify"]:
        planner.mark_primitive(p, cost=1.0)

    plan = planner.plan("help_user")
    assert plan is not None
    assert len(plan) == 3

    # MCTS selects best action
    selector = MCTSActionSelector(n_iterations=50)
    best = selector.search(
        ["suggest_break", "suggest_resource", "stay_quiet"],
        {"flow_score": 0.3},
        lambda a, c: 0.8 if a == "suggest_break" and c.get("flow_score", 1) < 0.5 else 0.2,
    )
    assert best == "suggest_break"

    # Self-reflection evaluates the action
    reflection = SelfReflection()
    result = reflection.score_action(
        best,
        {"flow_score": 0.3},
        {"relevance": 0.9, "helpfulness": 0.8, "urgency": 0.6},
    )
    assert result.calibrated_confidence > 0.5
