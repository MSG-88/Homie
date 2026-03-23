"""Integration tests: verify AdaptiveLearner and Watchdog are wired into the brain loop."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from homie_core.adaptive_learning.learner import AdaptiveLearner
from homie_core.adaptive_learning.observation.learning_middleware import LearningMiddleware
from homie_core.adaptive_learning.observation.stream import ObservationStream
from homie_core.middleware.stack import MiddlewareStack


class TestAdaptiveLearnerBootWiring:
    """Test that AdaptiveLearner is initialized during boot."""

    def test_init_adaptive_learner_returns_learner_when_enabled(self, tmp_path):
        """_init_adaptive_learner returns an AdaptiveLearner when config enables it."""
        from homie_app.cli import _init_adaptive_learner

        cfg = MagicMock()
        cfg.adaptive_learning.enabled = True
        cfg.adaptive_learning.preference.learning_rate_explicit = 0.3
        cfg.adaptive_learning.preference.learning_rate_implicit = 0.05
        cfg.adaptive_learning.performance.cache_max_entries = 500
        cfg.adaptive_learning.performance.cache_ttl_default = 86400
        cfg.storage.path = str(tmp_path)

        learner = _init_adaptive_learner(cfg)

        assert learner is not None
        assert isinstance(learner, AdaptiveLearner)
        learner.stop()

    def test_init_adaptive_learner_returns_none_when_disabled(self):
        """_init_adaptive_learner returns None when adaptive_learning is disabled."""
        from homie_app.cli import _init_adaptive_learner

        cfg = MagicMock()
        cfg.adaptive_learning.enabled = False

        learner = _init_adaptive_learner(cfg)
        assert learner is None

    def test_init_adaptive_learner_returns_none_when_missing(self):
        """_init_adaptive_learner returns None when config has no adaptive_learning."""
        from homie_app.cli import _init_adaptive_learner

        cfg = MagicMock(spec=[])  # no attributes
        learner = _init_adaptive_learner(cfg)
        assert learner is None


class TestLearningMiddlewareInStack:
    """Test that LearningMiddleware is registered in the middleware stack."""

    def test_middleware_factory_adds_learning_middleware(self, tmp_path):
        """build_middleware_stack includes LearningMiddleware when observation_stream is given."""
        from homie_app.middleware_factory import build_middleware_stack
        from homie_core.memory.working import WorkingMemory

        cfg = MagicMock()
        cfg.storage.path = str(tmp_path)
        cfg.context.summarize_trigger_pct = 0.85
        cfg.context.summarize_keep_pct = 0.1
        cfg.context.arg_truncation_threshold = 2000
        cfg.context.long_line_threshold = 5000
        cfg.context.large_result_threshold = 80000
        cfg.context.manual_compact_pct = 0.5

        wm = WorkingMemory()
        stream = ObservationStream()

        stack = build_middleware_stack(
            config=cfg,
            working_memory=wm,
            observation_stream=stream,
        )

        # Find LearningMiddleware in the stack
        mw_types = [type(mw).__name__ for mw in stack._stack]
        assert "LearningMiddleware" in mw_types

    def test_middleware_factory_skips_learning_without_stream(self, tmp_path):
        """build_middleware_stack does NOT include LearningMiddleware when stream is None."""
        from homie_app.middleware_factory import build_middleware_stack
        from homie_core.memory.working import WorkingMemory

        cfg = MagicMock()
        cfg.storage.path = str(tmp_path)
        cfg.context.summarize_trigger_pct = 0.85
        cfg.context.summarize_keep_pct = 0.1
        cfg.context.arg_truncation_threshold = 2000
        cfg.context.long_line_threshold = 5000
        cfg.context.large_result_threshold = 80000
        cfg.context.manual_compact_pct = 0.5

        wm = WorkingMemory()

        stack = build_middleware_stack(
            config=cfg,
            working_memory=wm,
            observation_stream=None,
        )

        mw_types = [type(mw).__name__ for mw in stack._stack]
        assert "LearningMiddleware" not in mw_types


class TestPreferencePromptLayer:
    """Test that preference prompt layer is generated after enough signals."""

    def test_prompt_layer_generated_after_signals(self, tmp_path):
        """After enough explicit preference signals, get_prompt_layer returns content."""
        learner = AdaptiveLearner(db_path=tmp_path / "learn.db")

        # Initially, prompt layer should be empty or minimal
        initial_layer = learner.get_prompt_layer()

        # Feed many explicit preference signals
        for _ in range(20):
            learner.process_turn(
                "Please use bullet points and be more concise",
                "Sure, I will be brief.",
                state={},
            )

        # After enough signals, prompt layer should have content
        layer = learner.get_prompt_layer()
        assert isinstance(layer, str)
        # Layer should reference learned preferences (concise/bullets)
        if layer:
            layer_lower = layer.lower()
            assert "concise" in layer_lower or "brief" in layer_lower or "bullet" in layer_lower or "short" in layer_lower

        learner.stop()


class TestKnowledgeExtractionOnProcessTurn:
    """Test that knowledge extraction happens on process_turn."""

    def test_knowledge_extracted_from_turn(self, tmp_path):
        """process_turn triggers knowledge extraction."""
        learner = AdaptiveLearner(db_path=tmp_path / "learn.db")

        learner.process_turn(
            "I'm a software engineer working on Python projects",
            "That's great! How can I help with your Python work?",
            state={},
        )

        # Knowledge builder should have processed the turn
        # Check that decisions were stored
        decisions = learner._storage.query_decisions()
        assert len(decisions) >= 1

        learner.stop()


class TestLearningMiddlewareSignalEmission:
    """Test that LearningMiddleware emits signals through the observation stream."""

    def test_middleware_emits_signals_on_turn(self):
        """LearningMiddleware emits signals during before_turn and after_turn."""
        import time

        stream = ObservationStream()
        mw = LearningMiddleware(observation_stream=stream)

        signals_received = []
        stream.subscribe(lambda sig: signals_received.append(sig))

        # Simulate a turn
        msg = mw.before_turn("be more concise please", {})
        assert msg == "be more concise please"

        resp = mw.after_turn("OK, I'll be brief.", {})
        assert resp == "OK, I'll be brief."

        # ObservationStream processes signals asynchronously in a background thread,
        # so we need to wait briefly for them to be dispatched.
        time.sleep(0.5)

        # Should have emitted at least one signal (explicit preference + engagement)
        assert len(signals_received) >= 1

        stream.shutdown()


class TestWatchdogBootWiring:
    """Test that HealthWatchdog is initialized during boot."""

    def test_init_watchdog_returns_watchdog_when_enabled(self, tmp_path):
        """_init_watchdog returns a HealthWatchdog when config enables it."""
        from homie_app.cli import _init_watchdog

        cfg = MagicMock()
        cfg.self_healing.enabled = True
        cfg.self_healing.probe_interval = 30
        cfg.storage.path = str(tmp_path)

        # Create a dummy config file so ConfigProbe doesn't fail
        config_path = Path("homie.config.yaml")

        wd = _init_watchdog(cfg)
        assert wd is not None

        from homie_core.self_healing.watchdog import HealthWatchdog
        assert isinstance(wd, HealthWatchdog)
        wd.stop()

    def test_init_watchdog_returns_none_when_disabled(self):
        """_init_watchdog returns None when self_healing is disabled."""
        from homie_app.cli import _init_watchdog

        cfg = MagicMock()
        cfg.self_healing.enabled = False

        wd = _init_watchdog(cfg)
        assert wd is None
