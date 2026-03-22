"""AdaptiveLearner — central coordinator for the adaptive learning engine."""

import logging
from pathlib import Path
from typing import Optional

from .knowledge.builder import KnowledgeBuilder
from .observation.signals import LearningSignal, SignalCategory, SignalType
from .observation.stream import ObservationStream
from .performance.optimizer import PerformanceOptimizer
from .preference.engine import PreferenceEngine
from .storage import LearningStorage

logger = logging.getLogger(__name__)


class AdaptiveLearner:
    """Central coordinator for adaptive learning — preference, performance, and knowledge."""

    def __init__(
        self,
        db_path: Path | str,
        learning_rate_explicit: float = 0.3,
        learning_rate_implicit: float = 0.05,
        cache_max_entries: int = 500,
        cache_ttl: float = 86400.0,
        graph_db_path: Optional[Path | str] = None,
    ) -> None:
        self._storage = LearningStorage(db_path=db_path)
        self._storage.initialize()

        self.observation_stream = ObservationStream()
        self.preference_engine = PreferenceEngine(
            storage=self._storage,
            learning_rate_explicit=learning_rate_explicit,
            learning_rate_implicit=learning_rate_implicit,
        )
        self.performance_optimizer = PerformanceOptimizer(
            storage=self._storage,
            cache_max_entries=cache_max_entries,
            cache_ttl=cache_ttl,
        )
        graph_path = Path(db_path).parent / "knowledge_graph.db" if graph_db_path is None else graph_db_path
        self.knowledge_builder = KnowledgeBuilder(
            storage=self._storage,
            graph_db_path=graph_path,
        )

        # Wire up observation subscriptions
        self.observation_stream.subscribe(
            self.preference_engine.on_signal,
            category=SignalCategory.PREFERENCE,
        )
        self.observation_stream.subscribe(
            self.preference_engine.on_signal,
            category=SignalCategory.ENGAGEMENT,
        )
        self.observation_stream.subscribe(
            self.performance_optimizer.on_signal,
            category=SignalCategory.PERFORMANCE,
        )
        self.observation_stream.subscribe(
            self.knowledge_builder.on_signal,
            category=SignalCategory.CONTEXT,
        )

    def process_turn(self, user_message: str, response: str, state: Optional[dict] = None) -> None:
        """Process a conversation turn — feeds all engines."""
        state = state or {}

        # Knowledge extraction
        self.knowledge_builder.process_turn(user_message, response)

        # Emit turn-level signals through the observation stream
        # (LearningMiddleware handles the detailed signal emission,
        #  but we also do direct processing for explicit preferences)
        msg_lower = user_message.lower()

        # Check for explicit format preferences — route to global layer
        # so they apply regardless of current domain context
        if "bullet" in msg_lower:
            self.preference_engine.on_signal(LearningSignal(
                signal_type=SignalType.EXPLICIT,
                category=SignalCategory.PREFERENCE,
                source="user_message",
                data={"dimension": "format", "value": "bullets"},
                context={},
            ))
        if "concise" in msg_lower or "shorter" in msg_lower or "brief" in msg_lower:
            self.preference_engine.on_signal(LearningSignal(
                signal_type=SignalType.EXPLICIT,
                category=SignalCategory.PREFERENCE,
                source="user_message",
                data={"dimension": "verbosity", "direction": "decrease"},
                context={},
            ))

    def get_prompt_layer(
        self,
        domain: Optional[str] = None,
        project: Optional[str] = None,
        hour: Optional[int] = None,
    ) -> str:
        """Get the preference prompt layer for system prompt injection."""
        return self.preference_engine.get_prompt_layer(domain=domain, project=project, hour=hour)

    def get_cached_response(self, query: str, context_hash: Optional[str] = None) -> Optional[str]:
        """Check the response cache."""
        return self.performance_optimizer.get_cached_response(query, context_hash)

    def start(self) -> None:
        """Start the adaptive learning engine."""
        logger.info("AdaptiveLearner started")

    def stop(self) -> None:
        """Stop the adaptive learning engine."""
        self.observation_stream.shutdown()
        self._storage.close()
        logger.info("AdaptiveLearner stopped")
