"""KnowledgeBuilder — coordinates conversation mining, project tracking, and profiling."""

import logging
from typing import Callable, Optional

from ..observation.signals import LearningSignal
from ..storage import LearningStorage
from .behavioral_profiler import BehavioralProfiler
from .conversation_miner import ConversationMiner
from .project_tracker import ProjectTracker

logger = logging.getLogger(__name__)


class KnowledgeBuilder:
    """Coordinates all knowledge-building engines."""

    def __init__(
        self,
        storage: LearningStorage,
        inference_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._storage = storage
        self.miner = ConversationMiner(storage=storage, inference_fn=inference_fn)
        self.project_tracker = ProjectTracker(storage=storage)
        self.profiler = BehavioralProfiler()

    def on_signal(self, signal: LearningSignal) -> None:
        """Process knowledge-related signals."""
        data = signal.data
        # Feed behavioral observations
        if "hour" in data:
            hour = data["hour"]
            for key in ("app", "activity", "topic"):
                if key in data:
                    self.profiler.record_observation(hour, key, data[key])

    def process_turn(self, user_message: str, response: str) -> list[str]:
        """Process a conversation turn for knowledge extraction."""
        return self.miner.process_turn(user_message, response)

    def get_work_hours(self) -> list[int]:
        """Get detected work hours."""
        return self.profiler.get_work_hours()
