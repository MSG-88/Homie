"""ObservationStream — central signal collector and dispatcher."""

import logging
import queue
import threading
from collections import deque
from typing import Callable, Optional

from .signals import LearningSignal, SignalCategory

logger = logging.getLogger(__name__)

SignalCallback = Callable[[LearningSignal], None]


class ObservationStream:
    """Collects learning signals and dispatches them to subscribers."""

    def __init__(self, history_size: int = 100) -> None:
        self._subscribers: list[tuple[SignalCallback, Optional[SignalCategory]]] = []
        self._lock = threading.Lock()
        self._queue: queue.Queue[LearningSignal | None] = queue.Queue()
        self._history: deque[LearningSignal] = deque(maxlen=history_size)
        self._running = True
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

    @property
    def recent_signals(self) -> list[LearningSignal]:
        """Return recent signal history."""
        return list(self._history)

    def subscribe(
        self,
        callback: SignalCallback,
        category: Optional[SignalCategory] = None,
    ) -> None:
        """Subscribe to signals. Optionally filter by category."""
        with self._lock:
            self._subscribers.append((callback, category))

    def emit(self, signal: LearningSignal) -> None:
        """Emit a signal to all matching subscribers."""
        if self._running:
            self._queue.put(signal)

    def shutdown(self) -> None:
        """Stop the observation stream."""
        self._running = False
        self._queue.put(None)
        self._worker.join(timeout=2.0)

    def _process_loop(self) -> None:
        while self._running:
            try:
                signal = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if signal is None:
                break

            self._history.append(signal)

            with self._lock:
                subscribers = list(self._subscribers)

            for callback, category_filter in subscribers:
                if category_filter is not None and signal.category != category_filter:
                    continue
                try:
                    callback(signal)
                except Exception:
                    logger.exception("Signal subscriber failed")
