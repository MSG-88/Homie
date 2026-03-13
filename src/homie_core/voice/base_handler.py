from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)
SENTINEL = b"END"


class BaseHandler(ABC):
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        name: str | None = None,
    ) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._name = name or self.__class__.__name__
        self._times: list[float] = []

    @abstractmethod
    def process(self, item: Any) -> Any:
        """Process single item. Return None to skip output."""

    def run(self) -> None:
        logger.debug("%s: started", self._name)
        while not self._stop_event.is_set():
            try:
                item = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is SENTINEL or item == SENTINEL:
                break
            start = time.perf_counter()
            try:
                result = self.process(item)
            except Exception:
                logger.exception("%s: error processing", self._name)
                continue
            self._times.append(time.perf_counter() - start)
            if result is not None:
                self._output_queue.put(result)
        logger.debug("%s: stopped", self._name)

    @property
    def last_time(self) -> float:
        return self._times[-1] if self._times else 0.0

    def send_sentinel(self) -> None:
        self._input_queue.put(SENTINEL)
