"""Hub Inference Server — processes inference requests from the priority queue."""
from __future__ import annotations

import logging
from typing import Optional

from homie_core.mesh.inference_queue import InferenceQueue, InferenceRequest

logger = logging.getLogger(__name__)


class InferenceServer:
    """Wraps a model engine and exposes synchronous + queued inference APIs."""

    def __init__(self, model_engine, max_concurrent: int = 2) -> None:
        self._engine = model_engine
        self._queue = InferenceQueue(max_concurrent=max_concurrent)

    # ------------------------------------------------------------------
    # Direct / synchronous inference
    # ------------------------------------------------------------------

    def process(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        timeout: int = 120,
    ) -> str:
        """Run inference immediately, bypassing the queue.

        Raises:
            RuntimeError: if no model is loaded on the engine.
        """
        if not self._engine.is_loaded:
            raise RuntimeError("No model loaded on Hub.")
        return self._engine.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Queued inference
    # ------------------------------------------------------------------

    def submit(self, request: InferenceRequest) -> None:
        """Add a request to the priority queue."""
        self._queue.submit(request)

    def process_next(self, timeout: float = 5.0) -> Optional[dict]:
        """Dequeue and process the highest-priority pending request.

        Returns:
            A result dict with keys ``request_id``, ``node_id``, ``content``,
            and ``error``, or *None* if the queue was empty within *timeout*.
        """
        req = self._queue.get(timeout=timeout)
        if req is None:
            return None

        self._queue.mark_active(req.request_id)
        try:
            content = self.process(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stop=req.stop,
            )
            return {
                "request_id": req.request_id,
                "node_id": req.node_id,
                "content": content,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Inference error for %s: %s", req.request_id, exc)
            return {
                "request_id": req.request_id,
                "node_id": req.node_id,
                "content": "",
                "error": str(exc),
            }
        finally:
            self._queue.mark_done(req.request_id)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def queue_stats(self) -> dict:
        """Return current queue statistics."""
        return self._queue.stats()
