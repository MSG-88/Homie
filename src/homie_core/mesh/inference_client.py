from __future__ import annotations
from typing import Callable, Iterator, Optional


class MeshInferenceClient:
    def __init__(self, node_id: str, hub_node_id: str, request_handler: Optional[Callable] = None):
        self._node_id = node_id
        self._hub_node_id = hub_node_id
        self._handler = request_handler

    @property
    def is_available(self) -> bool:
        return self._handler is not None

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7,
                 stop: Optional[list[str]] = None, timeout: int = 120) -> str:
        if not self._handler:
            raise RuntimeError("Mesh inference not available — no hub connected.")
        return self._handler(prompt, max_tokens, temperature, stop)

    def stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7,
               stop: Optional[list[str]] = None) -> Iterator[str]:
        yield self.generate(prompt, max_tokens, temperature, stop)
