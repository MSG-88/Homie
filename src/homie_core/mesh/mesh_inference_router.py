from __future__ import annotations
import logging
from typing import Iterator, Optional
logger = logging.getLogger(__name__)

class MeshInferenceRouter:
    def __init__(self, model_engine, mesh_client=None, qubrid_client=None, vertex_client=None, priority=None):
        self._engine = model_engine
        self._mesh_client = mesh_client
        self._qubrid = qubrid_client
        self._vertex = vertex_client
        self._priority = priority or ["local", "hub", "qubrid"]

    @property
    def active_source(self) -> str:
        for s in self._priority:
            if s == "local" and self._engine.is_loaded: return "Local"
            if s == "hub" and self._mesh_client and self._mesh_client.is_available: return "Mesh Hub"
            if s == "qubrid" and self._qubrid and getattr(self._qubrid, "is_available", False): return "Cloud (Qubrid)"
            if s == "vertex" and self._vertex and getattr(self._vertex, "is_available", False): return "Cloud (Vertex AI)"
        return "None"

    def generate(self, prompt, max_tokens=1024, temperature=0.7, stop=None, timeout=120) -> str:
        errors = []
        for s in self._priority:
            try:
                if s == "local" and self._engine.is_loaded:
                    return self._engine.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop, timeout=timeout)
                if s == "hub" and self._mesh_client and self._mesh_client.is_available:
                    return self._mesh_client.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
                if s == "qubrid" and self._qubrid and getattr(self._qubrid, "is_available", False):
                    return self._qubrid.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
                if s == "vertex" and self._vertex and getattr(self._vertex, "is_available", False):
                    return self._vertex.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
            except Exception as e:
                errors.append(f"{s}: {e}"); continue
        raise RuntimeError(f"All inference sources unavailable. Errors: {'; '.join(errors) if errors else 'none configured'}")

    def stream(self, prompt, max_tokens=1024, temperature=0.7, stop=None) -> Iterator[str]:
        errors = []
        for s in self._priority:
            try:
                if s == "local" and self._engine.is_loaded:
                    yield from self._engine.stream(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop); return
                if s == "hub" and self._mesh_client and self._mesh_client.is_available:
                    yield from self._mesh_client.stream(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop); return
                if s == "qubrid" and self._qubrid and getattr(self._qubrid, "is_available", False):
                    yield from self._qubrid.stream(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop); return
                if s == "vertex" and self._vertex and getattr(self._vertex, "is_available", False):
                    yield from self._vertex.stream(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop); return
            except Exception as e:
                errors.append(f"{s}: {e}"); continue
        raise RuntimeError(f"All inference sources unavailable. Errors: {'; '.join(errors) if errors else 'none configured'}")
