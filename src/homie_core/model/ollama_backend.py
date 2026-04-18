"""Ollama backend — inference via the local Ollama server.

Uses the Ollama REST API (http://localhost:11434) for generate/stream.
This replaces the need for llama-cpp-python or HuggingFace backends
when models are managed via Ollama.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://localhost:11434"


class OllamaBackend:
    """Inference backend using the Ollama REST API."""

    def __init__(self):
        self._model: str = ""
        self._base_url: str = _DEFAULT_URL
        self._loaded: bool = False

    def load(self, model: str, base_url: str = _DEFAULT_URL) -> None:
        """Register the model name. Ollama handles actual loading on first inference."""
        self._model = model
        self._base_url = base_url
        # Verify Ollama is running and model exists
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                model_names = [m["name"] for m in data.get("models", [])]
                if model not in model_names:
                    # Try without tag
                    base_name = model.split(":")[0]
                    matches = [m for m in model_names if m.startswith(base_name)]
                    if not matches:
                        logger.warning("Model %s not found in Ollama. Available: %s",
                                      model, model_names[:5])
                self._loaded = True
                logger.info("Ollama backend ready: model=%s", model)
        except Exception as e:
            logger.warning("Ollama not reachable at %s: %s", self._base_url, e)
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        timeout: int = 120,
    ) -> str:
        """Generate a response via Ollama API."""
        if not self._loaded:
            raise RuntimeError("Ollama backend not loaded")

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            return result.get("message", {}).get("content", "")

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """Stream tokens via Ollama API."""
        if not self._loaded:
            raise RuntimeError("Ollama backend not loaded")

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break

    def unload(self) -> None:
        """Unload the model from Ollama VRAM."""
        if self._model:
            try:
                data = json.dumps({"model": self._model, "keep_alive": 0}).encode()
                req = urllib.request.Request(
                    f"{self._base_url}/api/generate",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=10)
            except Exception:
                pass
        self._loaded = False
        self._model = ""
