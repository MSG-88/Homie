from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from homie_core.model.registry import ModelEntry


class ModelEngine:
    def __init__(self):
        self._backend = None
        self._loaded = False
        self._current_model: Optional[ModelEntry] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def current_model(self) -> Optional[ModelEntry]:
        return self._current_model

    def load(self, entry: ModelEntry, n_ctx: int = 16384, n_gpu_layers: int = -1, **kwargs) -> None:
        if entry.format == "gguf":
            from homie_core.model.gguf_backend import GGUFBackend
            backend = GGUFBackend()
            backend.load(entry.path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
            self._backend = backend
        elif entry.format == "safetensors":
            from homie_core.model.transformers_backend import TransformersBackend
            backend = TransformersBackend()
            backend.load(entry.path)
            self._backend = backend
        elif entry.format == "hf":
            from homie_core.model.hf_backend import HFBackend
            backend = HFBackend()
            backend.load(
                model_id=entry.path,
                api_key=kwargs.get("api_key", ""),
            )
            self._backend = backend
        elif entry.format == "cloud":
            from homie_core.model.cloud_backend import CloudBackend
            backend = CloudBackend()
            backend.load(
                model_id=entry.path,
                api_key=kwargs.get("api_key", ""),
                base_url=kwargs.get("base_url", "https://api.openai.com/v1"),
            )
            self._backend = backend
        elif entry.format == "ollama":
            from homie_core.model.ollama_backend import OllamaBackend
            backend = OllamaBackend()
            backend.load(
                model=entry.path,
                base_url=kwargs.get("base_url", "http://localhost:11434"),
            )
            self._backend = backend
        elif entry.format == "hf_local":
            from homie_core.model.hf_local_backend import HFLocalBackend
            backend = HFLocalBackend()
            backend.load(
                model_id=entry.path,
                quantize_4bit=kwargs.get("quantize_4bit", True),
            )
            self._backend = backend
        else:
            raise ValueError(f"Unsupported format: {entry.format}")
        self._loaded = True
        self._current_model = entry

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None, timeout: int = 120) -> str:
        """Generate a response. Raises TimeoutError if no response within timeout seconds."""
        import threading

        if not self._loaded:
            raise RuntimeError("No model loaded")

        result = [None]
        error = [None]

        def _run():
            try:
                result[0] = self._backend.generate(
                    prompt, max_tokens=max_tokens,
                    temperature=temperature, stop=stop,
                )
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Model did not respond within {timeout}s. "
                f"The model may be too large for your hardware, or the server may be unresponsive. "
                f"Try a smaller model or check 'homie model list'."
            )
        if error[0]:
            raise error[0]
        return result[0] or ""

    def stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> Iterator[str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        return self._backend.stream(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)

    def unload(self) -> None:
        if self._backend:
            self._backend.unload()
        self._backend = None
        self._loaded = False
        self._current_model = None
