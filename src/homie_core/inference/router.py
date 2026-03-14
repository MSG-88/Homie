"""Unified inference router — local -> LAN -> Qubrid."""
from __future__ import annotations

import logging
from typing import Iterator, Optional

from homie_core.config import HomieConfig
from homie_core.model.engine import ModelEngine

logger = logging.getLogger(__name__)

_FALLBACK_BANNER = (
    "No local model found! Using Homie's intelligence until local model is setup!"
)


class InferenceRouter:
    """Routes inference requests through the priority chain."""

    def __init__(
        self,
        config: HomieConfig,
        model_engine: ModelEngine,
        qubrid_api_key: str = "",
    ):
        self._config = config
        self._engine = model_engine
        self._qubrid = None
        self._priority = config.inference.priority

        if qubrid_api_key and config.inference.qubrid.enabled:
            from homie_core.inference.qubrid import QubridClient
            self._qubrid = QubridClient(
                api_key=qubrid_api_key,
                model=config.inference.qubrid.model,
                base_url=config.inference.qubrid.base_url,
                timeout=config.inference.qubrid.timeout,
            )
            self._qubrid.check_available()

    @property
    def active_source(self) -> str:
        for source in self._priority:
            if source == "local" and self._engine.is_loaded:
                return "Local"
            if source == "lan":
                continue  # LAN support added later
            if source == "qubrid" and self._qubrid and self._qubrid.is_available:
                return "Homie Intelligence (Cloud)"
        return "None"

    @property
    def fallback_banner(self) -> Optional[str]:
        if self._engine.is_loaded:
            return None
        if self._qubrid and self._qubrid.is_available:
            return _FALLBACK_BANNER
        return None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        timeout: int = 120,
    ) -> str:
        errors: list[str] = []
        for source in self._priority:
            try:
                if source == "local" and self._engine.is_loaded:
                    return self._engine.generate(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop, timeout=timeout,
                    )
                if source == "lan":
                    continue
                if source == "qubrid" and self._qubrid and self._qubrid.is_available:
                    return self._qubrid.generate(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
            except Exception as e:
                logger.warning("Inference source '%s' failed: %s", source, e)
                errors.append(f"{source}: {e}")
                continue

        raise RuntimeError(
            "All inference sources unavailable. "
            "Please check your connection or download a local model. "
            f"Errors: {'; '.join(errors) if errors else 'no sources configured'}"
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        errors: list[str] = []
        for source in self._priority:
            try:
                if source == "local" and self._engine.is_loaded:
                    yield from self._engine.stream(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                    return
                if source == "lan":
                    continue
                if source == "qubrid" and self._qubrid and self._qubrid.is_available:
                    yield from self._qubrid.stream(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                    return
            except Exception as e:
                logger.warning("Inference source '%s' failed: %s", source, e)
                errors.append(f"{source}: {e}")
                continue

        raise RuntimeError(
            "All inference sources unavailable. "
            f"Errors: {'; '.join(errors) if errors else 'no sources configured'}"
        )
