"""Homie plugin for audio classification and voice-command understanding using CLAP.

Uses the laion/clap-htsat-fused model to classify audio segments against
free-form text labels.  This enables Homie to understand ambient sounds
(e.g. doorbell, alarm, glass breaking) and to match spoken utterances to
command intents without a fixed grammar.

The model runs entirely on-device via the Hugging Face transformers library.
No network calls are made after the initial (opt-in) model download.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "laion/clap-htsat-fused"
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_LABELS = [
    "voice command",
    "doorbell",
    "alarm",
    "glass breaking",
    "knock",
    "silence",
    "music",
    "speech",
]


@dataclass
class CLAPConfig:
    """Runtime configuration for the CLAP plugin."""

    model_id: str = DEFAULT_MODEL_ID
    cache_dir: Optional[str] = None
    device: str = "cpu"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    default_labels: List[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    confidence_threshold: float = 0.3


@dataclass
class ClassificationResult:
    """Single classification output."""

    label: str
    score: float


class CLAPAudioPlugin:
    """Audio classification and voice-command understanding via CLAP.

    Provides zero-shot audio classification: given an audio waveform and a set
    of candidate text labels the plugin returns ranked scores indicating how
    well the audio matches each label.
    """

    name: str = "clap_audio"

    def __init__(self, config: Optional[CLAPConfig] = None) -> None:
        self._config = config or CLAPConfig()
        self._model: Any = None
        self._processor: Any = None
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Load the CLAP model into memory.

        The first call may trigger a one-time download from Hugging Face Hub
        if the model is not already cached locally.  All subsequent inference
        is fully offline.
        """
        if self._active:
            logger.debug("CLAPAudioPlugin already active")
            return

        try:
            from transformers import ClapModel, ClapProcessor  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' and 'torch' packages are required for the "
                "CLAP plugin.  Install them with:  pip install transformers torch"
            ) from exc

        logger.info(
            "Loading CLAP model '%s' on %s â€¦",
            self._config.model_id,
            self._config.device,
        )

        cache_kwargs: Dict[str, Any] = {}
        if self._config.cache_dir:
            cache_kwargs["cache_dir"] = self._config.cache_dir

        self._processor = ClapProcessor.from_pretrained(
            self._config.model_id, **cache_kwargs
        )
        self._model = ClapModel.from_pretrained(
            self._config.model_id, **cache_kwargs
        ).to(self._config.device)
        self._model.eval()
        self._active = True
        logger.info("CLAPAudioPlugin activated")

    def deactivate(self) -> None:
        """Unload the model and free memory."""
        self._model = None
        self._processor = None
        self._active = False
        logger.info("CLAPAudioPlugin deactivated")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        audio: np.ndarray,
        labels: Optional[Sequence[str]] = None,
        *,
        top_k: int = 5,
    ) -> List[ClassificationResult]:
        """Classify *audio* against free-form text *labels*.

        Parameters
        ----------
        audio:
            1-D float32 waveform at ``self._config.sample_rate`` Hz.
        labels:
            Candidate text descriptions.  Falls back to
            ``CLAPConfig.default_labels`` when *None*.
        top_k:
            Maximum number of results to return, ordered by descending score.

        Returns
        -------
        list[ClassificationResult]
            Ranked list of (label, score) pairs.
        """
        if not self._active:
            raise RuntimeError("Plugin is not active. Call activate() first.")

        import torch  # type: ignore[import-untyped]

        labels = list(labels or self._config.default_labels)

        inputs = self._processor(
            text=labels,
            audios=[audio],
            return_tensors="pt",
            sampling_rate=self._config.sample_rate,
            padding=True,
        ).to(self._config.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # outputs.logits_per_audio has shape (1, num_labels)
        logits = outputs.logits_per_audio[0]
        probs = logits.softmax(dim=-1).cpu().numpy()

        scored = [
            ClassificationResult(label=lbl, score=float(prob))
            for lbl, prob in zip(labels, probs)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def match_command(
        self,
        audio: np.ndarray,
        commands: Sequence[str],
    ) -> Optional[Tuple[str, float]]:
        """Return the best-matching command if it exceeds the confidence threshold.

        Parameters
        ----------
        audio:
            1-D float32 waveform.
        commands:
            List of natural-language command descriptions
            (e.g. ``["turn on the lights", "play music", "set a timer"]``).

        Returns
        -------
        tuple[str, float] | None
            The matched command and its confidence, or *None* if no command
            exceeds ``CLAPConfig.confidence_threshold``.
        """
        results = self.classify(audio, labels=commands, top_k=1)
        if results and results[0].score >= self._config.confidence_threshold:
            return results[0].label, results[0].score
        return None

    # ------------------------------------------------------------------
    # Homie integration helpers
    # ------------------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        """Serialise plugin state for the Homie dashboard / ledger."""
        return {
            "plugin": self.name,
            "active": self._active,
            "model_id": self._config.model_id,
            "device": self._config.device,
            "sample_rate": self._config.sample_rate,
            "confidence_threshold": self._config.confidence_threshold,
        }


# ------------------------------------------------------------------
# Module-level convenience (register / activate / deactivate)
# ------------------------------------------------------------------

_instance: Optional[CLAPAudioPlugin] = None


def register(config: Optional[CLAPConfig] = None) -> CLAPAudioPlugin:
    """Create (or return) the singleton plugin instance."""
    global _instance
    if _instance is None:
        _instance = CLAPAudioPlugin(config)
    return _instance


def activate(config: Optional[CLAPConfig] = None) -> CLAPAudioPlugin:
    """Register *and* activate in one call."""
    plugin = register(config)
    plugin.activate()
    return plugin


def deactivate() -> None:
    """Deactivate the singleton, if any."""
    if _instance is not None:
        _instance.deactivate()


__all__ = [
    "CLAPConfig",
    "ClassificationResult",
    "CLAPAudioPlugin",
    "register",
    "activate",
    "deactivate",
]
