"""Homie plugin for audio-text matching and classification using LAION CLAP.

Uses the laion/clap-htsat-fused model to perform zero-shot audio classification
and audio-text similarity scoring entirely on-device. This enables Homie to
classify ambient sounds, match voice commands to intent labels, and tag audio
snippets without any cloud API calls.

Requires: transformers, torch, librosa
Install:  pip install transformers torch librosa
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "laion/clap-htsat-fused"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_MAX_DURATION_SEC = 30

# Sensible default labels for household / assistant audio classification
DEFAULT_LABELS: List[str] = [
    "speech",
    "music",
    "doorbell",
    "alarm",
    "dog barking",
    "glass breaking",
    "silence",
    "keyboard typing",
    "applause",
    "coughing",
]


@dataclass
class ClassificationResult:
    """Single classification outcome."""

    label: str
    score: float


@dataclass
class AudioMatchResult:
    """Result of matching audio against candidate text labels."""

    source: str
    results: List[ClassificationResult] = field(default_factory=list)
    top_label: str = ""
    top_score: float = 0.0


class ClapAudioClassifierPlugin:
    """Zero-shot audio classification and audio-text similarity plugin for Homie.

    Loads the CLAP model locally and exposes helpers for:
    - Classifying an audio file against a set of text labels.
    - Computing similarity scores between audio and free-form text descriptions.
    - Providing a top-1 prediction suitable for Homie automation triggers.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        labels: Optional[List[str]] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_duration_sec: int = DEFAULT_MAX_DURATION_SEC,
    ) -> None:
        self.model_id = model_id
        self.labels = labels or list(DEFAULT_LABELS)
        self._device = device
        self._cache_dir = cache_dir
        self.max_duration_sec = max_duration_sec
        self._model: Any = None
        self._processor: Any = None
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Load the CLAP model and processor into memory."""
        if self._active:
            logger.debug("ClapAudioClassifierPlugin already active")
            return

        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for ClapAudioClassifierPlugin. "
                "Install them with: pip install transformers torch librosa"
            ) from exc

        resolved_device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading CLAP model '%s' on %s (cache=%s)",
            self.model_id,
            resolved_device,
            self._cache_dir or "default",
        )

        self._processor = ClapProcessor.from_pretrained(
            self.model_id,
            cache_dir=self._cache_dir,
        )
        self._model = ClapModel.from_pretrained(
            self.model_id,
            cache_dir=self._cache_dir,
        ).to(resolved_device)
        self._model.eval()
        self._device = resolved_device
        self._active = True
        logger.info("ClapAudioClassifierPlugin activated")

    def deactivate(self) -> None:
        """Release model resources."""
        self._model = None
        self._processor = None
        self._active = False
        logger.info("ClapAudioClassifierPlugin deactivated")

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, audio_path: str) -> "numpy.ndarray":
        """Load and resample an audio file to the expected sample rate."""
        import librosa
        import numpy as np

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sr = librosa.load(
            str(path),
            sr=DEFAULT_SAMPLE_RATE,
            mono=True,
            duration=self.max_duration_sec,
        )
        return waveform.astype(np.float32)

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    def classify(
        self,
        audio_path: str,
        labels: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> AudioMatchResult:
        """Classify an audio file against candidate text labels.

        Parameters
        ----------
        audio_path:
            Path to a local audio file (wav, mp3, flac, ogg, etc.).
        labels:
            Text labels to score against. Falls back to ``self.labels``.
        top_k:
            Number of top results to return.

        Returns
        -------
        AudioMatchResult with ranked labels and scores.
        """
        if not self._active:
            raise RuntimeError("Plugin is not activated. Call activate() first.")

        import torch

        candidate_labels = labels or self.labels
        if not candidate_labels:
            raise ValueError("At least one candidate label is required")

        waveform = self._load_audio(audio_path)

        inputs = self._processor(
            text=candidate_labels,
            audios=[waveform],
            return_tensors="pt",
            padding=True,
            sampling_rate=DEFAULT_SAMPLE_RATE,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_audio  # (1, num_labels)
            probs = logits.softmax(dim=-1).squeeze(0).cpu().tolist()

        scored = [
            ClassificationResult(label=lbl, score=round(sc, 6))
            for lbl, sc in zip(candidate_labels, probs)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        scored = scored[:top_k]

        return AudioMatchResult(
            source=audio_path,
            results=scored,
            top_label=scored[0].label if scored else "",
            top_score=scored[0].score if scored else 0.0,
        )

    def similarity(
        self,
        audio_path: str,
        descriptions: List[str],
    ) -> List[Tuple[str, float]]:
        """Compute cosine similarity between one audio clip and several text descriptions.

        Returns a list of (description, similarity_score) sorted descending.
        """
        if not self._active:
            raise RuntimeError("Plugin is not activated. Call activate() first.")

        import torch

        waveform = self._load_audio(audio_path)

        audio_inputs = self._processor(
            audios=[waveform],
            return_tensors="pt",
            sampling_rate=DEFAULT_SAMPLE_RATE,
        )
        audio_inputs = {k: v.to(self._device) for k, v in audio_inputs.items()}

        text_inputs = self._processor(
            text=descriptions,
            return_tensors="pt",
            padding=True,
        )
        text_inputs = {k: v.to(self._device) for k, v in text_inputs.items()}

        with torch.no_grad():
            audio_embeds = self._model.get_audio_features(**audio_inputs)
            text_embeds = self._model.get_text_features(**text_inputs)

            audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            sims = (audio_embeds @ text_embeds.T).squeeze(0).cpu().tolist()

        if isinstance(sims, float):
            sims = [sims]

        pairs = sorted(
            zip(descriptions, sims),
            key=lambda p: p[1],
            reverse=True,
        )
        return [(desc, round(score, 6)) for desc, score in pairs]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_speech(self, audio_path: str, threshold: float = 0.4) -> bool:
        """Quick check: does this audio contain speech?"""
        result = self.classify(audio_path, labels=["speech", "not speech", "silence"])
        return result.top_label == "speech" and result.top_score >= threshold

    def detect_alert_sound(
        self,
        audio_path: str,
        alert_labels: Optional[List[str]] = None,
        threshold: float = 0.3,
    ) -> Optional[ClassificationResult]:
        """Detect whether audio contains an alert/notification sound.

        Returns the matching alert label if above threshold, else None.
        """
        labels = alert_labels or [
            "doorbell",
            "alarm",
            "smoke detector",
            "glass breaking",
            "siren",
            "phone ringing",
        ]
        result = self.classify(audio_path, labels=labels, top_k=1)
        if result.top_score >= threshold:
            return result.results[0]
        return None

    def to_dict(self, result: AudioMatchResult) -> Dict[str, Any]:
        """Serialize an AudioMatchResult to a plain dict for logging or JSON."""
        return {
            "source": result.source,
            "top_label": result.top_label,
            "top_score": result.top_score,
            "results": [
                {"label": r.label, "score": r.score} for r in result.results
            ],
        }


# ------------------------------------------------------------------
# Module-level register / convenience
# ------------------------------------------------------------------

_default_instance: Optional[ClapAudioClassifierPlugin] = None


def register(config: Optional[Dict[str, Any]] = None) -> ClapAudioClassifierPlugin:
    """Create, activate, and return the default plugin instance.

    Parameters
    ----------
    config:
        Optional dict with keys ``model_id``, ``labels``, ``device``,
        ``cache_dir``, ``max_duration_sec``.
    """
    global _default_instance
    cfg = config or {}
    plugin = ClapAudioClassifierPlugin(
        model_id=cfg.get("model_id", DEFAULT_MODEL_ID),
        labels=cfg.get("labels"),
        device=cfg.get("device"),
        cache_dir=cfg.get("cache_dir"),
        max_duration_sec=cfg.get("max_duration_sec", DEFAULT_MAX_DURATION_SEC),
    )
    plugin.activate()
    _default_instance = plugin
    return plugin


def get_instance() -> ClapAudioClassifierPlugin:
    """Return the default plugin instance, raising if not registered."""
    if _default_instance is None:
        raise RuntimeError(
            "ClapAudioClassifierPlugin not registered. Call register() first."
        )
    return _default_instance
