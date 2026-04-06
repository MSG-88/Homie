"""CLAP Audio Classification Plugin for Homie.

Uses the laion/clap-htsat-fused model to classify audio events and provide
sound understanding capabilities. Runs inference locally via the
transformers library â€” no network calls after initial model download.

Typical use-cases:
  - Ambient sound detection (doorbell, glass breaking, dog barking)
  - Voice-activity vs. noise discrimination before STT
  - Custom audio-event triggers for Homie automations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_ID = "laion/clap-htsat-fused"

# Default candidate labels used when the caller does not supply any.
DEFAULT_LABELS: List[str] = [
    "speech",
    "music",
    "silence",
    "doorbell",
    "dog barking",
    "glass breaking",
    "alarm",
    "knocking",
    "footsteps",
    "typing",
]


@dataclass
class ClassificationResult:
    """Single classification output."""

    label: str
    score: float


@dataclass
class CLAPPluginConfig:
    """Runtime configuration for the CLAP plugin."""

    model_id: str = MODEL_ID
    cache_dir: Optional[str] = None
    device: str = "cpu"
    default_labels: List[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    top_k: int = 3
    score_threshold: float = 0.1


class CLAPAudioClassifierPlugin:
    """Homie plugin that wraps laion/clap-htsat-fused for zero-shot audio classification.

    The model is loaded lazily on first inference so activation is cheap.
    """

    name: str = "clap_audio_classifier"
    version: str = "0.1.0"

    def __init__(self, config: Optional[CLAPPluginConfig] = None) -> None:
        self.config = config or CLAPPluginConfig()
        self._processor: Any = None
        self._model: Any = None
        self._active: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Activate the plugin. Model loading is deferred to first use."""
        if self._active:
            return
        logger.info("Activating %s (model will load on first classify call)", self.name)
        self._active = True

    def deactivate(self) -> None:
        """Release model resources."""
        self._processor = None
        self._model = None
        self._active = False
        logger.info("Deactivated %s", self.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy-load model and processor on first call."""
        if self._model is not None:
            return

        try:
            from transformers import ClapModel, ClapProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for the CLAP plugin. "
                "Install with: pip install transformers torch"
            ) from exc

        cache_dir = self.config.cache_dir
        logger.info("Loading CLAP model %s (device=%s) â€¦", self.config.model_id, self.config.device)
        self._processor = ClapProcessor.from_pretrained(
            self.config.model_id, cache_dir=cache_dir
        )
        self._model = ClapModel.from_pretrained(
            self.config.model_id, cache_dir=cache_dir
        )
        self._model.to(self.config.device)  # type: ignore[union-attr]
        self._model.eval()  # type: ignore[union-attr]
        logger.info("CLAP model loaded.")

    @staticmethod
    def _load_audio(path: Path, target_sr: int = 48_000) -> Any:
        """Load an audio file and resample to *target_sr* Hz.

        Returns a 1-D numpy array of float32 samples.
        """
        try:
            import librosa  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "librosa is required to load audio files. "
                "Install with: pip install librosa"
            ) from exc

        waveform, _ = librosa.load(str(path), sr=target_sr, mono=True)
        return waveform

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        audio_path: str | Path,
        candidate_labels: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[ClassificationResult]:
        """Classify an audio file against *candidate_labels* using zero-shot CLAP.

        Parameters
        ----------
        audio_path:
            Path to a local audio file (wav, mp3, flac, etc.).
        candidate_labels:
            Text descriptions of possible sounds. Falls back to
            ``config.default_labels`` when *None*.
        top_k:
            Return at most this many results, sorted by descending score.

        Returns
        -------
        List of ``ClassificationResult`` ordered by score (highest first).
        """
        if not self._active:
            raise RuntimeError("Plugin is not active. Call activate() first.")

        import torch  # type: ignore

        self._ensure_model()

        labels = list(candidate_labels or self.config.default_labels)
        k = top_k if top_k is not None else self.config.top_k
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform = self._load_audio(audio_path)

        inputs = self._processor(
            text=labels,
            audios=[waveform],
            return_tensors="pt",
            padding=True,
            sampling_rate=48_000,
        )
        inputs = {key: val.to(self.config.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)  # type: ignore[misc]

        logits = outputs.logits_per_audio[0]  # shape: (num_labels,)
        probs = logits.softmax(dim=-1).cpu().tolist()

        results = [
            ClassificationResult(label=lbl, score=round(score, 4))
            for lbl, score in zip(labels, probs)
            if score >= self.config.score_threshold
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def classify_batch(
        self,
        audio_paths: Sequence[str | Path],
        candidate_labels: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, List[ClassificationResult]]:
        """Convenience wrapper: classify multiple files, keyed by filename."""
        return {
            Path(p).name: self.classify(p, candidate_labels=candidate_labels, top_k=top_k)
            for p in audio_paths
        }


# ------------------------------------------------------------------
# Module-level registration helper (Homie convention)
# ------------------------------------------------------------------

def register(homie_config: Optional[Dict[str, Any]] = None) -> CLAPAudioClassifierPlugin:
    """Create, configure, and activate a CLAPAudioClassifierPlugin.

    Parameters
    ----------
    homie_config:
        Optional dict (or ``HomieConfig.raw``) with a ``plugins.clap``
        section. Recognised keys mirror ``CLAPPluginConfig`` fields::

            plugins:
              clap:
                device: cpu          # or cuda
                cache_dir: ~/.cache/homie/models
                top_k: 5
                score_threshold: 0.15
                default_labels:
                  - speech
                  - music
    """
    plugin_cfg = CLAPPluginConfig()

    if homie_config:
        clap_section = (homie_config.get("plugins") or {}).get("clap", {})
        if clap_section:
            if "device" in clap_section:
                plugin_cfg.device = str(clap_section["device"])
            if "cache_dir" in clap_section:
                plugin_cfg.cache_dir = str(clap_section["cache_dir"])
            if "top_k" in clap_section:
                plugin_cfg.top_k = int(clap_section["top_k"])
            if "score_threshold" in clap_section:
                plugin_cfg.score_threshold = float(clap_section["score_threshold"])
            if "default_labels" in clap_section:
                plugin_cfg.default_labels = list(clap_section["default_labels"])
            if "model_id" in clap_section:
                plugin_cfg.model_id = str(clap_section["model_id"])

    plugin = CLAPAudioClassifierPlugin(config=plugin_cfg)
    plugin.activate()
    return plugin
