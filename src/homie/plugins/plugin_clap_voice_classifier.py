"""CLAP Voice Command Classification Plugin for Homie.

Uses the LAION CLAP (Contrastive Language-Audio Pretraining) model to classify
audio input against a configurable set of text labels. This enables Homie to
understand voice commands by computing audio-text similarity scores locally,
without any network calls.

The plugin loads a GGUF-quantised or safetensors CLAP checkpoint from disk,
encodes incoming audio and candidate command labels into a shared embedding
space, and returns ranked similarity scores so Homie can dispatch the most
likely intent.

Requires: transformers, torch, librosa (all local, no network at runtime).
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
    "check status",
    "run command",
    "stop",
    "help",
    "list machines",
    "schedule task",
    "cancel",
]


@dataclass
class ClassificationResult:
    """Ranked voice-command classification output."""

    label: str
    score: float


@dataclass
class ClapVoiceClassifierConfig:
    """Plugin configuration."""

    model_path: Optional[str] = None
    model_id: str = DEFAULT_MODEL_ID
    labels: List[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    sample_rate: int = DEFAULT_SAMPLE_RATE
    score_threshold: float = 0.3
    device: str = "cpu"


class ClapVoiceClassifier:
    """Classifies audio waveforms against text command labels using CLAP.

    Integrates with Homie's local-first architecture: the model is loaded from
    disk (or Hugging Face cache on first use) and all inference runs on-device.
    """

    def __init__(self, config: Optional[ClapVoiceClassifierConfig] = None) -> None:
        self.config = config or ClapVoiceClassifierConfig()
        self._model: Any = None
        self._processor: Any = None
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Load the CLAP model and processor into memory."""
        if self._active:
            logger.debug("ClapVoiceClassifier already active")
            return

        try:
            from transformers import ClapModel, ClapProcessor  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for the CLAP plugin. "
                "Install with: pip install transformers torch"
            ) from exc

        source = self.config.model_path or self.config.model_id
        logger.info("Loading CLAP model from %s on %s", source, self.config.device)

        self._processor = ClapProcessor.from_pretrained(source)
        self._model = ClapModel.from_pretrained(source).to(self.config.device)  # type: ignore[union-attr]
        self._model.eval()
        self._active = True
        logger.info("ClapVoiceClassifier activated with %d labels", len(self.config.labels))

    def deactivate(self) -> None:
        """Release model resources."""
        self._model = None
        self._processor = None
        self._active = False
        logger.info("ClapVoiceClassifier deactivated")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        audio: np.ndarray,
        labels: Optional[Sequence[str]] = None,
        top_k: int = 3,
    ) -> List[ClassificationResult]:
        """Classify an audio waveform against candidate text labels.

        Parameters
        ----------
        audio:
            1-D float32 numpy array of raw audio samples at
            ``self.config.sample_rate`` Hz.
        labels:
            Override the default command labels for this call.
        top_k:
            Number of top results to return.

        Returns
        -------
        List of :class:`ClassificationResult` sorted by descending score.
        """
        if not self._active:
            raise RuntimeError("Plugin is not activated. Call activate() first.")

        import torch  # type: ignore[import-untyped]

        candidate_labels = list(labels or self.config.labels)

        inputs = self._processor(
            text=candidate_labels,
            audios=[audio],
            return_tensors="pt",
            padding=True,
            sampling_rate=self.config.sample_rate,
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Cosine similarity between audio and each text embedding
        logits = outputs.logits_per_audio.squeeze(0)
        probs = logits.softmax(dim=-1).cpu().numpy()

        results = [
            ClassificationResult(label=lbl, score=float(score))
            for lbl, score in zip(candidate_labels, probs)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def best_match(
        self,
        audio: np.ndarray,
        labels: Optional[Sequence[str]] = None,
    ) -> Optional[ClassificationResult]:
        """Return the highest-scoring label if it exceeds the threshold."""
        results = self.classify(audio, labels=labels, top_k=1)
        if results and results[0].score >= self.config.score_threshold:
            return results[0]
        return None

    def set_labels(self, labels: List[str]) -> None:
        """Update the candidate command labels at runtime."""
        self.config.labels = list(labels)
        logger.info("Updated command labels: %s", self.config.labels)

    # ------------------------------------------------------------------
    # Homie integration helpers
    # ------------------------------------------------------------------

    def classify_from_file(self, path: str | Path, labels: Optional[Sequence[str]] = None) -> List[ClassificationResult]:
        """Load a WAV file and classify it."""
        try:
            import librosa  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("librosa is required for file loading: pip install librosa") from exc

        audio, _ = librosa.load(str(path), sr=self.config.sample_rate, mono=True)
        return self.classify(audio, labels=labels)


def register(homie_config: Optional[Dict[str, Any]] = None) -> ClapVoiceClassifier:
    """Factory called by the Homie plugin loader.

    Reads optional plugin configuration from the Homie config dict under
    ``plugins.clap_voice_classifier`` and returns an *inactive* plugin
    instance.  The caller is responsible for calling ``activate()`` when
    the plugin should load its model.

    Parameters
    ----------
    homie_config:
        The raw Homie configuration dict (``HomieConfig.raw``).  If *None*,
        default settings are used.
    """
    plugin_cfg = ClapVoiceClassifierConfig()

    if homie_config:
        section = homie_config.get("plugins", {}).get("clap_voice_classifier", {})
        if section.get("model_path"):
            plugin_cfg.model_path = section["model_path"]
        if section.get("model_id"):
            plugin_cfg.model_id = section["model_id"]
        if section.get("labels"):
            plugin_cfg.labels = list(section["labels"])
        if section.get("sample_rate"):
            plugin_cfg.sample_rate = int(section["sample_rate"])
        if section.get("score_threshold"):
            plugin_cfg.score_threshold = float(section["score_threshold"])
        if section.get("device"):
            plugin_cfg.device = section["device"]

    logger.info("Registered ClapVoiceClassifier plugin (model=%s)", plugin_cfg.model_id)
    return ClapVoiceClassifier(config=plugin_cfg)


__all__ = [
    "ClapVoiceClassifier",
    "ClapVoiceClassifierConfig",
    "ClassificationResult",
    "register",
]
