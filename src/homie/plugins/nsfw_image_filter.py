"""Homie Content Safety Filter Plugin â€” Local NSFW Image Moderation.

Uses the Falconsai/nsfw_image_detection Vision Transformer (ViT) model to
classify images as *nsfw* or *normal* entirely on-device.  No network calls
are made after the one-time model download.

Integration points:
  * Other Homie plugins can call ``NsfwImageFilter.check(path)`` to gate
    image processing pipelines.
  * The controller can invoke ``is_safe()`` as a pre-flight check before
    displaying, forwarding, or indexing user-supplied images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded heavy imports so the module stays fast to import
# ---------------------------------------------------------------------------
_pipeline = None
_Image = None

MODEL_ID = "Falconsai/nsfw_image_detection"
DEFAULT_THRESHOLD = 0.5
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@dataclass
class ModerationResult:
    """Outcome of a single image moderation check."""

    path: str
    label: str
    score: float
    is_safe: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "label": self.label,
            "score": round(self.score, 4),
            "is_safe": self.is_safe,
        }


def _load_model() -> None:
    """Lazy-load the image-classification pipeline and PIL."""
    global _pipeline, _Image  # noqa: PLW0603
    if _pipeline is not None:
        return

    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]
        from PIL import Image as PILImage  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "nsfw_image_filter requires 'transformers', 'torch', and 'Pillow'. "
            "Install them with: pip install transformers torch pillow"
        ) from exc

    logger.info("Loading NSFW detection model %s (first run downloads ~350 MB)â€¦", MODEL_ID)
    _pipeline = hf_pipeline("image-classification", model=MODEL_ID)
    _Image = PILImage
    logger.info("NSFW detection model loaded.")


class NsfwImageFilter:
    """Local content-safety filter for images.

    Parameters
    ----------
    threshold:
        Confidence threshold above which an *nsfw* label causes the image to
        be flagged as unsafe.  Defaults to ``0.5``.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._active = False

    # -- lifecycle -----------------------------------------------------------

    def activate(self) -> None:
        """Pre-load the model so subsequent calls are fast."""
        _load_model()
        self._active = True
        logger.info("NsfwImageFilter activated (threshold=%.2f).", self.threshold)

    def deactivate(self) -> None:
        """Mark the plugin as inactive.  Model stays cached in-process."""
        self._active = False
        logger.info("NsfwImageFilter deactivated.")

    # -- public API ----------------------------------------------------------

    def check(self, image_path: str | Path) -> ModerationResult:
        """Classify a single image and return a ``ModerationResult``.

        Raises
        ------
        FileNotFoundError
            If *image_path* does not exist.
        ValueError
            If the file extension is not in ``SUPPORTED_EXTENSIONS``.
        RuntimeError
            If required dependencies are missing.
        """
        _load_model()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        image = _Image.open(path).convert("RGB")
        predictions: List[Dict[str, object]] = _pipeline(image)  # type: ignore[arg-type]

        nsfw_score = 0.0
        top_label = "normal"
        top_score = 0.0
        for pred in predictions:
            label = str(pred["label"]).lower()
            score = float(pred["score"])  # type: ignore[arg-type]
            if label == "nsfw" and score > nsfw_score:
                nsfw_score = score
            if score > top_score:
                top_label = label
                top_score = score

        is_safe = nsfw_score < self.threshold

        result = ModerationResult(
            path=str(path),
            label=top_label,
            score=top_score,
            is_safe=is_safe,
        )
        logger.debug("Moderation result: %s", result.as_dict())
        return result

    def check_batch(self, image_paths: List[str | Path]) -> List[ModerationResult]:
        """Classify multiple images and return results for each."""
        return [self.check(p) for p in image_paths]

    def is_safe(self, image_path: str | Path) -> bool:
        """Convenience wrapper â€” returns ``True`` when the image is safe."""
        return self.check(image_path).is_safe


# ---------------------------------------------------------------------------
# Module-level convenience for Homie plugin registration
# ---------------------------------------------------------------------------

_instance: Optional[NsfwImageFilter] = None


def register(threshold: float = DEFAULT_THRESHOLD) -> NsfwImageFilter:
    """Create, activate, and return the singleton filter instance."""
    global _instance  # noqa: PLW0603
    if _instance is None:
        _instance = NsfwImageFilter(threshold=threshold)
        _instance.activate()
    return _instance
