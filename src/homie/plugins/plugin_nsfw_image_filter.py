"""Homie Content Safety Plugin â€” Local NSFW Image Filtering

Privacy-first plugin that uses the Falconsai/nsfw_image_detection model
(ViT-based image classifier) to detect NSFW content in images entirely
on-device. No images or results are ever sent over the network.

The model is downloaded once from Hugging Face and cached locally.
Subsequent runs use the cached weights with zero network traffic.

Usage:
    plugin = NSFWImageFilterPlugin()
    plugin.activate()
    result = plugin.classify("/path/to/image.jpg")
    # result: {"label": "nsfw" | "normal", "score": 0.97, "safe": False}
    plugin.deactivate()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Falconsai/nsfw_image_detection"
DEFAULT_CACHE_DIR = Path.home() / ".homie" / "models" / "nsfw_filter"
DEFAULT_THRESHOLD = 0.5


@dataclass
class ClassificationResult:
    """Result of a single image classification."""

    path: str
    label: str
    score: float
    safe: bool
    all_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "label": self.label,
            "score": round(self.score, 4),
            "safe": self.safe,
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
        }


class NSFWImageFilterPlugin:
    """Local NSFW image detection plugin for Homie.

    Uses a ViT (Vision Transformer) classifier from
    Falconsai/nsfw_image_detection to label images as
    'normal' or 'nsfw' entirely on-device.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier. Defaults to
        ``Falconsai/nsfw_image_detection``.
    cache_dir:
        Local directory for cached model weights.
    threshold:
        Confidence threshold above which the top label is accepted.
        Images below this threshold are conservatively marked unsafe.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        Defaults to ``"cpu"`` for broadest compatibility.
    """

    name: str = "nsfw_image_filter"
    version: str = "1.0.0"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Optional[Union[str, Path]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.threshold = threshold
        self.device = device
        self._pipeline: Any = None
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Load the model into memory and prepare the classification pipeline."""
        if self._active:
            logger.debug("NSFWImageFilterPlugin already active")
            return

        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "transformers package is required. "
                "Install with: pip install transformers torch pillow"
            ) from exc

        logger.info(
            "Loading NSFW detection model '%s' on device '%s' â€¦",
            self.model_id,
            self.device,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline = hf_pipeline(
            task="image-classification",
            model=self.model_id,
            device=self.device,
            model_kwargs={"cache_dir": str(self.cache_dir)},
        )
        self._active = True
        logger.info("NSFWImageFilterPlugin activated")

    def deactivate(self) -> None:
        """Release model resources."""
        self._pipeline = None
        self._active = False
        logger.info("NSFWImageFilterPlugin deactivated")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, image_path: Union[str, Path]) -> ClassificationResult:
        """Classify a single image as normal or nsfw.

        Parameters
        ----------
        image_path:
            Path to a local image file (JPEG, PNG, BMP, etc.).

        Returns
        -------
        ClassificationResult
            Contains the predicted label, confidence score, and a boolean
            ``safe`` flag.
        """
        self._ensure_active()
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        from PIL import Image  # type: ignore

        image = Image.open(path).convert("RGB")
        predictions: List[Dict[str, Any]] = self._pipeline(image)  # type: ignore[misc]

        all_scores = {p["label"]: float(p["score"]) for p in predictions}
        top = max(predictions, key=lambda p: p["score"])
        label: str = top["label"]
        score: float = float(top["score"])

        safe = label.lower() in {"normal", "safe", "sfw"} and score >= self.threshold

        return ClassificationResult(
            path=str(path),
            label=label,
            score=score,
            safe=safe,
            all_scores=all_scores,
        )

    def classify_batch(
        self, image_paths: Sequence[Union[str, Path]]
    ) -> List[ClassificationResult]:
        """Classify multiple images and return results for each."""
        return [self.classify(p) for p in image_paths]

    def is_safe(self, image_path: Union[str, Path]) -> bool:
        """Convenience check â€” returns ``True`` if the image is SFW."""
        return self.classify(image_path).safe

    # ------------------------------------------------------------------
    # Homie integration helpers
    # ------------------------------------------------------------------

    def scan_directory(
        self,
        directory: Union[str, Path],
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        recursive: bool = False,
    ) -> List[ClassificationResult]:
        """Scan a directory for images and classify each one.

        Parameters
        ----------
        directory:
            Path to the directory to scan.
        extensions:
            File extensions to consider as images.
        recursive:
            If ``True``, scan subdirectories as well.

        Returns
        -------
        list[ClassificationResult]
            One result per image found.
        """
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        pattern_fn = root.rglob if recursive else root.glob
        image_files = sorted(
            p for ext in extensions for p in pattern_fn(f"*{ext}") if p.is_file()
        )
        logger.info("Scanning %d images in %s", len(image_files), root)
        return self.classify_batch(image_files)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_active(self) -> None:
        if not self._active or self._pipeline is None:
            raise RuntimeError(
                "Plugin is not active. Call activate() before classifying images."
            )

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        return (
            f"NSFWImageFilterPlugin(model={self.model_id!r}, "
            f"device={self.device!r}, threshold={self.threshold}, {status})"
        )


# ------------------------------------------------------------------
# Module-level register / helpers for Homie plugin discovery
# ------------------------------------------------------------------

_instance: Optional[NSFWImageFilterPlugin] = None


def register(
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Optional[Union[str, Path]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = "cpu",
) -> NSFWImageFilterPlugin:
    """Create, activate, and return the singleton plugin instance."""
    global _instance
    if _instance is None:
        _instance = NSFWImageFilterPlugin(
            model_id=model_id,
            cache_dir=cache_dir,
            threshold=threshold,
            device=device,
        )
    if not _instance._active:
        _instance.activate()
    return _instance


def get_instance() -> Optional[NSFWImageFilterPlugin]:
    """Return the current singleton instance, if any."""
    return _instance
