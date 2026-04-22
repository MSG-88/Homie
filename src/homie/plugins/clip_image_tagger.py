"""Homie plugin for local zero-shot image classification and tagging using OpenAI CLIP.

Uses the openai/clip-vit-large-patch14 model via the transformers library to perform
zero-shot image classification entirely on-device. Supports tagging individual images
or batch-searching a directory for images matching a set of candidate labels.

No network calls are made after the initial model download.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_LABELS = [
    "photo", "screenshot", "diagram", "document", "meme",
    "nature", "person", "animal", "food", "vehicle",
    "building", "art", "text", "chart", "map",
]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@dataclass
class TagResult:
    """Result of classifying a single image."""

    path: Path
    scores: Dict[str, float]
    top_label: str
    top_score: float


@dataclass
class CLIPImageTaggerConfig:
    """Plugin configuration."""

    model_id: str = MODEL_ID
    candidate_labels: List[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    score_threshold: float = 0.15
    device: str = "cpu"
    batch_size: int = 8


class CLIPImageTagger:
    """Zero-shot image classifier and tagger powered by CLIP.

    Provides two main capabilities for Homie:
    1. Tag a single image with confidence scores across candidate labels.
    2. Search a directory of images, returning those that match a query label
       above a configurable confidence threshold.
    """

    def __init__(self, config: Optional[CLIPImageTaggerConfig] = None) -> None:
        self.config = config or CLIPImageTaggerConfig()
        self._pipeline: Any = None
        self._active = False

    # -- lifecycle -------------------------------------------------------

    def activate(self) -> None:
        """Load the CLIP model and prepare the zero-shot pipeline."""
        if self._active:
            logger.debug("CLIPImageTagger already active")
            return

        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for the CLIP tagger plugin. "
                "Install them with: pip install transformers torch pillow"
            ) from exc

        logger.info("Loading CLIP model %s on %s ...", self.config.model_id, self.config.device)
        self._pipeline = hf_pipeline(
            task="zero-shot-image-classification",
            model=self.config.model_id,
            device=self.config.device if self.config.device != "cpu" else -1,
        )
        self._active = True
        logger.info("CLIPImageTagger activated")

    def deactivate(self) -> None:
        """Release model resources."""
        self._pipeline = None
        self._active = False
        logger.info("CLIPImageTagger deactivated")

    # -- public API ------------------------------------------------------

    def tag_image(
        self,
        image_path: str | Path,
        candidate_labels: Optional[Sequence[str]] = None,
    ) -> TagResult:
        """Classify a single image against candidate labels.

        Args:
            image_path: Path to a local image file.
            candidate_labels: Override default labels for this call.

        Returns:
            TagResult with per-label scores.
        """
        self._ensure_active()
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        labels = list(candidate_labels or self.config.candidate_labels)
        results = self._pipeline(str(path), candidate_labels=labels)

        scores: Dict[str, float] = {r["label"]: round(r["score"], 4) for r in results}
        top = max(scores, key=scores.get)  # type: ignore[arg-type]
        return TagResult(path=path, scores=scores, top_label=top, top_score=scores[top])

    def search_directory(
        self,
        directory: str | Path,
        query_labels: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
        recursive: bool = True,
    ) -> List[TagResult]:
        """Search a directory for images matching query labels above a threshold.

        Args:
            directory: Root directory to scan.
            query_labels: Labels to classify against (defaults to config labels).
            threshold: Minimum score for the top label (defaults to config threshold).
            recursive: Whether to recurse into subdirectories.

        Returns:
            List of TagResult for images whose top score meets the threshold,
            sorted by descending top score.
        """
        self._ensure_active()
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        thresh = threshold if threshold is not None else self.config.score_threshold
        labels = list(query_labels or self.config.candidate_labels)
        pattern = "**/*" if recursive else "*"

        image_paths = sorted(
            p for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not image_paths:
            logger.warning("No images found in %s", root)
            return []

        logger.info("Classifying %d images in %s ...", len(image_paths), root)
        hits: List[TagResult] = []

        for i in range(0, len(image_paths), self.config.batch_size):
            batch = image_paths[i : i + self.config.batch_size]
            for img_path in batch:
                try:
                    result = self.tag_image(img_path, candidate_labels=labels)
                    if result.top_score >= thresh:
                        hits.append(result)
                except Exception:  # noqa: BLE001
                    logger.warning("Skipping %s (failed to process)", img_path)

        hits.sort(key=lambda r: r.top_score, reverse=True)
        return hits

    def find_by_label(
        self,
        directory: str | Path,
        label: str,
        threshold: float = 0.25,
        recursive: bool = True,
    ) -> List[Tuple[Path, float]]:
        """Convenience method: find images in a directory matching a specific label.

        Returns:
            List of (path, score) tuples sorted by descending score.
        """
        results = self.search_directory(
            directory,
            query_labels=[label],
            threshold=threshold,
            recursive=recursive,
        )
        return [(r.path, r.top_score) for r in results]

    # -- internals -------------------------------------------------------

    def _ensure_active(self) -> None:
        if not self._active or self._pipeline is None:
            raise RuntimeError(
                "CLIPImageTagger is not active. Call activate() first."
            )


def register() -> Dict[str, Any]:
    """Register this plugin with Homie's plugin system.

    Returns:
        Plugin metadata dict including a factory for creating tagger instances.
    """
    return {
        "name": "clip_image_tagger",
        "version": "1.0.0",
        "description": "Zero-shot image classification and local image search using CLIP",
        "model": MODEL_ID,
        "factory": CLIPImageTagger,
        "config_class": CLIPImageTaggerConfig,
    }
