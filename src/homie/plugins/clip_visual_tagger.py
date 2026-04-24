"""CLIP Visual Tagger plugin for Homie.

Provides zero-shot image classification and visual search/tagging using
OpenAI's CLIP (ViT-L/14) model.  All inference runs locally via the
transformers library â€” no network calls are made at runtime.

Typical use-cases:
  - Tag photos on disk with natural-language labels.
  - Search a local image collection by text query.
  - Auto-organise screenshots, receipts, or camera imports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-large-patch14"

# Lazy-loaded heavy imports
_pipeline = None  # type: ignore[assignment]


@dataclass
class TagResult:
    """Single classification result for an image."""

    label: str
    score: float


@dataclass
class VisualTaggerConfig:
    """Runtime configuration for the CLIP tagger."""

    model_id: str = MODEL_ID
    candidate_labels: List[str] = field(
        default_factory=lambda: [
            "a photo of a person",
            "a screenshot",
            "a document or receipt",
            "a landscape or nature photo",
            "a pet or animal",
            "food or drink",
            "a chart or diagram",
            "artwork or illustration",
        ]
    )
    score_threshold: float = 0.15
    device: str = "cpu"
    batch_size: int = 8


class CLIPVisualTagger:
    """Zero-shot image classifier and tagger powered by CLIP ViT-L/14.

    The model is downloaded once on first use and cached locally by the
    transformers library.  All inference is performed on-device.
    """

    def __init__(self, config: Optional[VisualTaggerConfig] = None) -> None:
        self.config = config or VisualTaggerConfig()
        self._classifier: Any = None
        self._active: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Load the CLIP model into memory."""
        if self._active:
            return
        self._classifier = _load_classifier(
            self.config.model_id, self.config.device
        )
        self._active = True
        logger.info(
            "CLIPVisualTagger activated (model=%s, device=%s)",
            self.config.model_id,
            self.config.device,
        )

    def deactivate(self) -> None:
        """Release model resources."""
        self._classifier = None
        self._active = False
        logger.info("CLIPVisualTagger deactivated")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        image_path: str | Path,
        candidate_labels: Optional[List[str]] = None,
    ) -> List[TagResult]:
        """Classify a single image against *candidate_labels*.

        Returns a list of :class:`TagResult` sorted by descending score,
        filtered by ``config.score_threshold``.
        """
        self._ensure_active()
        labels = candidate_labels or self.config.candidate_labels
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        from PIL import Image  # lazy import

        image = Image.open(path).convert("RGB")
        results = self._classifier(image, candidate_labels=labels)
        return [
            TagResult(label=r["label"], score=round(r["score"], 4))
            for r in results
            if r["score"] >= self.config.score_threshold
        ]

    def classify_batch(
        self,
        image_paths: Sequence[str | Path],
        candidate_labels: Optional[List[str]] = None,
    ) -> Dict[str, List[TagResult]]:
        """Classify multiple images.  Returns ``{path: [TagResult, ...]}``."""
        return {
            str(p): self.classify(p, candidate_labels)
            for p in image_paths
        }

    def search(
        self,
        directory: str | Path,
        query: str,
        *,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Search *directory* for images matching a text *query*.

        Returns up to *top_k* ``(path, score)`` pairs sorted by relevance.
        """
        self._ensure_active()
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        candidates = [
            p
            for ext in extensions
            for p in dir_path.rglob(f"*{ext}")
            if p.is_file()
        ]
        if not candidates:
            return []

        scored: List[Tuple[str, float]] = []
        for img_path in candidates:
            try:
                tags = self.classify(img_path, candidate_labels=[query])
                if tags:
                    scored.append((str(img_path), tags[0].score))
            except Exception:  # noqa: BLE001
                logger.debug("Skipping %s", img_path, exc_info=True)

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def tag_directory(
        self,
        directory: str | Path,
        *,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        candidate_labels: Optional[List[str]] = None,
    ) -> Dict[str, List[TagResult]]:
        """Tag every image under *directory* with the configured labels."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        paths = [
            p
            for ext in extensions
            for p in dir_path.rglob(f"*{ext}")
            if p.is_file()
        ]
        return self.classify_batch(paths, candidate_labels)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_active(self) -> None:
        if not self._active:
            self.activate()


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _load_classifier(model_id: str, device: str) -> Any:
    """Load the HuggingFace zero-shot-image-classification pipeline."""
    try:
        from transformers import pipeline  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' and 'torch' packages are required.  "
            "Install them with: pip install transformers torch pillow"
        ) from exc

    return pipeline(
        "zero-shot-image-classification",
        model=model_id,
        device=device,
    )


def register() -> Dict[str, Any]:
    """Return plugin metadata for Homie's plugin loader."""
    return {
        "name": "clip_visual_tagger",
        "version": "1.0.0",
        "description": (
            "Zero-shot image classification and visual search using "
            "CLIP ViT-L/14.  Runs fully on-device."
        ),
        "factory": CLIPVisualTagger,
        "config_class": VisualTaggerConfig,
    }


__all__ = [
    "CLIPVisualTagger",
    "VisualTaggerConfig",
    "TagResult",
    "register",
]
