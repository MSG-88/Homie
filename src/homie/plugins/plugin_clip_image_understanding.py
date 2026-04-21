"""Homie Plugin: Local Zero-Shot Image Classification using CLIP ViT-Base-Patch32.

This plugin provides on-device image understanding capabilities using OpenAI's
CLIP (Contrastive Language-Image Pre-training) model. It enables Homie to classify
images into arbitrary categories without requiring task-specific training data.

Model: openai/clip-vit-base-patch32 (~600MB)
Backend: transformers + PyTorch (CPU-friendly base variant)
Network: Download on first use only; fully offline after caching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_CACHE_DIR = Path.home() / ".homie" / "models" / "clip-vit-base-patch32"


class CLIPImageUnderstandingPlugin:
    """Zero-shot image classification plugin using CLIP ViT-Base-Patch32.

    Allows Homie to answer questions like "What is in this image?" or
    "Does this photo contain a cat or a dog?" entirely on-device.
    """

    name: str = "clip_image_understanding"
    version: str = "1.0.0"

    def __init__(self, cache_dir: Optional[Path] = None, device: str = "cpu") -> None:
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._device = device
        self._model = None
        self._processor = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Load the CLIP model and processor into memory."""
        if self._active:
            logger.debug("CLIPImageUnderstandingPlugin already active.")
            return

        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies. Install with: "
                "pip install transformers torch pillow"
            ) from exc

        logger.info("Loading CLIP model from %s (cache: %s)", MODEL_ID, self._cache_dir)
        self._processor = CLIPProcessor.from_pretrained(
            MODEL_ID, cache_dir=str(self._cache_dir)
        )
        self._model = CLIPModel.from_pretrained(
            MODEL_ID, cache_dir=str(self._cache_dir)
        )
        self._model.to(self._device)
        self._model.eval()
        self._active = True
        logger.info("CLIPImageUnderstandingPlugin activated on device=%s", self._device)

    def deactivate(self) -> None:
        """Unload the model to free memory."""
        self._model = None
        self._processor = None
        self._active = False
        logger.info("CLIPImageUnderstandingPlugin deactivated.")

    def classify(
        self,
        image_path: str | Path,
        candidate_labels: List[str],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Classify an image against a set of candidate text labels.

        Args:
            image_path: Path to a local image file (JPEG, PNG, BMP, etc.).
            candidate_labels: Free-form text descriptions to score against.
            top_k: Number of top results to return.

        Returns:
            A list of (label, probability) tuples sorted by descending score.
        """
        if not self._active:
            raise RuntimeError("Plugin is not active. Call activate() first.")

        from PIL import Image  # type: ignore
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze(0).cpu().tolist()

        scored = sorted(
            zip(candidate_labels, probs), key=lambda x: x[1], reverse=True
        )
        return scored[:top_k]

    def describe(
        self,
        image_path: str | Path,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """High-level image description for Homie's assistant responses.

        Args:
            image_path: Path to a local image file.
            categories: Optional custom categories. Falls back to a general set.

        Returns:
            Dict with 'top_label', 'confidence', and full 'scores'.
        """
        if categories is None:
            categories = [
                "a photo of a person",
                "a photo of an animal",
                "a photo of food",
                "a photo of a vehicle",
                "a photo of a building",
                "a photo of nature or landscape",
                "a photo of text or a document",
                "a photo of electronics or a screen",
                "a photo of artwork or illustration",
                "a photo of an indoor scene",
            ]

        results = self.classify(image_path, categories, top_k=len(categories))
        top_label, top_score = results[0]

        return {
            "top_label": top_label,
            "confidence": round(top_score, 4),
            "scores": {label: round(score, 4) for label, score in results},
        }

    def compute_similarity(
        self,
        image_path: str | Path,
        text: str,
    ) -> float:
        """Compute cosine similarity between a single image and text query.

        Useful for Homie's RAG pipeline to score image relevance.

        Args:
            image_path: Path to a local image file.
            text: A natural language query.

        Returns:
            A similarity score (higher = more relevant).
        """
        results = self.classify(image_path, [text, "something else entirely"])
        return results[0][1] if results[0][0] == text else 1.0 - results[0][1]


# --- Homie registration interface ---

_instance: Optional[CLIPImageUnderstandingPlugin] = None


def register(config: Optional[Dict[str, Any]] = None) -> CLIPImageUnderstandingPlugin:
    """Register and return the plugin instance.

    Args:
        config: Optional dict with keys 'cache_dir' and 'device'.

    Returns:
        The plugin instance (not yet activated; call activate() when ready).
    """
    global _instance
    config = config or {}
    _instance = CLIPImageUnderstandingPlugin(
        cache_dir=Path(config["cache_dir"]) if "cache_dir" in config else None,
        device=config.get("device", "cpu"),
    )
    logger.info("Registered %s v%s", _instance.name, _instance.version)
    return _instance
