"""Homie plugin for zero-shot image classification and visual search/tagging.

Uses OpenAI's CLIP (ViT-L/14) to classify images against arbitrary text labels
without any fine-tuning.  Runs entirely on-device via the transformers library.
No network calls are made after the initial (opt-in) model download.

Capabilities:
  - Zero-shot image classification with user-defined labels
  - Batch tagging of local image directories
  - Semantic image search: rank a folder of images by relevance to a text query
  - Embedding extraction for downstream vector-search in ChromaDB
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-large-patch14"

# Lazy-loaded globals so the heavy imports only happen when the plugin activates.
_processor: Any = None
_model: Any = None


@dataclass
class ClassificationResult:
    """Single image classification output."""

    image_path: str
    labels: List[str]
    scores: List[float]

    @property
    def best_label(self) -> str:
        return self.labels[0] if self.labels else ""

    @property
    def best_score(self) -> float:
        return self.scores[0] if self.scores else 0.0

    def above_threshold(self, threshold: float = 0.25) -> List[Tuple[str, float]]:
        """Return (label, score) pairs that exceed *threshold*."""
        return [(l, s) for l, s in zip(self.labels, self.scores) if s >= threshold]


@dataclass
class SearchResult:
    """A single image scored against a text query."""

    image_path: str
    score: float


@dataclass
class PluginConfig:
    """Runtime configuration for the CLIP visual-search plugin."""

    model_id: str = MODEL_ID
    device: str = "cpu"
    default_labels: List[str] = field(
        default_factory=lambda: [
            "photo", "screenshot", "diagram", "document",
            "meme", "artwork", "receipt", "handwriting",
        ]
    )
    score_threshold: float = 0.25
    batch_size: int = 8
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


class CLIPVisualSearchPlugin:
    """Zero-shot image classification and visual search for Homie.

    Typical lifecycle::

        plugin = CLIPVisualSearchPlugin()
        plugin.activate()                     # loads CLIP onto device
        results = plugin.classify("photo.jpg", ["cat", "dog"])
        plugin.deactivate()                   # frees VRAM / RAM
    """

    def __init__(self, config: Optional[PluginConfig] = None) -> None:
        self.config = config or PluginConfig()
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Load the CLIP model and processor into memory."""
        if self._active:
            logger.debug("CLIPVisualSearchPlugin already active")
            return

        global _processor, _model  # noqa: PLW0603

        try:
            from transformers import CLIPProcessor, CLIPModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required.  "
                "Install with:  pip install transformers torch pillow"
            ) from exc

        logger.info("Loading CLIP model %s on %s â€¦", self.config.model_id, self.config.device)
        _processor = CLIPProcessor.from_pretrained(self.config.model_id)
        _model = CLIPModel.from_pretrained(self.config.model_id).to(self.config.device)
        _model.eval()
        self._active = True
        logger.info("CLIPVisualSearchPlugin activated")

    def deactivate(self) -> None:
        """Release model resources."""
        global _processor, _model  # noqa: PLW0603
        _processor = None
        _model = None
        self._active = False
        logger.info("CLIPVisualSearchPlugin deactivated")

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError("Plugin is not activated. Call activate() first.")

    @staticmethod
    def _load_image(path: str | Path) -> Any:
        from PIL import Image  # type: ignore[import-untyped]

        img = Image.open(path).convert("RGB")
        return img

    def _classify_single(
        self, image_path: str | Path, candidate_labels: List[str]
    ) -> ClassificationResult:
        """Run zero-shot classification on one image."""
        import torch  # type: ignore[import-untyped]

        self._ensure_active()
        image = self._load_image(image_path)
        inputs = _processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.config.device)

        with torch.no_grad():
            outputs = _model(**inputs)

        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=-1).cpu().tolist()

        paired = sorted(zip(candidate_labels, probs), key=lambda x: x[1], reverse=True)
        sorted_labels = [p[0] for p in paired]
        sorted_scores = [round(p[1], 4) for p in paired]

        return ClassificationResult(
            image_path=str(image_path),
            labels=sorted_labels,
            scores=sorted_scores,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        image_path: str | Path,
        candidate_labels: Optional[List[str]] = None,
    ) -> ClassificationResult:
        """Classify a single image against *candidate_labels*.

        Falls back to ``config.default_labels`` when no labels are provided.
        """
        labels = candidate_labels or self.config.default_labels
        return self._classify_single(image_path, labels)

    def tag_directory(
        self,
        directory: str | Path,
        candidate_labels: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> List[ClassificationResult]:
        """Classify every image in *directory* and return results.

        Only labels with a probability >= *threshold* are kept in each result's
        ``above_threshold`` view; the full ranked list is always available.
        """
        self._ensure_active()
        labels = candidate_labels or self.config.default_labels
        threshold = threshold if threshold is not None else self.config.score_threshold
        dirpath = Path(directory)
        if not dirpath.is_dir():
            raise FileNotFoundError(f"Directory not found: {dirpath}")

        image_files = sorted(
            p for p in dirpath.iterdir()
            if p.suffix.lower() in self.config.image_extensions
        )
        results: List[ClassificationResult] = []
        for img_path in image_files:
            try:
                results.append(self._classify_single(img_path, labels))
            except Exception:  # noqa: BLE001
                logger.warning("Skipping %s â€” failed to process", img_path)
        return results

    def search(
        self,
        query: str,
        directory: str | Path,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Rank images in *directory* by relevance to a free-text *query*.

        Returns the top-*k* results sorted by descending similarity score.
        """
        import torch  # type: ignore[import-untyped]

        self._ensure_active()
        dirpath = Path(directory)
        if not dirpath.is_dir():
            raise FileNotFoundError(f"Directory not found: {dirpath}")

        image_files = sorted(
            p for p in dirpath.iterdir()
            if p.suffix.lower() in self.config.image_extensions
        )
        if not image_files:
            return []

        scored: List[SearchResult] = []
        for batch_start in range(0, len(image_files), self.config.batch_size):
            batch_paths = image_files[batch_start : batch_start + self.config.batch_size]
            images = [self._load_image(p) for p in batch_paths]
            inputs = _processor(
                text=[query],
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.config.device)

            with torch.no_grad():
                outputs = _model(**inputs)

            sims = outputs.logits_per_image[:, 0].cpu().tolist()
            for path, sim in zip(batch_paths, sims):
                scored.append(SearchResult(image_path=str(path), score=round(sim, 4)))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def image_embedding(self, image_path: str | Path) -> List[float]:
        """Return the CLIP image embedding as a flat list of floats.

        Useful for inserting into ChromaDB or another vector store for
        persistent semantic image search.
        """
        import torch  # type: ignore[import-untyped]

        self._ensure_active()
        image = self._load_image(image_path)
        inputs = _processor(images=image, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            emb = _model.get_image_features(**inputs)

        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().tolist()

    def text_embedding(self, text: str) -> List[float]:
        """Return the CLIP text embedding as a flat list of floats."""
        import torch  # type: ignore[import-untyped]

        self._ensure_active()
        inputs = _processor(text=[text], return_tensors="pt", padding=True).to(self.config.device)

        with torch.no_grad():
            emb = _model.get_text_features(**inputs)

        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().tolist()


# ------------------------------------------------------------------
# Module-level convenience (register / deregister)
# ------------------------------------------------------------------

_default_plugin: Optional[CLIPVisualSearchPlugin] = None


def register(config: Optional[PluginConfig] = None) -> CLIPVisualSearchPlugin:
    """Create, activate, and return the default plugin instance."""
    global _default_plugin  # noqa: PLW0603
    _default_plugin = CLIPVisualSearchPlugin(config)
    _default_plugin.activate()
    return _default_plugin


def deregister() -> None:
    """Deactivate and discard the default plugin instance."""
    global _default_plugin  # noqa: PLW0603
    if _default_plugin is not None:
        _default_plugin.deactivate()
        _default_plugin = None
