"""Homie plugin for local zero-shot image classification and tagging using OpenAI CLIP.

Uses the openai/clip-vit-large-patch14 model via the transformers library to perform
zero-shot image classification entirely on-device. Supports tagging images with
arbitrary user-defined labels and searching a local image directory by semantic query.

No network calls are made after the initial model download (which is opt-in).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment,misc]

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
except ImportError:  # pragma: no cover
    CLIPProcessor = None  # type: ignore[assignment,misc]
    CLIPModel = None  # type: ignore[assignment,misc]
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-large-patch14"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

DEFAULT_LABELS = [
    "photo of a person",
    "photo of a pet",
    "photo of food",
    "screenshot",
    "document or receipt",
    "landscape or nature",
    "architecture or building",
    "artwork or illustration",
    "meme or text image",
    "chart or diagram",
]


@dataclass
class TagResult:
    """Result of classifying a single image."""

    path: Path
    scores: Dict[str, float]

    @property
    def top_label(self) -> str:
        return max(self.scores, key=self.scores.get)  # type: ignore[arg-type]

    @property
    def top_score(self) -> float:
        return self.scores[self.top_label]


@dataclass
class SearchResult:
    """An image matched by semantic search."""

    path: Path
    score: float


class CLIPImageTagger:
    """Zero-shot image classifier and semantic search engine powered by CLIP.

    All inference runs locally via the transformers library. The model is loaded
    lazily on first use and cached in memory for the lifetime of the plugin.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
        default_labels: Optional[List[str]] = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._default_labels = default_labels or list(DEFAULT_LABELS)
        self._model: Any = None
        self._processor: Any = None

    # -- lifecycle --------------------------------------------------------

    def _ensure_deps(self) -> None:
        """Verify that optional dependencies are installed."""
        missing: List[str] = []
        if Image is None:
            missing.append("Pillow")
        if CLIPModel is None:
            missing.append("transformers")
        if torch is None:
            missing.append("torch")
        if missing:
            raise RuntimeError(
                f"Missing dependencies for CLIP plugin: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )

    def _load_model(self) -> None:
        """Load model and processor (downloads on first run, cached locally afterwards)."""
        if self._model is not None:
            return
        self._ensure_deps()
        logger.info("Loading CLIP model %s (first run may download ~1.7 GB)...", self._model_id)
        self._processor = CLIPProcessor.from_pretrained(self._model_id)
        self._model = CLIPModel.from_pretrained(self._model_id)
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device).eval()
        self._device = device
        logger.info("CLIP model loaded on %s", self._device)

    # -- core operations --------------------------------------------------

    def classify(
        self,
        image_path: str | Path,
        labels: Optional[List[str]] = None,
    ) -> TagResult:
        """Classify a single image against a set of text labels.

        Args:
            image_path: Path to the image file.
            labels: Candidate labels. Falls back to ``default_labels``.

        Returns:
            A ``TagResult`` with per-label probabilities.
        """
        self._load_model()
        labels = labels or self._default_labels
        path = Path(image_path)
        img = Image.open(path).convert("RGB")

        inputs = self._processor(
            text=labels,
            images=img,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=-1).cpu().tolist()

        scores = {label: round(prob, 4) for label, prob in zip(labels, probs)}
        return TagResult(path=path, scores=scores)

    def tag_directory(
        self,
        directory: str | Path,
        labels: Optional[List[str]] = None,
        threshold: float = 0.25,
        recursive: bool = True,
    ) -> List[TagResult]:
        """Tag every image in a directory.

        Args:
            directory: Root directory to scan.
            labels: Candidate labels.
            threshold: Minimum top-score to include in results.
            recursive: Whether to recurse into subdirectories.

        Returns:
            List of ``TagResult`` objects whose top score meets the threshold.
        """
        root = Path(directory)
        pattern = "**/*" if recursive else "*"
        results: List[TagResult] = []
        for p in sorted(root.glob(pattern)):
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                result = self.classify(p, labels=labels)
                if result.top_score >= threshold:
                    results.append(result)
            except Exception:  # noqa: BLE001
                logger.warning("Skipping unreadable image: %s", p)
        return results

    def search(
        self,
        directory: str | Path,
        query: str,
        top_k: int = 10,
        recursive: bool = True,
    ) -> List[SearchResult]:
        """Semantic image search: rank images by similarity to a text query.

        Args:
            directory: Root directory to scan.
            query: Natural-language description of what to find.
            top_k: Maximum number of results to return.
            recursive: Whether to recurse into subdirectories.

        Returns:
            Top-k ``SearchResult`` objects sorted by descending similarity.
        """
        self._load_model()
        root = Path(directory)
        pattern = "**/*" if recursive else "*"

        image_paths: List[Path] = [
            p for p in sorted(root.glob(pattern))
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not image_paths:
            return []

        scored: List[Tuple[Path, float]] = []
        # Process in batches to limit memory usage
        batch_size = 16
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            valid_paths = []
            for p in batch_paths:
                try:
                    images.append(Image.open(p).convert("RGB"))
                    valid_paths.append(p)
                except Exception:  # noqa: BLE001
                    logger.warning("Skipping unreadable image: %s", p)

            if not images:
                continue

            inputs = self._processor(
                text=[query],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                sims = outputs.logits_per_image[:, 0].cpu().tolist()

            for path, sim in zip(valid_paths, sims):
                scored.append((path, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(path=p, score=round(s, 4))
            for p, s in scored[:top_k]
        ]


# -- Plugin registration (Homie convention) --------------------------------

_instance: Optional[CLIPImageTagger] = None


def activate(config: Optional[Dict[str, Any]] = None) -> CLIPImageTagger:
    """Activate the CLIP image tagger plugin.

    Args:
        config: Optional dict with keys ``model_id``, ``device``, ``default_labels``.

    Returns:
        The plugin instance (lazy-loads model on first classification call).
    """
    global _instance
    config = config or {}
    _instance = CLIPImageTagger(
        model_id=config.get("model_id", MODEL_ID),
        device=config.get("device"),
        default_labels=config.get("default_labels"),
    )
    logger.info("CLIP Image Tagger plugin activated (model=%s)", _instance._model_id)
    return _instance


def deactivate() -> None:
    """Release model resources."""
    global _instance
    if _instance is not None:
        _instance._model = None
        _instance._processor = None
        _instance = None
        if torch is not None:
            torch.cuda.empty_cache()
    logger.info("CLIP Image Tagger plugin deactivated")


def register() -> Dict[str, Any]:
    """Return plugin metadata for Homie's plugin registry."""
    return {
        "name": "clip_image_tagger",
        "version": "1.0.0",
        "description": "Zero-shot image classification and semantic search using CLIP",
        "author": "Homie",
        "dependencies": ["transformers", "torch", "Pillow"],
        "activate": activate,
        "deactivate": deactivate,
    }
