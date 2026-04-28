"""CLIP Visual Search & Tagging Plugin for Homie.

Provides zero-shot image classification, visual search, and automatic tagging
using OpenAI's CLIP (ViT-Large/14) model running entirely on-device.

Capabilities:
- Classify images against arbitrary text labels without training
- Tag images in a directory with user-defined or auto-discovered categories
- Search a folder of images by natural-language query
- Rank images by relevance to a text description

All inference runs locally via the ``transformers`` library. No network
calls are made after the initial (opt-in) model download.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Single image classification output."""

    image_path: str
    scores: Dict[str, float]  # label -> probability
    top_label: str
    top_score: float


@dataclass
class SearchResult:
    """Single search hit."""

    image_path: str
    score: float


@dataclass
class TaggingReport:
    """Result of batch-tagging a directory of images."""

    results: List[ClassificationResult] = field(default_factory=list)
    errors: List[Tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class CLIPVisualSearchTagger:
    """Zero-shot image classification and visual search powered by CLIP.

    The model is loaded lazily on first use so that plugin registration is
    instantaneous and memory is not consumed until needed.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier.  Defaults to
        ``openai/clip-vit-large-patch14``.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).  When
        *None* the plugin auto-selects the best available device.
    cache_dir:
        Optional local directory for caching model weights.  Falls back to
        the Hugging Face default cache when *None*.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self._device = device
        self._cache_dir = cache_dir
        self._model: Any = None
        self._processor: Any = None
        self._active = False

    # -- lifecycle -----------------------------------------------------------

    def activate(self) -> None:
        """Mark the plugin as active.  Model loading is deferred to first use."""
        self._active = True
        logger.info("CLIPVisualSearchTagger activated (model will load on first use)")

    def deactivate(self) -> None:
        """Release model resources and deactivate the plugin."""
        self._model = None
        self._processor = None
        self._active = False
        logger.info("CLIPVisualSearchTagger deactivated")

    @property
    def is_active(self) -> bool:
        return self._active

    # -- lazy loading --------------------------------------------------------

    def _resolve_device(self) -> str:
        """Pick the best available device if the user did not specify one."""
        if self._device is not None:
            return self._device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _ensure_model(self) -> None:
        """Load model and processor on first use."""
        if self._model is not None:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' and 'torch' packages are required. "
                "Install them with: pip install transformers torch"
            ) from exc

        device = self._resolve_device()
        logger.info("Loading CLIP model %s on %s â€¦", self.model_id, device)

        self._processor = CLIPProcessor.from_pretrained(
            self.model_id, cache_dir=self._cache_dir
        )
        self._model = CLIPModel.from_pretrained(
            self.model_id, cache_dir=self._cache_dir
        ).to(device)
        self._model.eval()
        self._device = device
        logger.info("CLIP model loaded successfully")

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _load_image(path: str | Path) -> Any:
        """Open an image file and convert to RGB."""
        from PIL import Image

        return Image.open(path).convert("RGB")

    @staticmethod
    def _collect_images(directory: str | Path) -> List[Path]:
        """Return all supported image files in *directory* (non-recursive)."""
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")
        return sorted(
            p for p in root.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    # -- public API ----------------------------------------------------------

    def classify(
        self,
        image_path: str | Path,
        labels: Sequence[str],
    ) -> ClassificationResult:
        """Classify a single image against the given text labels.

        Returns a :class:`ClassificationResult` with per-label probabilities.
        """
        if not labels:
            raise ValueError("At least one label is required")

        self._ensure_model()
        import torch

        image = self._load_image(image_path)
        inputs = self._processor(
            text=list(labels), images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image  # (1, num_labels)
            probs = logits.softmax(dim=-1).squeeze(0).cpu().tolist()

        scores = {label: round(prob, 4) for label, prob in zip(labels, probs)}
        top_label = max(scores, key=scores.get)  # type: ignore[arg-type]
        return ClassificationResult(
            image_path=str(image_path),
            scores=scores,
            top_label=top_label,
            top_score=scores[top_label],
        )

    def tag_directory(
        self,
        directory: str | Path,
        labels: Sequence[str],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> TaggingReport:
        """Classify every image in *directory* and return a report.

        Labels whose probability falls below *threshold* are omitted from
        each result's ``scores`` dict.
        """
        report = TaggingReport()
        for img_path in self._collect_images(directory):
            try:
                result = self.classify(img_path, labels)
                result.scores = {
                    k: v for k, v in result.scores.items() if v >= threshold
                }
                report.results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to classify %s: %s", img_path, exc)
                report.errors.append((str(img_path), str(exc)))
        return report

    def search(
        self,
        directory: str | Path,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[SearchResult]:
        """Rank images in *directory* by relevance to a text *query*.

        Returns the top-*k* results sorted by descending similarity.
        """
        self._ensure_model()
        import torch

        image_paths = self._collect_images(directory)
        if not image_paths:
            return []

        images = [self._load_image(p) for p in image_paths]
        inputs = self._processor(
            text=[query], images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # logits_per_text shape: (1, num_images)
            scores = outputs.logits_per_text.softmax(dim=-1).squeeze(0).cpu().tolist()

        ranked = sorted(
            zip(image_paths, scores), key=lambda t: t[1], reverse=True
        )
        return [
            SearchResult(image_path=str(p), score=round(s, 4))
            for p, s in ranked[:top_k]
        ]

    def similarity(
        self,
        image_path: str | Path,
        query: str,
    ) -> float:
        """Return the cosine similarity between an image and a text query."""
        self._ensure_model()
        import torch

        image = self._load_image(image_path)
        inputs = self._processor(
            text=[query], images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            img_embeds = outputs.image_embeds  # (1, dim)
            txt_embeds = outputs.text_embeds   # (1, dim)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)
            cosine = (img_embeds * txt_embeds).sum().item()

        return round(cosine, 4)


# ---------------------------------------------------------------------------
# Module-level convenience (register / unregister)
# ---------------------------------------------------------------------------

_default_instance: Optional[CLIPVisualSearchTagger] = None


def register(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> CLIPVisualSearchTagger:
    """Create, activate, and return the default plugin instance."""
    global _default_instance
    _default_instance = CLIPVisualSearchTagger(
        model_id=model_id, device=device, cache_dir=cache_dir
    )
    _default_instance.activate()
    return _default_instance


def get_instance() -> Optional[CLIPVisualSearchTagger]:
    """Return the current default instance, if registered."""
    return _default_instance
