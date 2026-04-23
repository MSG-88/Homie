"""CLIP Visual Search & Tagging Plugin for Homie.

Provides zero-shot image classification, visual search, and automatic
image tagging using OpenAI's CLIP (ViT-Large/14) model running locally.
No network calls are made after initial model download.

Capabilities:
- Classify images against arbitrary text labels (zero-shot)
- Tag images with relevant categories from a configurable label set
- Search a local image directory by natural-language query
- Rank images by similarity to a text description

Requires: transformers, torch, Pillow
Model: openai/clip-vit-large-patch14 (~900 MB, downloaded once)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:
    raise ImportError(
        "clip_visual_search requires 'transformers', 'torch', and 'Pillow'. "
        "Install with: pip install transformers torch Pillow"
    ) from exc

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-large-patch14"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

DEFAULT_TAGS = [
    "photo of a person",
    "photo of an animal",
    "photo of food",
    "photo of a landscape",
    "photo of a building",
    "photo of a vehicle",
    "photo of text or a document",
    "screenshot",
    "artwork or illustration",
    "photo of electronics or gadgets",
]


@dataclass
class ClassificationResult:
    """Single image classification result."""

    label: str
    score: float


@dataclass
class ImageSearchResult:
    """Result from a visual search query."""

    path: Path
    score: float
    top_label: Optional[str] = None


@dataclass
class CLIPVisualSearchPlugin:
    """Zero-shot image classification and visual search plugin for Homie.

    Uses CLIP ViT-Large/14 to embed images and text into a shared space,
    enabling classification against arbitrary labels and text-based image
    search â€” all running locally with no network dependency at inference.
    """

    model_id: str = MODEL_ID
    device: str = ""
    default_tags: List[str] = field(default_factory=lambda: list(DEFAULT_TAGS))
    confidence_threshold: float = 0.25
    _model: Optional[CLIPModel] = field(default=None, init=False, repr=False)
    _processor: Optional[CLIPProcessor] = field(default=None, init=False, repr=False)
    _active: bool = field(default=False, init=False)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def activate(self) -> None:
        """Load the CLIP model and processor into memory."""
        if self._active:
            logger.debug("CLIPVisualSearchPlugin already active")
            return

        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CLIP model %s on %s ...", self.model_id, resolved_device)

        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(resolved_device)
        self._model.eval()
        self.device = resolved_device
        self._active = True
        logger.info("CLIPVisualSearchPlugin activated (device=%s)", self.device)

    def deactivate(self) -> None:
        """Release model resources."""
        self._model = None
        self._processor = None
        self._active = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CLIPVisualSearchPlugin deactivated")

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------ #
    #  Core inference                                                     #
    # ------------------------------------------------------------------ #

    def _ensure_active(self) -> None:
        if not self._active or self._model is None or self._processor is None:
            raise RuntimeError("Plugin is not activated. Call activate() first.")

    def _load_image(self, image_path: str | Path) -> Image.Image:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    @torch.inference_mode()
    def classify(
        self,
        image_path: str | Path,
        labels: Optional[Sequence[str]] = None,
    ) -> List[ClassificationResult]:
        """Classify an image against a set of text labels (zero-shot).

        Args:
            image_path: Path to a local image file.
            labels: Candidate text labels. Falls back to *default_tags*.

        Returns:
            List of :class:`ClassificationResult` sorted by descending score.
        """
        self._ensure_active()
        labels = list(labels or self.default_tags)
        image = self._load_image(image_path)

        inputs = self._processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0)

        results = [
            ClassificationResult(label=lbl, score=round(prob.item(), 4))
            for lbl, prob in zip(labels, probs)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def tag(
        self,
        image_path: str | Path,
        labels: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Return labels whose confidence exceeds *threshold*.

        Convenience wrapper around :meth:`classify` for auto-tagging.
        """
        threshold = threshold if threshold is not None else self.confidence_threshold
        results = self.classify(image_path, labels=labels)
        return [r.label for r in results if r.score >= threshold]

    # ------------------------------------------------------------------ #
    #  Visual search                                                      #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def search(
        self,
        query: str,
        directory: str | Path,
        top_k: int = 10,
        recursive: bool = True,
    ) -> List[ImageSearchResult]:
        """Search a directory of images using a natural-language query.

        Each image is scored by CLIP similarity to *query*. Only the
        top *top_k* results are returned, sorted by descending score.

        Args:
            query: Free-text description (e.g. "a sunset over the ocean").
            directory: Local directory containing images.
            top_k: Maximum number of results to return.
            recursive: Whether to search subdirectories.

        Returns:
            List of :class:`ImageSearchResult`.
        """
        self._ensure_active()
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        glob_pattern = "**/*" if recursive else "*"
        image_paths = sorted(
            p for p in dir_path.glob(glob_pattern)
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
        )
        if not image_paths:
            logger.warning("No images found in %s", dir_path)
            return []

        # Encode query text once
        text_inputs = self._processor(text=[query], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_embeds = self._model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        scored: List[Tuple[Path, float]] = []
        batch_size = 16

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images: List[Image.Image] = []
            valid_paths: List[Path] = []
            for p in batch_paths:
                try:
                    images.append(Image.open(p).convert("RGB"))
                    valid_paths.append(p)
                except Exception:  # noqa: BLE001
                    logger.debug("Skipping unreadable image: %s", p)

            if not images:
                continue

            img_inputs = self._processor(images=images, return_tensors="pt")
            img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
            img_embeds = self._model.get_image_features(**img_inputs)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

            similarities = (img_embeds @ text_embeds.T).squeeze(-1)
            for path, sim in zip(valid_paths, similarities):
                scored.append((path, sim.item()))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            ImageSearchResult(path=p, score=round(s, 4))
            for p, s in scored[:top_k]
        ]

    # ------------------------------------------------------------------ #
    #  Batch tagging                                                      #
    # ------------------------------------------------------------------ #

    def tag_directory(
        self,
        directory: str | Path,
        labels: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
        recursive: bool = True,
    ) -> Dict[str, List[str]]:
        """Tag every image in a directory.

        Returns:
            Mapping of ``{file_path: [matched_labels]}``.
        """
        dir_path = Path(directory)
        glob_pattern = "**/*" if recursive else "*"
        results: Dict[str, List[str]] = {}

        for p in sorted(dir_path.glob(glob_pattern)):
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS or not p.is_file():
                continue
            try:
                tags = self.tag(p, labels=labels, threshold=threshold)
                results[str(p)] = tags
            except Exception:  # noqa: BLE001
                logger.debug("Failed to tag %s", p)

        return results

    # ------------------------------------------------------------------ #
    #  Homie integration helpers                                          #
    # ------------------------------------------------------------------ #

    def handle_command(self, intent: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a Homie voice/text command to the right method.

        Supported intents:
            - ``classify_image``  â€” requires ``image_path``; optional ``labels``
            - ``tag_image``       â€” requires ``image_path``; optional ``labels``, ``threshold``
            - ``search_images``   â€” requires ``query``, ``directory``; optional ``top_k``
            - ``tag_directory``   â€” requires ``directory``; optional ``labels``, ``threshold``

        Returns:
            Dict with ``status`` and ``results`` keys.
        """
        try:
            if intent == "classify_image":
                results = self.classify(
                    args["image_path"],
                    labels=args.get("labels"),
                )
                return {"status": "ok", "results": [{"label": r.label, "score": r.score} for r in results]}

            if intent == "tag_image":
                tags = self.tag(
                    args["image_path"],
                    labels=args.get("labels"),
                    threshold=args.get("threshold"),
                )
                return {"status": "ok", "results": tags}

            if intent == "search_images":
                results = self.search(
                    query=args["query"],
                    directory=args["directory"],
                    top_k=args.get("top_k", 10),
                )
                return {"status": "ok", "results": [{"path": str(r.path), "score": r.score} for r in results]}

            if intent == "tag_directory":
                results = self.tag_directory(
                    directory=args["directory"],
                    labels=args.get("labels"),
                    threshold=args.get("threshold"),
                )
                return {"status": "ok", "results": results}

            return {"status": "error", "error": f"Unknown intent: {intent}"}

        except (FileNotFoundError, NotADirectoryError) as exc:
            return {"status": "error", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("CLIPVisualSearchPlugin command failed")
            return {"status": "error", "error": str(exc)}


# ------------------------------------------------------------------ #
#  Module-level registration for Homie plugin loader                  #
# ------------------------------------------------------------------ #

_plugin_instance: Optional[CLIPVisualSearchPlugin] = None


def register(config: Optional[Dict[str, Any]] = None) -> CLIPVisualSearchPlugin:
    """Create and activate the plugin (called by Homie's plugin loader)."""
    global _plugin_instance  # noqa: PLW0603
    cfg = config or {}
    _plugin_instance = CLIPVisualSearchPlugin(
        model_id=cfg.get("model_id", MODEL_ID),
        device=cfg.get("device", ""),
        default_tags=cfg.get("default_tags", list(DEFAULT_TAGS)),
        confidence_threshold=cfg.get("confidence_threshold", 0.25),
    )
    _plugin_instance.activate()
    return _plugin_instance


def get_plugin() -> Optional[CLIPVisualSearchPlugin]:
    """Return the active plugin instance, if any."""
    return _plugin_instance


__all__ = [
    "CLIPVisualSearchPlugin",
    "ClassificationResult",
    "ImageSearchResult",
    "register",
    "get_plugin",
]
