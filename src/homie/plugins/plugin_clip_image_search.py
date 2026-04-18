"""Homie plugin for local image search by text query using OpenAI CLIP.

Uses the openai/clip-vit-base-patch32 model to encode images and text into
a shared embedding space, enabling natural-language search over a local
image directory.  All inference runs locally via transformers + PyTorch;
no network calls are made after the initial model download.

Typical usage through Homie:
    plugin = CLIPImageSearchPlugin(config)
    plugin.activate()
    plugin.index_directory("/home/user/Photos")
    results = plugin.search("a dog playing in the snow", top_k=5)
    plugin.deactivate()
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
except ImportError as _exc:
    raise ImportError(
        "clip-image-search plugin requires torch, Pillow, and transformers. "
        "Install them with:  pip install torch torchvision Pillow transformers"
    ) from _exc

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
})


@dataclass
class SearchResult:
    """A single image search hit."""

    path: str
    score: float


@dataclass
class _ImageRecord:
    path: Path
    embedding: torch.Tensor


class CLIPImageSearchPlugin:
    """Local image search powered by CLIP embeddings.

    Parameters
    ----------
    config : dict | None
        Optional configuration dict.  Recognised keys:
        - ``model_id``  : HuggingFace model ID (default: openai/clip-vit-base-patch32)
        - ``device``    : "cpu", "cuda", or "auto" (default: "auto")
        - ``batch_size``: number of images encoded per batch (default: 16)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._model_id: str = self._config.get("model_id", MODEL_ID)
        self._device: str = self._resolve_device(self._config.get("device", "auto"))
        self._batch_size: int = int(self._config.get("batch_size", 16))
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None
        self._index: List[_ImageRecord] = []
        self._active: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Load the CLIP model and processor into memory."""
        if self._active:
            logger.debug("CLIPImageSearchPlugin already active")
            return
        logger.info("Loading CLIP model %s on %s", self._model_id, self._device)
        self._processor = CLIPProcessor.from_pretrained(self._model_id)
        self._model = CLIPModel.from_pretrained(self._model_id).to(self._device).eval()
        self._active = True
        logger.info("CLIPImageSearchPlugin activated")

    def deactivate(self) -> None:
        """Release model resources and clear the index."""
        self._model = None
        self._processor = None
        self._index.clear()
        self._active = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CLIPImageSearchPlugin deactivated")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
    ) -> int:
        """Scan *directory* for images and compute CLIP embeddings.

        Returns the number of images successfully indexed.
        """
        self._assert_active()
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")

        pattern = "**/*" if recursive else "*"
        paths = sorted(
            p for p in root.glob(pattern)
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
        )
        if not paths:
            logger.warning("No supported images found in %s", root)
            return 0

        logger.info("Indexing %d images from %s", len(paths), root)
        indexed = 0
        for batch_start in range(0, len(paths), self._batch_size):
            batch_paths = paths[batch_start : batch_start + self._batch_size]
            images: List[Image.Image] = []
            valid_paths: List[Path] = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    valid_paths.append(p)
                except Exception:  # noqa: BLE001
                    logger.warning("Skipping unreadable image: %s", p)
            if not images:
                continue
            embeddings = self._encode_images(images)
            for path, emb in zip(valid_paths, embeddings):
                self._index.append(_ImageRecord(path=path, embedding=emb))
                indexed += 1

        logger.info("Indexed %d images", indexed)
        return indexed

    def index_count(self) -> int:
        """Return the number of currently indexed images."""
        return len(self._index)

    def clear_index(self) -> None:
        """Remove all indexed embeddings."""
        self._index.clear()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Find images that best match *query*.

        Parameters
        ----------
        query : str
            Natural-language description of the desired image.
        top_k : int
            Maximum number of results to return.
        threshold : float
            Minimum cosine-similarity score (0-1) to include a result.

        Returns
        -------
        list[SearchResult]
            Matching images sorted by descending similarity.
        """
        self._assert_active()
        if not self._index:
            return []

        text_emb = self._encode_text(query)
        image_embs = torch.stack([r.embedding for r in self._index])  # (N, D)
        similarities = torch.nn.functional.cosine_similarity(text_emb, image_embs)

        scores, indices = similarities.topk(min(top_k, len(self._index)))
        results: List[SearchResult] = []
        for score_t, idx_t in zip(scores, indices):
            score = float(score_t)
            if score < threshold:
                continue
            results.append(SearchResult(
                path=str(self._index[int(idx_t)].path),
                score=round(score, 4),
            ))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(preference: str) -> str:
        if preference == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return preference

    def _assert_active(self) -> None:
        if not self._active or self._model is None or self._processor is None:
            raise RuntimeError(
                "Plugin is not active. Call activate() before using search or indexing."
            )

    @torch.inference_mode()
    def _encode_images(self, images: Sequence[Image.Image]) -> List[torch.Tensor]:
        assert self._processor is not None and self._model is not None
        inputs = self._processor(images=list(images), return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        embs = self._model.get_image_features(**inputs)  # (B, D)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return list(embs.cpu().unbind(0))

    @torch.inference_mode()
    def _encode_text(self, text: str) -> torch.Tensor:
        assert self._processor is not None and self._model is not None
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        embs = self._model.get_text_features(**inputs)  # (1, D)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu()  # (1, D)


# ------------------------------------------------------------------
# Module-level register / helpers for Homie plugin discovery
# ------------------------------------------------------------------

_instance: Optional[CLIPImageSearchPlugin] = None


def register(config: Optional[Dict[str, Any]] = None) -> CLIPImageSearchPlugin:
    """Create and return the plugin instance (does NOT activate yet)."""
    global _instance
    _instance = CLIPImageSearchPlugin(config)
    return _instance


def activate(config: Optional[Dict[str, Any]] = None) -> CLIPImageSearchPlugin:
    """Register and activate the plugin in one call."""
    plugin = register(config)
    plugin.activate()
    return plugin


def deactivate() -> None:
    """Deactivate the module-level instance if present."""
    global _instance
    if _instance is not None:
        _instance.deactivate()
        _instance = None
