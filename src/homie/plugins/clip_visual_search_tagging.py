"""CLIP Visual Search & Tagging Plugin for Homie.

Provides zero-shot image classification, visual search, and automatic
image tagging using OpenAI's CLIP (ViT-Large/patch14) model running
locally. No network calls are made after initial model download.

Capabilities:
  - Classify images against arbitrary text labels (zero-shot)
  - Tag images with confidence scores from a configurable tag vocabulary
  - Search a local image collection by natural-language query
  - Build and query a persistent embedding index for fast retrieval

Requires: transformers, torch, Pillow
Optional: chromadb (for persistent vector search index)
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
        "clip_visual_search_tagging requires 'transformers', 'torch', and 'Pillow'. "
        "Install with: pip install transformers torch Pillow"
    ) from _exc

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-large-patch14"

DEFAULT_TAGS: List[str] = [
    "photo", "screenshot", "diagram", "document", "meme",
    "nature", "animal", "person", "food", "vehicle",
    "building", "art", "text", "chart", "map",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@dataclass
class ClassificationResult:
    """Result of a zero-shot classification or tagging operation."""
    label: str
    score: float


@dataclass
class SearchResult:
    """Result of a visual search query."""
    path: str
    score: float


class CLIPVisualPlugin:
    """Zero-shot image classification and visual search plugin for Homie.

    Loads the CLIP model lazily on first use and caches it for the
    lifetime of the plugin. All inference runs locally on CPU or CUDA.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        default_tags: Optional[List[str]] = None,
    ) -> None:
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._default_tags = default_tags or DEFAULT_TAGS

        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Activate the plugin and lazily prepare for inference."""
        self._active = True
        logger.info("CLIPVisualPlugin activated (model loads on first use)")

    def deactivate(self) -> None:
        """Release model resources."""
        self._model = None
        self._processor = None
        self._active = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CLIPVisualPlugin deactivated")

    def _ensure_model(self) -> None:
        """Load model and processor on first call."""
        if self._model is not None:
            return
        logger.info("Loading CLIP model %s on %s â€¦", self._model_id, self._device)
        self._processor = CLIPProcessor.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        )
        self._model = CLIPModel.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        ).to(self._device)
        self._model.eval()
        logger.info("CLIP model ready")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def classify(
        self,
        image_path: str,
        labels: Sequence[str],
        top_k: int = 5,
    ) -> List[ClassificationResult]:
        """Zero-shot classify an image against arbitrary text labels.

        Args:
            image_path: Path to a local image file.
            labels: Candidate text labels to score against.
            top_k: Number of top results to return.

        Returns:
            Sorted list of ``ClassificationResult`` (highest score first).
        """
        self._ensure_model()
        assert self._processor is not None and self._model is not None

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(
            text=list(labels), images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=-1).cpu().tolist()

        scored = [
            ClassificationResult(label=lbl, score=round(prob, 4))
            for lbl, prob in zip(labels, probs)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def tag(
        self,
        image_path: str,
        tags: Optional[List[str]] = None,
        threshold: float = 0.1,
    ) -> List[ClassificationResult]:
        """Auto-tag an image using the default or supplied tag vocabulary.

        Only tags whose confidence exceeds *threshold* are returned.
        """
        vocabulary = tags or self._default_tags
        results = self.classify(image_path, vocabulary, top_k=len(vocabulary))
        return [r for r in results if r.score >= threshold]

    def encode_image(self, image_path: str) -> List[float]:
        """Return the CLIP embedding vector for a single image."""
        self._ensure_model()
        assert self._processor is not None and self._model is not None

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = self._model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().tolist()

    def encode_text(self, text: str) -> List[float]:
        """Return the CLIP embedding vector for a text query."""
        self._ensure_model()
        assert self._processor is not None and self._model is not None

        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = self._model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().tolist()

    def search(
        self,
        query: str,
        image_dir: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search a directory of images by natural-language query.

        Encodes every image on-the-fly (no persistent index). For large
        collections use ``build_index`` + ``search_index`` with ChromaDB.

        Args:
            query: Natural-language description of what to find.
            image_dir: Directory containing image files.
            top_k: Number of results to return.
        """
        self._ensure_model()
        assert self._processor is not None and self._model is not None

        dir_path = Path(image_dir)
        image_paths = [
            p for p in dir_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_paths:
            return []

        # Encode query text
        text_emb = torch.tensor([self.encode_text(query)], device=self._device)

        scores: List[Tuple[str, float]] = []
        for img_path in image_paths:
            try:
                img_emb = torch.tensor(
                    [self.encode_image(str(img_path))], device=self._device
                )
                sim = torch.nn.functional.cosine_similarity(text_emb, img_emb).item()
                scores.append((str(img_path), sim))
            except Exception:
                logger.warning("Skipping unreadable image: %s", img_path)

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(path=path, score=round(score, 4))
            for path, score in scores[:top_k]
        ]

    # ------------------------------------------------------------------
    # ChromaDB-backed persistent index (opt-in)
    # ------------------------------------------------------------------

    def build_index(
        self,
        image_dir: str,
        collection_name: str = "homie_images",
        chroma_path: Optional[str] = None,
    ) -> int:
        """Index all images in a directory into a ChromaDB collection.

        Args:
            image_dir: Directory of images to index.
            collection_name: ChromaDB collection name.
            chroma_path: Persistent path for ChromaDB storage.

        Returns:
            Number of images indexed.
        """
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "Persistent index requires 'chromadb'. "
                "Install with: pip install chromadb"
            ) from exc

        settings = chromadb.Settings(anonymized_telemetry=False)
        if chroma_path:
            client = chromadb.PersistentClient(path=chroma_path, settings=settings)
        else:
            client = chromadb.Client(settings=settings)

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        dir_path = Path(image_dir)
        image_paths = [
            p for p in dir_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        count = 0
        for img_path in image_paths:
            str_path = str(img_path.resolve())
            try:
                embedding = self.encode_image(str_path)
                collection.upsert(
                    ids=[str_path],
                    embeddings=[embedding],
                    metadatas=[{"filename": img_path.name}],
                )
                count += 1
            except Exception:
                logger.warning("Failed to index: %s", img_path)

        logger.info("Indexed %d images into '%s'", count, collection_name)
        return count

    def search_index(
        self,
        query: str,
        collection_name: str = "homie_images",
        chroma_path: Optional[str] = None,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search the persistent ChromaDB index by natural-language query."""
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "Persistent index requires 'chromadb'. "
                "Install with: pip install chromadb"
            ) from exc

        settings = chromadb.Settings(anonymized_telemetry=False)
        if chroma_path:
            client = chromadb.PersistentClient(path=chroma_path, settings=settings)
        else:
            client = chromadb.Client(settings=settings)

        collection = client.get_collection(name=collection_name)
        query_emb = self.encode_text(query)

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
        )

        search_results: List[SearchResult] = []
        if results["ids"] and results["distances"]:
            for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
                score = round(1.0 - distance, 4)  # cosine distance â†’ similarity
                search_results.append(SearchResult(path=doc_id, score=score))

        return search_results


def register() -> Dict[str, Any]:
    """Register the plugin with Homie's plugin system."""
    return {
        "name": "clip_visual_search_tagging",
        "version": "1.0.0",
        "description": (
            "Zero-shot image classification, auto-tagging, and visual search "
            "powered by CLIP (ViT-Large/patch14), running fully local."
        ),
        "plugin_class": CLIPVisualPlugin,
        "capabilities": [
            "image_classification",
            "image_tagging",
            "visual_search",
            "image_embedding",
        ],
    }
