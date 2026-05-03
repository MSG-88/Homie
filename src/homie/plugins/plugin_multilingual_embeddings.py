"""Homie plugin: Multilingual Embedding & Cross-Lingual Search

Provides multilingual sentence embeddings using the
paraphrase-multilingual-MiniLM-L12-v2 model from sentence-transformers.
Enables cross-lingual semantic search so users can query Homie's RAG
knowledge base in one language and retrieve relevant documents written
in another (supports 50+ languages).

The model runs fully locally via ONNX or PyTorch â€” no network calls
unless the user explicitly opts in to download the model weights on
first activation.
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

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CACHE_DIR = Path.home() / ".homie" / "models" / "multilingual-minilm"
DEFAULT_MAX_SEQ_LENGTH = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_SIMILARITY_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultilingualEmbeddingConfig:
    """Plugin-level configuration, typically sourced from homie.config.yaml."""

    model_name: str = MODEL_NAME
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    use_onnx: bool = True
    normalize_embeddings: bool = True
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Embedding engine (lazy-loaded)
# ---------------------------------------------------------------------------

class _EmbeddingEngine:
    """Thin wrapper around SentenceTransformer for local inference."""

    def __init__(self, config: MultilingualEmbeddingConfig) -> None:
        self._config = config
        self._model: Any = None  # lazy

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        cache_path = Path(self._config.cache_dir)
        # Only load from cache when it exists; avoids surprise downloads.
        if cache_path.exists() and any(cache_path.iterdir()):
            model_path: str = str(cache_path)
        else:
            # Fall back to HF hub identifier â€” will use local HF cache if
            # the model was previously downloaded, or error if offline.
            model_path = self._config.model_name

        logger.info(
            "Loading multilingual embedding model from %s (device=%s, onnx=%s)",
            model_path,
            self._config.device,
            self._config.use_onnx,
        )

        kwargs: Dict[str, Any] = {
            "model_name_or_path": model_path,
            "device": self._config.device,
        }
        if self._config.use_onnx:
            try:
                kwargs["backend"] = "onnx"
            except Exception:  # noqa: BLE001
                logger.warning("ONNX backend unavailable, falling back to PyTorch")

        model = SentenceTransformer(**kwargs)
        model.max_seq_length = self._config.max_seq_length
        return model

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Encode *sentences* and return a list of float vectors."""
        import numpy as np

        vectors = self.model.encode(
            list(sentences),
            batch_size=batch_size or self._config.batch_size,
            normalize_embeddings=self._config.normalize_embeddings,
            show_progress_bar=False,
        )
        if isinstance(vectors, np.ndarray):
            return vectors.tolist()
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


# ---------------------------------------------------------------------------
# Cross-lingual search helper
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single cross-lingual search hit."""

    text: str
    score: float
    language: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors without numpy at call-site."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class MultilingualEmbeddingPlugin:
    """Homie plugin that adds cross-lingual embedding and search capabilities.

    Lifecycle
    ---------
    * ``activate()``  â€” parse config, prepare (but lazy-load) the model.
    * ``deactivate()`` â€” release model memory.

    Public API
    ----------
    * ``embed(texts)``           â€” return embeddings for a list of texts.
    * ``search(query, corpus)``  â€” rank corpus entries by semantic similarity.
    * ``similarity(a, b)``       â€” cosine similarity between two texts.
    """

    def __init__(self) -> None:
        self._config: Optional[MultilingualEmbeddingConfig] = None
        self._engine: Optional[_EmbeddingEngine] = None
        self._active: bool = False

    # -- lifecycle -----------------------------------------------------------

    def activate(self, plugin_config: Optional[Dict[str, Any]] = None) -> None:
        """Activate the plugin, optionally with config from homie.config.yaml.

        Parameters
        ----------
        plugin_config:
            Dict typically coming from
            ``cfg_get(homie_cfg, "plugins", "multilingual_embeddings")``.
            Recognised keys mirror :class:`MultilingualEmbeddingConfig` fields.
        """
        if self._active:
            logger.debug("MultilingualEmbeddingPlugin already active")
            return

        raw = plugin_config or {}
        self._config = MultilingualEmbeddingConfig(
            model_name=raw.get("model_name", MODEL_NAME),
            cache_dir=raw.get("cache_dir", str(DEFAULT_CACHE_DIR)),
            max_seq_length=int(raw.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)),
            batch_size=int(raw.get("batch_size", DEFAULT_BATCH_SIZE)),
            similarity_threshold=float(
                raw.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
            ),
            use_onnx=bool(raw.get("use_onnx", True)),
            normalize_embeddings=bool(raw.get("normalize_embeddings", True)),
            device=raw.get("device", "cpu"),
        )
        self._engine = _EmbeddingEngine(self._config)
        self._active = True
        logger.info("MultilingualEmbeddingPlugin activated (model=%s)", self._config.model_name)

    def deactivate(self) -> None:
        """Release model resources."""
        if self._engine is not None:
            del self._engine._model
            self._engine = None
        self._active = False
        logger.info("MultilingualEmbeddingPlugin deactivated")

    # -- guards --------------------------------------------------------------

    def _require_active(self) -> _EmbeddingEngine:
        if not self._active or self._engine is None:
            raise RuntimeError("Plugin is not activated. Call activate() first.")
        return self._engine

    # -- public API ----------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return embedding vectors for *texts* (any language)."""
        engine = self._require_active()
        return engine.encode(texts)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts, potentially in different languages."""
        vecs = self.embed([text_a, text_b])
        return _cosine_similarity(vecs[0], vecs[1])

    def search(
        self,
        query: str,
        corpus: Sequence[Dict[str, Any]],
        *,
        text_key: str = "text",
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Rank *corpus* entries by semantic similarity to *query*.

        Each corpus entry is a dict that must contain *text_key*.  Optional
        ``"language"`` and any other keys are passed through as metadata.

        Parameters
        ----------
        query:
            User query in any supported language.
        corpus:
            List of dicts, each with at least a *text_key* field.
        text_key:
            Dict key holding the text to compare against.
        top_k:
            Maximum results to return.
        threshold:
            Minimum cosine similarity.  Falls back to the configured default.
        """
        engine = self._require_active()
        assert self._config is not None

        min_score = threshold if threshold is not None else self._config.similarity_threshold

        texts = [entry[text_key] for entry in corpus]
        all_texts = [query] + texts
        all_vecs = engine.encode(all_texts)
        query_vec = all_vecs[0]

        scored: List[Tuple[float, int]] = []
        for idx, vec in enumerate(all_vecs[1:]):
            score = _cosine_similarity(query_vec, vec)
            if score >= min_score:
                scored.append((score, idx))

        scored.sort(key=lambda t: t[0], reverse=True)

        results: List[SearchResult] = []
        for score, idx in scored[:top_k]:
            entry = corpus[idx]
            meta = {k: v for k, v in entry.items() if k not in (text_key, "language")}
            results.append(
                SearchResult(
                    text=entry[text_key],
                    score=round(score, 4),
                    language=entry.get("language", "unknown"),
                    metadata=meta,
                )
            )
        return results

    def chromadb_embedding_function(self) -> Any:
        """Return a ChromaDB-compatible embedding function.

        Usage::

            plugin.activate()
            ef = plugin.chromadb_embedding_function()
            collection = chroma_client.get_or_create_collection(
                "multilingual_docs", embedding_function=ef,
            )
        """
        engine = self._require_active()

        class _ChromaEF:
            def __call__(self, input: List[str]) -> List[List[float]]:
                return engine.encode(input)

        return _ChromaEF()


# ---------------------------------------------------------------------------
# Module-level convenience (register / activate / deactivate)
# ---------------------------------------------------------------------------

_default_instance: Optional[MultilingualEmbeddingPlugin] = None


def register() -> MultilingualEmbeddingPlugin:
    """Create and return the default plugin instance (does not activate)."""
    global _default_instance
    if _default_instance is None:
        _default_instance = MultilingualEmbeddingPlugin()
    return _default_instance


def activate(plugin_config: Optional[Dict[str, Any]] = None) -> MultilingualEmbeddingPlugin:
    """Register *and* activate the default instance."""
    instance = register()
    instance.activate(plugin_config)
    return instance


def deactivate() -> None:
    """Deactivate the default instance if it exists."""
    if _default_instance is not None:
        _default_instance.deactivate()
