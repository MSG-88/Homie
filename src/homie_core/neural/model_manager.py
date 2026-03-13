from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False


_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_CACHE = Path.home() / ".homie" / "models" / "neural"


class EmbeddingModel:
    """Manages a local sentence embedding model via ONNX Runtime.

    Lazy-loads on first use. Thread-safe. GPU-accelerated when available,
    falls back to CPU.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL,
                 cache_dir: Path | str | None = None):
        self._model_name = model_name
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        self._session = None
        self._tokenizer = None
        self._loaded = False
        self._dimension = 0
        self._lock = threading.Lock()

    def load(self, device: str = "auto") -> None:
        """Load the ONNX model and tokenizer."""
        if not _HAS_NUMPY:
            raise ImportError(
                "numpy is required for neural embeddings. "
                "Install with: pip install homie-ai[neural]"
            )
        with self._lock:
            if self._loaded:
                return

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            model_dir = self._download_model()

            self._session = self._load_onnx_session(model_dir, device)
            self._tokenizer = self._load_tokenizer(model_dir)

            # Determine embedding dimension from a test inference
            test_embedding = self._infer(["test"])
            self._dimension = test_embedding.shape[1]
            self._loaded = True

    def _download_model(self) -> Path:
        """Download model files if not cached."""
        model_dir = self._cache_dir / self._model_name.replace("/", "--")
        if model_dir.exists() and any(model_dir.glob("*.onnx")):
            return model_dir

        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=self._model_name,
                local_dir=str(model_dir),
                allow_patterns=["*.onnx", "*.json", "*.txt", "tokenizer*"],
            )
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required to download models. "
                "Install with: pip install huggingface_hub"
            )
        return model_dir

    def _load_onnx_session(self, model_dir: Path,
                           device: str = "auto"):
        """Create an ONNX Runtime inference session."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "onnxruntime is required. "
                "Install with: pip install onnxruntime"
            )

        onnx_path = None
        for candidate in ["model.onnx", "model_optimized.onnx"]:
            p = model_dir / candidate
            if p.exists():
                onnx_path = p
                break

        if not onnx_path:
            # Try onnx subdirectory
            onnx_dir = model_dir / "onnx"
            if onnx_dir.exists():
                for candidate in ["model.onnx", "model_optimized.onnx"]:
                    p = onnx_dir / candidate
                    if p.exists():
                        onnx_path = p
                        break

        if not onnx_path:
            raise FileNotFoundError(
                f"No ONNX model found in {model_dir}. "
                f"Ensure the model has an ONNX export."
            )

        providers = []
        if device == "auto" or device == "gpu":
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
            if "DmlExecutionProvider" in ort.get_available_providers():
                providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")

        session = ort.InferenceSession(str(onnx_path), providers=providers)
        return session

    def _load_tokenizer(self, model_dir: Path):
        """Load tokenizer from model directory."""
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise RuntimeError(
                "tokenizers is required. "
                "Install with: pip install tokenizers"
            )

        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found in {model_dir}"
            )
        return Tokenizer.from_file(str(tokenizer_path))

    def _infer(self, texts: list[str]) -> np.ndarray:
        """Run inference on a batch of texts."""
        encodings = [self._tokenizer.encode(t) for t in texts]

        max_len = max(len(e.ids) for e in encodings)
        input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)

        for i, enc in enumerate(encodings):
            length = len(enc.ids)
            input_ids[i, :length] = enc.ids
            attention_mask[i, :length] = enc.attention_mask

        input_names = [inp.name for inp in self._session.get_inputs()]
        feeds = {}
        if "input_ids" in input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in input_names:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, feeds)
        embeddings = outputs[0]

        # Mean pooling over token dimension
        if len(embeddings.shape) == 3:
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            summed = (embeddings * mask_expanded).sum(axis=1)
            counts = mask_expanded.sum(axis=1).clip(min=1e-9)
            embeddings = summed / counts

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a list of floats."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        with self._lock:
            result = self._infer([text])
        return result[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns list of float lists."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if not texts:
            return []
        with self._lock:
            result = self._infer(texts)
        return [row.tolist() for row in result]

    def unload(self) -> None:
        """Release model resources."""
        with self._lock:
            self._session = None
            self._tokenizer = None
            self._loaded = False
            self._dimension = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dimension(self) -> int:
        return self._dimension
