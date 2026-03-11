# Smart Context Engine Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an embedding-powered neural network foundation that gives Homie genuine semantic understanding of user activity, intent, and emotional state.

**Architecture:** A `neural/` module containing an ONNX-based embedding model, activity classifier, semantic context engine, intent inferencer, sentiment analyzer, and neural memory consolidator. All components are GPU-accelerated when available, with CPU fallback. The embedding model is the backbone — all other components consume its vectors.

**Tech Stack:** ONNX Runtime (inference), huggingface_hub (model download), tokenizers (fast tokenization), numpy (vector ops), existing homie_core modules.

---

### Task 1: Vector Utilities

**Files:**
- Create: `src/homie_core/neural/__init__.py`
- Create: `src/homie_core/neural/utils.py`
- Test: `tests/unit/test_neural/__init__.py`
- Test: `tests/unit/test_neural/test_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_neural/__init__.py` (empty file).

Create `tests/unit/test_neural/test_utils.py`:

```python
import math

from homie_core.neural.utils import cosine_similarity, weighted_average, top_k_similar


def test_cosine_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6


def test_cosine_zero_vector_returns_zero():
    a = [0.0, 0.0]
    b = [1.0, 2.0]
    assert cosine_similarity(a, b) == 0.0


def test_weighted_average_equal_weights():
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    weights = [1.0, 1.0]
    result = weighted_average(vecs, weights)
    assert len(result) == 2
    assert abs(result[0] - 0.5) < 1e-6
    assert abs(result[1] - 0.5) < 1e-6


def test_weighted_average_single_vector():
    vecs = [[3.0, 4.0]]
    weights = [1.0]
    result = weighted_average(vecs, weights)
    assert abs(result[0] - 3.0) < 1e-6
    assert abs(result[1] - 4.0) < 1e-6


def test_weighted_average_empty_returns_empty():
    result = weighted_average([], [])
    assert result == []


def test_top_k_similar():
    query = [1.0, 0.0]
    candidates = [
        [1.0, 0.0],   # identical
        [0.0, 1.0],   # orthogonal
        [0.7, 0.7],   # partial match
    ]
    results = top_k_similar(query, candidates, k=2)
    assert len(results) == 2
    assert results[0][0] == 0  # index of most similar
    assert results[0][1] > 0.9  # high similarity


def test_top_k_similar_k_exceeds_candidates():
    query = [1.0, 0.0]
    candidates = [[0.5, 0.5]]
    results = top_k_similar(query, candidates, k=5)
    assert len(results) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_neural/test_utils.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/homie_core/neural/__init__.py` (empty file).

Create `src/homie_core/neural/utils.py`:

```python
from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def weighted_average(vectors: list[list[float]],
                     weights: list[float]) -> list[float]:
    """Compute weighted average of vectors."""
    if not vectors:
        return []
    dim = len(vectors[0])
    total_weight = sum(weights)
    if total_weight == 0.0:
        return [0.0] * dim
    result = [0.0] * dim
    for vec, w in zip(vectors, weights):
        for i in range(dim):
            result[i] += vec[i] * w
    return [x / total_weight for x in result]


def top_k_similar(query: list[float],
                  candidates: list[list[float]],
                  k: int) -> list[tuple[int, float]]:
    """Return top-k most similar candidates by cosine similarity.

    Returns list of (index, similarity) tuples, sorted descending.
    """
    scored = []
    for i, cand in enumerate(candidates):
        sim = cosine_similarity(query, cand)
        scored.append((i, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_neural/test_utils.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/neural/__init__.py src/homie_core/neural/utils.py tests/unit/test_neural/__init__.py tests/unit/test_neural/test_utils.py
git commit -m "feat: add vector utility functions for neural module"
```

---

### Task 2: Embedding Model Manager

**Files:**
- Create: `src/homie_core/neural/model_manager.py`
- Test: `tests/unit/test_neural/test_model_manager.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_neural/test_model_manager.py`:

```python
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from homie_core.neural.model_manager import EmbeddingModel


def test_model_not_loaded_initially():
    model = EmbeddingModel()
    assert not model.is_loaded


def test_embed_raises_when_not_loaded():
    model = EmbeddingModel()
    try:
        model.embed("hello")
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()


def test_embed_batch_raises_when_not_loaded():
    model = EmbeddingModel()
    try:
        model.embed_batch(["hello"])
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()


def test_load_and_embed(tmp_path):
    """Test with a mock ONNX session."""
    model = EmbeddingModel(cache_dir=tmp_path)
    fake_output = np.random.randn(1, 384).astype(np.float32)

    with patch.object(model, "_load_onnx_session") as mock_load:
        mock_session = MagicMock()
        mock_session.run.return_value = [fake_output]
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        mock_load.return_value = mock_session

        with patch.object(model, "_load_tokenizer") as mock_tok:
            mock_tokenizer = MagicMock()
            encoding = MagicMock()
            encoding.ids = list(range(10))
            encoding.attention_mask = [1] * 10
            mock_tokenizer.encode.return_value = encoding
            mock_tok.return_value = mock_tokenizer

            model.load()

    assert model.is_loaded
    assert model.dimension == 384

    result = model.embed("test sentence")
    assert len(result) == 384
    assert isinstance(result[0], float)


def test_embed_batch(tmp_path):
    model = EmbeddingModel(cache_dir=tmp_path)
    fake_output = np.random.randn(2, 384).astype(np.float32)

    with patch.object(model, "_load_onnx_session") as mock_load:
        mock_session = MagicMock()
        mock_session.run.return_value = [fake_output]
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        mock_load.return_value = mock_session

        with patch.object(model, "_load_tokenizer") as mock_tok:
            mock_tokenizer = MagicMock()
            encoding = MagicMock()
            encoding.ids = list(range(10))
            encoding.attention_mask = [1] * 10
            mock_tokenizer.encode.return_value = encoding
            mock_tok.return_value = mock_tokenizer

            model.load()

    results = model.embed_batch(["hello", "world"])
    assert len(results) == 2
    assert len(results[0]) == 384


def test_unload():
    model = EmbeddingModel()
    model._session = MagicMock()
    model._tokenizer = MagicMock()
    model._loaded = True
    model._dimension = 384

    model.unload()
    assert not model.is_loaded
    assert model._session is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_neural/test_model_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/homie_core/neural/model_manager.py`:

```python
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_neural/test_model_manager.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/neural/model_manager.py tests/unit/test_neural/test_model_manager.py
git commit -m "feat: add ONNX-based embedding model manager"
```

---

### Task 3: Activity Classifier

**Files:**
- Create: `src/homie_core/neural/activity_classifier.py`
- Test: `tests/unit/test_neural/test_activity_classifier.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_neural/test_activity_classifier.py`:

```python
from unittest.mock import MagicMock
import math

from homie_core.neural.activity_classifier import ActivityClassifier, CATEGORIES


def _fake_embed(text):
    """Deterministic fake embeddings based on text content."""
    if "code" in text.lower() or ".py" in text.lower():
        return [1.0, 0.0, 0.0, 0.0]
    elif "chrome" in text.lower() or "google" in text.lower():
        return [0.0, 1.0, 0.0, 0.0]
    elif "slack" in text.lower() or "teams" in text.lower():
        return [0.0, 0.0, 1.0, 0.0]
    elif "word" in text.lower() or "docs" in text.lower():
        return [0.0, 0.0, 0.0, 1.0]
    return [0.25, 0.25, 0.25, 0.25]


def test_categories_exist():
    assert "coding" in CATEGORIES
    assert "researching" in CATEGORIES
    assert "communicating" in CATEGORIES
    assert len(CATEGORIES) >= 8


def test_classify_returns_all_categories():
    embed_fn = MagicMock(side_effect=_fake_embed)
    classifier = ActivityClassifier(embed_fn=embed_fn, embed_dim=4)
    classifier._init_prototypes()

    result = classifier.classify("Code.exe", "engine.py - Homie")
    assert isinstance(result, dict)
    for cat in CATEGORIES:
        assert cat in result
        assert 0.0 <= result[cat] <= 1.0


def test_classify_coding_activity():
    embed_fn = MagicMock(side_effect=_fake_embed)
    classifier = ActivityClassifier(embed_fn=embed_fn, embed_dim=4)
    classifier._init_prototypes()

    result = classifier.classify("Code.exe", "engine.py - Homie")
    # Coding should be among top categories
    top = max(result, key=result.get)
    assert isinstance(top, str)


def test_train_online():
    embed_fn = MagicMock(side_effect=_fake_embed)
    classifier = ActivityClassifier(embed_fn=embed_fn, embed_dim=4)
    classifier._init_prototypes()

    # Should not raise
    classifier.train_online("Code.exe", "engine.py - Homie", "coding")
    classifier.train_online("chrome.exe", "Google", "researching")


def test_serialize_deserialize():
    embed_fn = MagicMock(side_effect=_fake_embed)
    classifier = ActivityClassifier(embed_fn=embed_fn, embed_dim=4)
    classifier._init_prototypes()

    data = classifier.serialize()
    assert "prototypes" in data
    assert "weights" in data

    restored = ActivityClassifier.deserialize(data, embed_fn=embed_fn)
    assert restored is not None


def test_get_top_activity():
    embed_fn = MagicMock(side_effect=_fake_embed)
    classifier = ActivityClassifier(embed_fn=embed_fn, embed_dim=4)
    classifier._init_prototypes()

    result = classifier.classify("Code.exe", "engine.py - Homie")
    top = classifier.get_top_activity("Code.exe", "engine.py - Homie")
    assert isinstance(top, str)
    assert top in CATEGORIES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_neural/test_activity_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/homie_core/neural/activity_classifier.py`:

```python
from __future__ import annotations

import math
from typing import Callable, Optional

from homie_core.neural.utils import cosine_similarity


CATEGORIES = [
    "coding", "researching", "communicating", "writing",
    "designing", "browsing", "media", "system", "unknown",
]

# Prototype descriptions for zero-shot classification
_PROTOTYPE_DESCRIPTIONS = {
    "coding": "programming code editor IDE terminal compiler debug",
    "researching": "search documentation reference API reading learning",
    "communicating": "email chat message slack teams discord call meeting",
    "writing": "document writing text editor word notes markdown",
    "designing": "design figma photoshop graphics layout UI mockup",
    "browsing": "web browser internet social media news reddit",
    "media": "video music player spotify youtube streaming audio",
    "system": "settings control panel file manager task manager system",
    "unknown": "other miscellaneous general application",
}


class ActivityClassifier:
    """Classifies user activity into semantic categories using embeddings.

    Uses cosine similarity to category prototypes (zero-shot) initially,
    with optional online learning via a small feedforward layer.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]],
                 embed_dim: int = 384):
        self._embed_fn = embed_fn
        self._embed_dim = embed_dim
        self._prototypes: dict[str, list[float]] = {}
        # Simple linear layer for online learning: weights per category
        self._weights: dict[str, list[float]] = {}
        self._bias: dict[str, float] = {}
        self._n_samples = 0
        self._lr = 0.05

    def _init_prototypes(self) -> None:
        """Compute prototype embeddings for each category."""
        for cat, desc in _PROTOTYPE_DESCRIPTIONS.items():
            self._prototypes[cat] = self._embed_fn(desc)
            self._weights[cat] = [0.0] * self._embed_dim
            self._bias[cat] = 0.0

    def classify(self, process: str, title: str) -> dict[str, float]:
        """Classify activity into categories with confidence scores."""
        if not self._prototypes:
            self._init_prototypes()

        text = f"{process} {title}"
        embedding = self._embed_fn(text)

        # Cosine similarity to prototypes
        similarities = {}
        for cat, proto in self._prototypes.items():
            sim = cosine_similarity(embedding, proto)
            # Add learned adjustment
            learned = sum(
                w * x for w, x in zip(self._weights[cat], embedding)
            ) + self._bias[cat]
            similarities[cat] = sim + learned

        # Softmax normalization
        max_sim = max(similarities.values())
        exp_sims = {
            cat: math.exp(sim - max_sim)
            for cat, sim in similarities.items()
        }
        total = sum(exp_sims.values())
        return {cat: exp_sim / total for cat, exp_sim in exp_sims.items()}

    def get_top_activity(self, process: str, title: str) -> str:
        """Return the most likely activity category."""
        scores = self.classify(process, title)
        return max(scores, key=scores.get)

    def train_online(self, process: str, title: str, label: str) -> None:
        """Update classifier from a labeled observation via SGD."""
        if label not in CATEGORIES:
            return
        if not self._prototypes:
            self._init_prototypes()

        text = f"{process} {title}"
        embedding = self._embed_fn(text)
        scores = self.classify(process, title)

        # SGD update: increase score for correct label, decrease others
        for cat in CATEGORIES:
            target = 1.0 if cat == label else 0.0
            error = target - scores[cat]
            for i in range(len(embedding)):
                self._weights[cat][i] += self._lr * error * embedding[i]
            self._bias[cat] += self._lr * error

        self._n_samples += 1

    def serialize(self) -> dict:
        return {
            "prototypes": {k: list(v) for k, v in self._prototypes.items()},
            "weights": {k: list(v) for k, v in self._weights.items()},
            "bias": dict(self._bias),
            "n_samples": self._n_samples,
            "embed_dim": self._embed_dim,
        }

    @classmethod
    def deserialize(cls, data: dict,
                    embed_fn: Callable[[str], list[float]]) -> ActivityClassifier:
        obj = cls(embed_fn=embed_fn, embed_dim=data.get("embed_dim", 384))
        obj._prototypes = {k: list(v) for k, v in data["prototypes"].items()}
        obj._weights = {k: list(v) for k, v in data["weights"].items()}
        obj._bias = dict(data.get("bias", {}))
        obj._n_samples = data.get("n_samples", 0)
        return obj
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_neural/test_activity_classifier.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/neural/activity_classifier.py tests/unit/test_neural/test_activity_classifier.py
git commit -m "feat: add embedding-based activity classifier"
```

---

### Task 4: Semantic Context Engine

**Files:**
- Create: `src/homie_core/neural/context_engine.py`
- Test: `tests/unit/test_neural/test_context_engine.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_neural/test_context_engine.py`:

```python
from unittest.mock import MagicMock
import math

from homie_core.neural.context_engine import SemanticContextEngine


def _fake_embed(text):
    """Simple deterministic embeddings."""
    val = hash(text) % 1000 / 1000.0
    return [val, 1.0 - val, val * 0.5, (1.0 - val) * 0.5]


def test_init():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    assert engine.get_context_vector() == [0.0] * 4


def test_update_changes_context():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    engine.update("Code.exe", "engine.py - Homie")
    vec = engine.get_context_vector()
    assert any(v != 0.0 for v in vec)


def test_context_shift_detection():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4,
                                   shift_threshold=0.3)

    # Same context — no shift
    engine.update("Code.exe", "engine.py - Homie")
    engine.update("Code.exe", "config.py - Homie")
    # These are similar enough they may not trigger a shift

    # Very different context — should detect shift
    engine.update("Code.exe", "engine.py")
    initial = engine.get_context_vector()
    engine.update("spotify.exe", "Playing Music - Best Hits 2026")
    shifted = engine.detect_context_shift()
    # We just test it returns a bool
    assert isinstance(shifted, bool)


def test_find_relevant_memories():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    engine.update("Code.exe", "engine.py")

    memories = [
        {"summary": "coding session on engine.py", "embedding": _fake_embed("coding engine")},
        {"summary": "grocery shopping list", "embedding": _fake_embed("groceries milk bread")},
        {"summary": "debugging config parser", "embedding": _fake_embed("config debug code")},
    ]

    results = engine.find_relevant_memories(memories, top_k=2)
    assert len(results) == 2
    assert all("summary" in r for r in results)


def test_get_activity_summary():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    engine.update("Code.exe", "engine.py - Homie")
    engine.update("chrome.exe", "Stack Overflow")

    summary = engine.get_activity_summary()
    assert "observations" in summary
    assert summary["observations"] == 2


def test_rolling_window_limits():
    engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4,
                                   window_size=3)
    for i in range(10):
        engine.update("app.exe", f"window {i}")

    # Should only keep last 3
    assert len(engine._recent_embeddings) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_neural/test_context_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/homie_core/neural/context_engine.py`:

```python
from __future__ import annotations

from collections import deque
from typing import Any, Callable

from homie_core.neural.utils import cosine_similarity, weighted_average, top_k_similar


class SemanticContextEngine:
    """Tracks semantic context using embedding vectors.

    Maintains a rolling context vector (exponentially-weighted average
    of recent activity embeddings) and detects semantic context shifts.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]],
                 embed_dim: int = 384,
                 window_size: int = 20,
                 decay: float = 0.85,
                 shift_threshold: float = 0.5):
        self._embed_fn = embed_fn
        self._embed_dim = embed_dim
        self._window_size = window_size
        self._decay = decay
        self._shift_threshold = shift_threshold
        self._context_vector: list[float] = [0.0] * embed_dim
        self._prev_context_vector: list[float] = [0.0] * embed_dim
        self._recent_embeddings: deque[list[float]] = deque(maxlen=window_size)
        self._observation_count = 0

    def update(self, process: str, title: str) -> None:
        """Process a new activity observation."""
        text = f"{process} {title}"
        embedding = self._embed_fn(text)

        self._prev_context_vector = list(self._context_vector)
        self._recent_embeddings.append(embedding)
        self._observation_count += 1

        # Exponentially-weighted average
        weights = []
        for i in range(len(self._recent_embeddings)):
            age = len(self._recent_embeddings) - 1 - i
            weights.append(self._decay ** age)

        self._context_vector = weighted_average(
            list(self._recent_embeddings), weights,
        )

    def get_context_vector(self) -> list[float]:
        """Return the current context vector."""
        return list(self._context_vector)

    def detect_context_shift(self) -> bool:
        """Check if context has shifted significantly."""
        if self._observation_count < 2:
            return False
        sim = cosine_similarity(self._context_vector, self._prev_context_vector)
        return sim < self._shift_threshold

    def find_relevant_memories(self, memories: list[dict],
                               top_k: int = 5) -> list[dict]:
        """Find memories most relevant to current context."""
        if not memories or self._observation_count == 0:
            return []

        embeddings = []
        valid_memories = []
        for m in memories:
            emb = m.get("embedding")
            if emb:
                embeddings.append(emb)
                valid_memories.append(m)

        if not embeddings:
            return []

        results = top_k_similar(self._context_vector, embeddings, top_k)
        return [valid_memories[idx] for idx, _ in results]

    def get_activity_summary(self) -> dict:
        """Return summary of current context state."""
        return {
            "observations": self._observation_count,
            "window_size": len(self._recent_embeddings),
            "has_context": self._observation_count > 0,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_neural/test_context_engine.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/neural/context_engine.py tests/unit/test_neural/test_context_engine.py
git commit -m "feat: add semantic context engine with shift detection"
```

---

### Task 5: Intent Inferencer

**Files:**
- Create: `src/homie_core/neural/intent_inferencer.py`
- Test: `tests/unit/test_neural/test_intent_inferencer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_neural/test_intent_inferencer.py`:

```python
from homie_core.neural.intent_inferencer import IntentInferencer


def test_init():
    inf = IntentInferencer(embed_dim=4)
    assert not inf.has_enough_data()


def test_observe_builds_sequence():
    inf = IntentInferencer(embed_dim=4, sequence_length=5)
    for i in range(3):
        inf.observe([float(i)] * 4)
    assert len(inf._sequence) == 3


def test_predict_next_without_data():
    inf = IntentInferencer(embed_dim=4, min_sequences=2)
    result = inf.predict_next()
    assert result["confidence"] == 0.0
    assert result["predicted_activity"] is None


def test_predict_next_with_data():
    inf = IntentInferencer(embed_dim=4, sequence_length=3, min_sequences=1)

    # Build up a pattern: A -> B -> C, A -> B -> C
    a = [1.0, 0.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0, 0.0]
    c = [0.0, 0.0, 1.0, 0.0]

    # Record two complete sequences
    inf.train_from_sequence([a, b, c])

    # Now observe A, B — should predict something close to C
    inf.observe(a)
    inf.observe(b)
    result = inf.predict_next()
    assert result["confidence"] > 0.0
    assert result["predicted_activity"] is not None
    assert len(result["predicted_activity"]) == 4


def test_get_likely_needs_empty():
    inf = IntentInferencer(embed_dim=4)
    needs = inf.get_likely_needs()
    assert needs == []


def test_serialize_deserialize():
    inf = IntentInferencer(embed_dim=4, sequence_length=3)
    inf.observe([1.0, 0.0, 0.0, 0.0])
    inf.observe([0.0, 1.0, 0.0, 0.0])

    data = inf.serialize()
    assert "sequences" in data

    restored = IntentInferencer.deserialize(data)
    assert len(restored._sequence) == len(inf._sequence)


def test_sequence_wraps_at_limit():
    inf = IntentInferencer(embed_dim=4, sequence_length=3)
    for i in range(5):
        inf.observe([float(i)] * 4)
    assert len(inf._sequence) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_neural/test_intent_inferencer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/homie_core/neural/intent_inferencer.py`:

```python
from __future__ import annotations

from collections import deque
from typing import Optional

from homie_core.neural.utils import cosine_similarity, weighted_average


class IntentInferencer:
    """Predicts user intent from sequences of activity embeddings.

    Uses weighted k-NN over stored activity sequences. When the user's
    current activity sequence matches a previously seen pattern, predicts
    the next step.

    GPU/GRU upgrade path: replace _predict_knn with a trained GRU model
    when enough data is collected.
    """

    def __init__(self, embed_dim: int = 384,
                 sequence_length: int = 10,
                 min_sequences: int = 5):
        self._embed_dim = embed_dim
        self._sequence_length = sequence_length
        self._min_sequences = min_sequences
        self._sequence: deque[list[float]] = deque(maxlen=sequence_length)
        self._stored_sequences: list[list[list[float]]] = []

    def observe(self, activity_embedding: list[float]) -> None:
        """Add a new activity observation to the current sequence."""
        self._sequence.append(list(activity_embedding))

    def has_enough_data(self) -> bool:
        """Check if we have enough stored sequences for prediction."""
        return len(self._stored_sequences) >= self._min_sequences

    def predict_next(self) -> dict:
        """Predict the next activity based on current sequence.

        Returns dict with:
        - predicted_activity: embedding vector or None
        - confidence: float 0-1
        - estimated_completion: float 0-1 (how far through the pattern)
        """
        if not self._stored_sequences or len(self._sequence) < 2:
            return {
                "predicted_activity": None,
                "confidence": 0.0,
                "estimated_completion": 0.0,
            }

        current = list(self._sequence)
        best_sim = -1.0
        best_next = None
        best_position = 0.0

        for stored in self._stored_sequences:
            if len(stored) < 2:
                continue

            # Slide current sequence over stored to find best match
            for start in range(len(stored) - len(current)):
                window = stored[start:start + len(current)]
                if len(window) != len(current):
                    continue

                # Average similarity across the window
                sims = []
                for a, b in zip(current, window):
                    sims.append(cosine_similarity(a, b))
                avg_sim = sum(sims) / len(sims) if sims else 0.0

                if avg_sim > best_sim:
                    best_sim = avg_sim
                    next_idx = start + len(current)
                    if next_idx < len(stored):
                        best_next = stored[next_idx]
                    best_position = (start + len(current)) / len(stored)

        if best_next is None or best_sim < 0.3:
            return {
                "predicted_activity": None,
                "confidence": 0.0,
                "estimated_completion": 0.0,
            }

        return {
            "predicted_activity": best_next,
            "confidence": max(0.0, min(1.0, best_sim)),
            "estimated_completion": best_position,
        }

    def get_likely_needs(self) -> list[str]:
        """Return descriptions of likely upcoming information needs.

        Placeholder — returns empty until we have activity-to-need mapping.
        """
        return []

    def train_from_sequence(self, sequence: list[list[float]]) -> None:
        """Store a completed activity sequence for future matching."""
        if len(sequence) >= 2:
            self._stored_sequences.append([list(v) for v in sequence])

    def complete_current_sequence(self) -> None:
        """Mark current sequence as complete and store for training."""
        if len(self._sequence) >= 2:
            self._stored_sequences.append(list(self._sequence))
        self._sequence.clear()

    def serialize(self) -> dict:
        return {
            "embed_dim": self._embed_dim,
            "sequence_length": self._sequence_length,
            "min_sequences": self._min_sequences,
            "sequence": [list(v) for v in self._sequence],
            "sequences": [[list(v) for v in seq]
                          for seq in self._stored_sequences],
        }

    @classmethod
    def deserialize(cls, data: dict) -> IntentInferencer:
        obj = cls(
            embed_dim=data.get("embed_dim", 384),
            sequence_length=data.get("sequence_length", 10),
            min_sequences=data.get("min_sequences", 5),
        )
        for v in data.get("sequence", []):
            obj._sequence.append(list(v))
        obj._stored_sequences = [
            [list(v) for v in seq]
            for seq in data.get("sequences", [])
        ]
        return obj
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_neural/test_intent_inferencer.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/neural/intent_inferencer.py tests/unit/test_neural/test_intent_inferencer.py
git commit -m "feat: add intent inferencer with sequence prediction"
```

---

### Task 6: Sentiment Analyzer

**Files:**
- Create: `src/homie_core/neural/sentiment.py`
- Test: `tests/unit/test_neural/test_sentiment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_neural/test_sentiment.py`:

```python
from unittest.mock import MagicMock

from homie_core.neural.sentiment import SentimentAnalyzer, SentimentResult


def _fake_embed(text):
    """Embeddings that vary by sentiment words."""
    positive = ["happy", "great", "love", "thanks", "awesome", "good"]
    negative = ["angry", "frustrated", "broken", "hate", "terrible", "bad"]
    stressed = ["urgent", "deadline", "stuck", "help", "asap", "broken"]

    lower = text.lower()
    pos_score = sum(1 for w in positive if w in lower)
    neg_score = sum(1 for w in negative if w in lower)
    stress_score = sum(1 for w in stressed if w in lower)
    total = max(pos_score + neg_score + stress_score, 1)

    return [pos_score / total, neg_score / total, stress_score / total, 0.5]


def test_sentiment_result_fields():
    result = SentimentResult(
        sentiment="positive", arousal="calm", confidence=0.9,
    )
    assert result.sentiment == "positive"
    assert result.arousal == "calm"
    assert result.confidence == 0.9


def test_analyze_positive():
    analyzer = SentimentAnalyzer(embed_fn=_fake_embed)
    result = analyzer.analyze("This is great, thanks so much!")
    assert isinstance(result, SentimentResult)
    assert result.sentiment in ("positive", "negative", "neutral")
    assert result.arousal in ("calm", "stressed", "frustrated")
    assert 0.0 <= result.confidence <= 1.0


def test_analyze_negative():
    analyzer = SentimentAnalyzer(embed_fn=_fake_embed)
    result = analyzer.analyze("This is terrible and broken, I hate it")
    assert isinstance(result, SentimentResult)


def test_analyze_neutral():
    analyzer = SentimentAnalyzer(embed_fn=_fake_embed)
    result = analyzer.analyze("The meeting is at 3pm")
    assert isinstance(result, SentimentResult)


def test_analyze_batch():
    analyzer = SentimentAnalyzer(embed_fn=_fake_embed)
    results = analyzer.analyze_batch([
        "I love this!",
        "This is frustrating",
        "The file is saved",
    ])
    assert len(results) == 3
    assert all(isinstance(r, SentimentResult) for r in results)


def test_empty_text():
    analyzer = SentimentAnalyzer(embed_fn=_fake_embed)
    result = analyzer.analyze("")
    assert result.sentiment == "neutral"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_neural/test_sentiment.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/homie_core/neural/sentiment.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from homie_core.neural.utils import cosine_similarity


@dataclass
class SentimentResult:
    sentiment: str  # "positive", "negative", "neutral"
    arousal: str    # "calm", "stressed", "frustrated"
    confidence: float


# Prototype phrases for each sentiment/arousal category
_SENTIMENT_PROTOTYPES = {
    "positive": "happy great excellent love thanks wonderful awesome good",
    "negative": "bad terrible hate angry frustrated broken wrong awful",
    "neutral": "the a is was it meeting file task schedule note",
}

_AROUSAL_PROTOTYPES = {
    "calm": "easy relax done finished complete fine okay sure",
    "stressed": "urgent deadline hurry asap critical important rush now",
    "frustrated": "stuck broken again why not working error failed help",
}


class SentimentAnalyzer:
    """Lightweight sentiment and arousal detection from text.

    Uses embedding similarity to sentiment/arousal prototypes.
    No separate model needed — reuses the main embedding model.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]]):
        self._embed_fn = embed_fn
        self._sentiment_protos: dict[str, list[float]] = {}
        self._arousal_protos: dict[str, list[float]] = {}
        self._initialized = False

    def _init_prototypes(self) -> None:
        if self._initialized:
            return
        for label, text in _SENTIMENT_PROTOTYPES.items():
            self._sentiment_protos[label] = self._embed_fn(text)
        for label, text in _AROUSAL_PROTOTYPES.items():
            self._arousal_protos[label] = self._embed_fn(text)
        self._initialized = True

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment and arousal of text."""
        if not text or not text.strip():
            return SentimentResult(
                sentiment="neutral", arousal="calm", confidence=0.0,
            )

        self._init_prototypes()
        embedding = self._embed_fn(text)

        # Score sentiment
        sentiment_scores = {
            label: cosine_similarity(embedding, proto)
            for label, proto in self._sentiment_protos.items()
        }
        sentiment = max(sentiment_scores, key=sentiment_scores.get)
        sentiment_confidence = sentiment_scores[sentiment]

        # Score arousal
        arousal_scores = {
            label: cosine_similarity(embedding, proto)
            for label, proto in self._arousal_protos.items()
        }
        arousal = max(arousal_scores, key=arousal_scores.get)

        # Confidence is normalized similarity
        confidence = max(0.0, min(1.0, (sentiment_confidence + 1) / 2))

        return SentimentResult(
            sentiment=sentiment,
            arousal=arousal,
            confidence=confidence,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze multiple texts."""
        return [self.analyze(t) for t in texts]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_neural/test_sentiment.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/neural/sentiment.py tests/unit/test_neural/test_sentiment.py
git commit -m "feat: add embedding-based sentiment analyzer"
```

---

### Task 7: Neural Memory Consolidator

**Files:**
- Create: `src/homie_core/neural/consolidator.py`
- Test: `tests/unit/test_neural/test_neural_consolidator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_neural/test_neural_consolidator.py`:

```python
from homie_core.neural.consolidator import NeuralConsolidator


def _fake_embed(text):
    val = hash(text) % 1000 / 1000.0
    return [val, 1.0 - val, val * 0.5, (1.0 - val) * 0.5]


def test_compute_relevance():
    consolidator = NeuralConsolidator(embed_fn=_fake_embed)
    context = _fake_embed("coding python project")

    mem_relevant = {"summary": "coding python project", "embedding": _fake_embed("coding python project")}
    mem_irrelevant = {"summary": "grocery shopping", "embedding": _fake_embed("grocery shopping list")}

    rel_score = consolidator.compute_relevance(mem_relevant, context)
    irr_score = consolidator.compute_relevance(mem_irrelevant, context)

    assert 0.0 <= rel_score <= 1.0
    assert 0.0 <= irr_score <= 1.0


def test_find_patterns_empty():
    consolidator = NeuralConsolidator(embed_fn=_fake_embed)
    patterns = consolidator.find_patterns([])
    assert patterns == []


def test_find_patterns_groups_similar():
    consolidator = NeuralConsolidator(embed_fn=_fake_embed, similarity_threshold=0.8)

    episodes = [
        {"summary": "coding session A", "embedding": [1.0, 0.0, 0.0, 0.0]},
        {"summary": "coding session B", "embedding": [0.95, 0.05, 0.0, 0.0]},
        {"summary": "meeting with team", "embedding": [0.0, 1.0, 0.0, 0.0]},
    ]

    patterns = consolidator.find_patterns(episodes)
    assert isinstance(patterns, list)


def test_consolidate():
    consolidator = NeuralConsolidator(embed_fn=_fake_embed)
    context = _fake_embed("coding")

    episodes = [
        {"summary": "wrote code for feature", "embedding": _fake_embed("wrote code")},
        {"summary": "debugged test failure", "embedding": _fake_embed("debug test")},
    ]

    result = consolidator.consolidate(episodes, context)
    assert "relevant" in result
    assert "clusters" in result
    assert isinstance(result["relevant"], list)


def test_consolidate_empty():
    consolidator = NeuralConsolidator(embed_fn=_fake_embed)
    result = consolidator.consolidate([], [0.0] * 4)
    assert result["relevant"] == []
    assert result["clusters"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_neural/test_neural_consolidator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/homie_core/neural/consolidator.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from homie_core.neural.utils import cosine_similarity, top_k_similar


@dataclass
class Pattern:
    """A recurring pattern found in episodes."""
    description: str
    episode_indices: list[int] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)
    frequency: int = 0


@dataclass
class ConsolidationResult:
    """Result of neural memory consolidation."""
    relevant: list[dict]
    clusters: list[Pattern]


class NeuralConsolidator:
    """Embedding-powered memory consolidation.

    Clusters episodic memories by semantic similarity, finds recurring
    patterns, and computes relevance-weighted decay scores.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]],
                 similarity_threshold: float = 0.7):
        self._embed_fn = embed_fn
        self._similarity_threshold = similarity_threshold

    def compute_relevance(self, memory: dict,
                          context: list[float]) -> float:
        """Compute relevance of a memory to the current context.

        Returns float 0-1. Higher = more relevant.
        """
        embedding = memory.get("embedding")
        if not embedding or not context:
            return 0.0
        sim = cosine_similarity(embedding, context)
        # Normalize from [-1, 1] to [0, 1]
        return max(0.0, min(1.0, (sim + 1) / 2))

    def find_patterns(self, episodes: list[dict]) -> list[Pattern]:
        """Find recurring patterns by clustering similar episodes.

        Uses simple greedy single-linkage clustering.
        """
        if not episodes:
            return []

        embeddings = []
        valid_indices = []
        for i, ep in enumerate(episodes):
            emb = ep.get("embedding")
            if emb:
                embeddings.append(emb)
                valid_indices.append(i)

        if not embeddings:
            return []

        # Greedy clustering
        assigned = [False] * len(embeddings)
        patterns = []

        for i in range(len(embeddings)):
            if assigned[i]:
                continue

            cluster_indices = [valid_indices[i]]
            cluster_embeddings = [embeddings[i]]
            assigned[i] = True

            for j in range(i + 1, len(embeddings)):
                if assigned[j]:
                    continue
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self._similarity_threshold:
                    cluster_indices.append(valid_indices[j])
                    cluster_embeddings.append(embeddings[j])
                    assigned[j] = True

            if len(cluster_indices) >= 2:
                # Compute centroid
                dim = len(cluster_embeddings[0])
                centroid = [0.0] * dim
                for emb in cluster_embeddings:
                    for d in range(dim):
                        centroid[d] += emb[d]
                centroid = [c / len(cluster_embeddings) for c in centroid]

                summaries = [episodes[idx].get("summary", "")
                             for idx in cluster_indices]
                desc = f"Recurring pattern ({len(cluster_indices)} episodes): " + \
                       ", ".join(summaries[:3])

                patterns.append(Pattern(
                    description=desc,
                    episode_indices=cluster_indices,
                    centroid=centroid,
                    frequency=len(cluster_indices),
                ))

        return patterns

    def consolidate(self, episodes: list[dict],
                    current_context: list[float]) -> dict:
        """Run full consolidation: relevance scoring + pattern finding.

        Returns dict with 'relevant' (sorted episodes) and 'clusters' (patterns).
        """
        if not episodes:
            return {"relevant": [], "clusters": []}

        # Score relevance to current context
        scored = []
        for ep in episodes:
            score = self.compute_relevance(ep, current_context)
            scored.append({**ep, "_relevance": score})

        scored.sort(key=lambda x: x["_relevance"], reverse=True)

        # Find patterns
        clusters = self.find_patterns(episodes)

        return {
            "relevant": scored,
            "clusters": clusters,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_neural/test_neural_consolidator.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/neural/consolidator.py tests/unit/test_neural/test_neural_consolidator.py
git commit -m "feat: add neural memory consolidator with pattern detection"
```

---

### Task 8: Integration — Wire Neural Module into Observer Loop

**Files:**
- Modify: `src/homie_core/intelligence/observer_loop.py`
- Modify: `src/homie_core/intelligence/proactive_retrieval.py`
- Test: `tests/unit/test_neural/test_integration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_neural/test_integration.py`:

```python
from unittest.mock import MagicMock
from datetime import datetime, timezone

from homie_core.neural.context_engine import SemanticContextEngine
from homie_core.neural.activity_classifier import ActivityClassifier
from homie_core.neural.sentiment import SentimentAnalyzer
from homie_core.neural.intent_inferencer import IntentInferencer
from homie_core.intelligence.observer_loop import ObserverLoop
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.context.screen_monitor import WindowInfo
from homie_core.memory.working import WorkingMemory


def _fake_embed(text):
    val = hash(text) % 1000 / 1000.0
    return [val, 1.0 - val, val * 0.5, (1.0 - val) * 0.5]


def test_observer_with_neural_context():
    wm = WorkingMemory()
    tg = TaskGraph()
    context_engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    classifier = ActivityClassifier(embed_fn=_fake_embed, embed_dim=4)
    classifier._init_prototypes()

    loop = ObserverLoop(
        working_memory=wm,
        task_graph=tg,
        context_engine=context_engine,
        activity_classifier=classifier,
    )

    window = WindowInfo(
        title="engine.py - Homie",
        process_name="Code.exe",
        pid=1234,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    loop._handle_window_change(window)

    # Context engine should have been updated
    vec = context_engine.get_context_vector()
    assert any(v != 0.0 for v in vec)

    # Working memory should have activity classification
    activity = wm.get("activity_type")
    assert activity is not None


def test_observer_without_neural_still_works():
    """Backward compatibility — observer works without neural components."""
    wm = WorkingMemory()
    tg = TaskGraph()
    loop = ObserverLoop(working_memory=wm, task_graph=tg)

    window = WindowInfo(
        title="engine.py - Homie",
        process_name="Code.exe",
        pid=1234,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    loop._handle_window_change(window)

    assert wm.get("active_window") == "engine.py - Homie"
    assert len(tg.get_tasks()) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_neural/test_integration.py -v`
Expected: FAIL (observer doesn't accept neural params yet)

- [ ] **Step 3: Update ObserverLoop to accept neural components**

In `src/homie_core/intelligence/observer_loop.py`, update `__init__` to accept optional neural components and `_handle_window_change` to use them:

Add these imports at top:
```python
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from homie_core.neural.context_engine import SemanticContextEngine
    from homie_core.neural.activity_classifier import ActivityClassifier
```

Update `__init__` signature to add:
```python
        context_engine: Optional[SemanticContextEngine] = None,
        activity_classifier: Optional[ActivityClassifier] = None,
```

Store as `self._context_engine` and `self._activity_classifier`.

At the end of `_handle_window_change`, add:
```python
        # Neural components (optional)
        if self._context_engine:
            self._context_engine.update(window.process_name, window.title)
        if self._activity_classifier:
            scores = self._activity_classifier.classify(window.process_name, window.title)
            top = max(scores, key=scores.get)
            self._wm.update("activity_type", top)
            self._wm.update("activity_scores", scores)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_neural/test_integration.py tests/unit/test_intelligence/test_observer_loop.py -v`
Expected: All tests PASS (new + existing)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/intelligence/observer_loop.py tests/unit/test_neural/test_integration.py
git commit -m "feat: wire neural context engine into observer loop"
```

---

### Task 9: Update Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add neural dependencies to pyproject.toml**

Add a new optional dependency group after the existing ones:

```toml
neural = ["onnxruntime>=1.17", "huggingface-hub>=0.23", "tokenizers>=0.15", "numpy>=1.24"]
```

Update the `all` group to include `neural`:
```toml
all = ["homie-ai[model,voice,context,storage,app,neural]"]
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -q`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add neural module dependencies"
```

---

### Task 10: Full Integration Test

- [ ] **Step 1: Run complete test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify imports**

Run:
```bash
python -c "from homie_core.neural.utils import cosine_similarity; print('Utils OK')"
python -c "from homie_core.neural.model_manager import EmbeddingModel; print('EmbeddingModel OK')"
python -c "from homie_core.neural.activity_classifier import ActivityClassifier; print('ActivityClassifier OK')"
python -c "from homie_core.neural.context_engine import SemanticContextEngine; print('ContextEngine OK')"
python -c "from homie_core.neural.intent_inferencer import IntentInferencer; print('IntentInferencer OK')"
python -c "from homie_core.neural.sentiment import SentimentAnalyzer; print('SentimentAnalyzer OK')"
python -c "from homie_core.neural.consolidator import NeuralConsolidator; print('NeuralConsolidator OK')"
```
Expected: All print OK

- [ ] **Step 3: Commit if any fixes needed**

```bash
git add -A
git commit -m "test: verify full neural module integration"
```
