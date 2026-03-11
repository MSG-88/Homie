# Smart Context Engine — Phase 1 of Deep Learning Intelligence

## Goal

Add a neural network foundation to Homie that provides genuine semantic understanding of user activity, intent, and emotional state. This is Phase 1 of a 4-phase plan to make Homie a deeply intelligent, learning assistant.

## Phases Overview

1. **Smart Context Engine** (this spec) — Embedding model, activity classification, semantic context, intent inference, sentiment analysis, neural memory consolidation
2. **Personal Neural Profile** — Behavioral DNA, work rhythm learning, preference adaptation
3. **Predictive Intelligence** — Workflow prediction, anomaly detection, attention-aware interruptions
4. **Autonomous Agent Brain** — Multi-step reasoning, task decomposition, tool use, self-correction

## Architecture

### Hardware Strategy: Hybrid

- GPU-accelerated when available (ONNX Runtime GPU, or PyTorch)
- Graceful CPU fallback with lightweight ONNX models
- All models small enough to co-exist with the main LLM

### New Module: `src/homie_core/neural/`

```
neural/
  __init__.py
  model_manager.py      # Embedding model loading/inference
  activity_classifier.py # Activity type classification
  context_engine.py     # Semantic context tracking
  intent_inferencer.py  # Sequence-based intent prediction
  sentiment.py          # Text sentiment/emotion analysis
  consolidator.py       # Neural memory consolidation
  utils.py              # Cosine similarity, vector ops
```

## Component Designs

### 1. Embedding Model Manager (`model_manager.py`)

Loads and manages a local sentence embedding model for converting text into semantic vectors.

- **Primary backend:** ONNX Runtime (fast, CPU-friendly, no PyTorch needed)
- **Fallback:** `sentence-transformers` when GPU is available
- **Default model:** `all-MiniLM-L6-v2` (~80MB, 384-dim vectors, <20ms/embedding on CPU)
- **Storage:** `~/.homie/models/neural/`
- **Auto-download:** Via `huggingface_hub` on first run

API:
```python
class EmbeddingModel:
    def load(model_name: str, device: str = "auto") -> None
    def embed(text: str) -> list[float]
    def embed_batch(texts: list[str]) -> list[list[float]]
    def unload() -> None
    @property
    def is_loaded(self) -> bool
    @property
    def dimension(self) -> int
```

Lazy-loaded on first use, stays resident. Thread-safe.

### 2. Activity Classifier (`activity_classifier.py`)

Classifies user activity into semantic categories using embeddings.

- **Categories:** `coding`, `researching`, `communicating`, `writing`, `designing`, `browsing`, `media`, `system`, `unknown`
- **Initial approach:** Cosine similarity to category prototype embeddings (zero-shot)
- **Online learning:** Small feedforward network (384 -> 64 -> 9) trained from user behavior via SGD
- **Output:** Dict of category -> confidence score (not just top label)

API:
```python
class ActivityClassifier:
    def classify(process: str, title: str) -> dict[str, float]
    def train_online(process: str, title: str, label: str) -> None
    def serialize() -> dict
    @classmethod
    def deserialize(data: dict) -> ActivityClassifier
```

### 3. Semantic Context Engine (`context_engine.py`)

Replaces keyword-based context matching with embedding-powered semantic understanding.

- **Context vector:** Rolling weighted average of recent activity embeddings (exponential decay)
- **Semantic shift detection:** Cosine similarity between current and recent context vectors; shift when similarity drops below threshold
- **Memory matching:** Finds relevant episodic/semantic memories by cosine similarity to current context vector
- **Integrates with:** `ProactiveRetrieval`, `ObserverLoop`, `WorkingMemory`

API:
```python
class SemanticContextEngine:
    def update(process: str, title: str) -> None
    def get_context_vector() -> list[float]
    def detect_context_shift() -> bool
    def find_relevant_memories(memories: list, top_k: int = 5) -> list
    def get_activity_summary() -> dict
```

### 4. Intent Inferencer (`intent_inferencer.py`)

Predicts what the user is trying to accomplish from activity sequences.

- **Architecture:** GRU network (384 -> 128 -> 64) processing sequences of activity embeddings
- **Input:** Last 10-20 activity embeddings as a sequence
- **Output:** Predicted next activity embedding, task completion estimate, likely information needs
- **Training:** Online from observed sequences; each completed sequence becomes training data
- **Fallback (CPU-only):** Weighted k-NN over stored activity sequences when GRU is too expensive

API:
```python
class IntentInferencer:
    def observe(activity_embedding: list[float]) -> None
    def predict_next() -> dict  # {predicted_activity, confidence, estimated_completion}
    def get_likely_needs() -> list[str]
    def train_from_sequence(sequence: list[list[float]]) -> None
    def serialize() -> dict
    @classmethod
    def deserialize(data: dict) -> IntentInferencer
```

### 5. Sentiment Analyzer (`sentiment.py`)

Lightweight emotion detection from user text.

- **Model:** Small ONNX classifier (~5MB), or distilled BERT-tiny for sentiment
- **Output dimensions:** Sentiment (positive/negative/neutral) and arousal (calm/stressed/frustrated)
- **Applied to:** Chat messages, optionally clipboard content (privacy-gated)
- **Integration:** Feeds into `InterruptionModel` — suppresses interruptions when user is frustrated, offers help when stuck

API:
```python
class SentimentAnalyzer:
    def analyze(text: str) -> SentimentResult  # {sentiment, arousal, confidence}
    def analyze_batch(texts: list[str]) -> list[SentimentResult]
```

### 6. Neural Memory Consolidator (`consolidator.py`)

Upgrades existing memory consolidation with embedding-powered clustering and insight generation.

- **Clustering:** Groups episodic memories by embedding similarity (DBSCAN or agglomerative on embedding vectors)
- **Pattern extraction:** Identifies recurring activity sequences and generates higher-order insights (e.g., "User researches before coding")
- **Smart decay:** Relevance-weighted forgetting using embedding similarity to current interests, not just timestamps
- **Idle consolidation:** Runs during low-activity periods (dream-like processing)

API:
```python
class NeuralConsolidator:
    def consolidate(episodes: list, current_context: list[float]) -> ConsolidationResult
    def find_patterns(episodes: list) -> list[Pattern]
    def compute_relevance(memory: dict, context: list[float]) -> float
```

## Vector Utilities (`utils.py`)

Pure Python vector operations (no numpy required, but uses it when available):

```python
def cosine_similarity(a: list[float], b: list[float]) -> float
def weighted_average(vectors: list[list[float]], weights: list[float]) -> list[float]
def top_k_similar(query: list[float], candidates: list[list[float]], k: int) -> list[tuple[int, float]]
```

## Integration Points

| Existing Component | Integration |
|---|---|
| `ObserverLoop` | Feeds raw observations to `SemanticContextEngine` and `ActivityClassifier` |
| `ProactiveRetrieval` | Uses `SemanticContextEngine.find_relevant_memories()` instead of keyword matching |
| `InterruptionModel` | Receives sentiment signals and activity classification confidence |
| `WorkingMemory` | Stores current context vector, activity classification, sentiment |
| `EpisodicMemory` | Stores activity embeddings alongside episodes for `NeuralConsolidator` |
| `BriefingGenerator` | Uses activity classification for richer summaries |
| `TaskGraph` | Enhanced with semantic task clustering via embeddings |

## Dependencies

- `onnxruntime` (CPU inference, ~15MB)
- `onnxruntime-gpu` (optional, for GPU acceleration)
- `huggingface_hub` (model download)
- `tokenizers` (fast tokenization for embedding model)
- `numpy` (vector operations, optional but recommended)

## Future Phases Built on This Foundation

1. **Predictive Workflow Engine** — Uses intent inferencer sequences to predict next 3 actions, pre-stages resources
2. **Behavioral DNA** — Neural fingerprint of work style learned from activity classifier + context engine patterns
3. **Anomaly Detection** — Statistical deviation from learned patterns triggers contextual support
4. **Transfer Learning** — Knowledge from one project improves assistance on similar future projects
5. **Attention-Aware Interruptions** — Activity embedding velocity detects deep focus vs. shallow browsing
6. **Compositional Reasoning** — Chains neural modules for complex queries across time and context
