# Neural Intelligence Phases 2-4 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Personal Neural Profile, Predictive Intelligence, and Autonomous Agent Brain to make Homie a uniquely intelligent assistant with cutting-edge algorithms.

**Architecture:** Three phases building on the Phase 1 neural foundation. Each phase adds a new intelligence layer: Phase 2 learns WHO the user is (behavioral DNA, circadian rhythms, preference drift). Phase 3 predicts WHAT happens next (Markov workflow prediction, isolation forest anomaly detection, entropy-based flow detection). Phase 4 enables HOW to act autonomously (HTN planning, MCTS action selection, reflective self-correction).

**Tech Stack:** Pure Python with optional numpy acceleration. ONNX Runtime embeddings from Phase 1. All algorithms implemented from scratch for full local/privacy-first operation.

---

## File Structure

### Phase 2: Personal Neural Profile
| File | Responsibility |
|------|---------------|
| `src/homie_core/neural/rhythm_model.py` | Fourier-based circadian productivity modeling |
| `src/homie_core/neural/behavioral_profile.py` | Behavioral eigenvector decomposition (PCA via power iteration) |
| `src/homie_core/neural/preference_engine.py` | Online preference learning with CUSUM change-point detection |
| `tests/unit/test_neural/test_rhythm_model.py` | Tests for rhythm model |
| `tests/unit/test_neural/test_behavioral_profile.py` | Tests for behavioral profile |
| `tests/unit/test_neural/test_preference_engine.py` | Tests for preference engine |

### Phase 3: Predictive Intelligence
| File | Responsibility |
|------|---------------|
| `src/homie_core/intelligence/workflow_predictor.py` | Sparse Markov chain with Bayesian smoothing for workflow prediction |
| `src/homie_core/intelligence/anomaly_detector.py` | Streaming isolation forest for anomaly detection |
| `src/homie_core/intelligence/flow_detector.py` | Shannon entropy-based flow state detection |
| `tests/unit/test_intelligence/test_workflow_predictor.py` | Tests for workflow predictor |
| `tests/unit/test_intelligence/test_anomaly_detector.py` | Tests for anomaly detector |
| `tests/unit/test_intelligence/test_flow_detector.py` | Tests for flow detector |

### Phase 4: Autonomous Agent Brain
| File | Responsibility |
|------|---------------|
| `src/homie_core/intelligence/planner.py` | Hierarchical Task Network (HTN) goal decomposition |
| `src/homie_core/intelligence/action_selector.py` | Monte Carlo Tree Search (UCT) for action selection |
| `src/homie_core/intelligence/self_reflection.py` | Reflective scoring with Platt-scaled confidence calibration |
| `tests/unit/test_intelligence/test_planner.py` | Tests for HTN planner |
| `tests/unit/test_intelligence/test_action_selector.py` | Tests for MCTS action selector |
| `tests/unit/test_intelligence/test_self_reflection.py` | Tests for self-reflection |

### Integration
| File | Responsibility |
|------|---------------|
| `src/homie_core/intelligence/observer_loop.py` | Modified to feed Phase 2+3 components |
| `tests/unit/test_neural/test_integration.py` | Modified to test full pipeline |

---

## Chunk 1: Phase 2 — Personal Neural Profile

### Task 1: Fourier-Based Circadian Rhythm Model

**Files:**
- Create: `src/homie_core/neural/rhythm_model.py`
- Test: `tests/unit/test_neural/test_rhythm_model.py`

**Algorithm:** Discrete Fourier Transform decomposition of productivity signals sampled at hourly resolution. Extracts dominant frequency components (daily ~24h, weekly ~168h cycles) and reconstructs a continuous productivity curve. Predicts optimal work windows by finding peaks in the reconstructed signal.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_neural/test_rhythm_model.py
import math
from homie_core.neural.rhythm_model import CircadianRhythmModel


def test_record_activity():
    model = CircadianRhythmModel()
    model.record_activity(hour=9, productivity_score=0.8, activity_type="coding")
    assert len(model._hourly_buckets[9]) == 1


def test_record_clamps_hour():
    model = CircadianRhythmModel()
    model.record_activity(hour=25, productivity_score=0.5)
    assert len(model._hourly_buckets[1]) == 1  # 25 % 24 = 1


def test_get_hourly_averages():
    model = CircadianRhythmModel()
    model.record_activity(hour=9, productivity_score=0.8)
    model.record_activity(hour=9, productivity_score=0.6)
    model.record_activity(hour=14, productivity_score=0.9)
    avgs = model.get_hourly_averages()
    assert abs(avgs[9] - 0.7) < 1e-6
    assert abs(avgs[14] - 0.9) < 1e-6
    assert avgs[0] == 0.0  # no data


def test_fourier_decompose_returns_components():
    model = CircadianRhythmModel()
    # Simulate a daily pattern: high at 10, low at 3
    for day in range(7):
        for h in range(24):
            score = 0.5 + 0.4 * math.cos(2 * math.pi * (h - 10) / 24)
            model.record_activity(hour=h, productivity_score=max(0, min(1, score)))
    components = model.fourier_decompose(top_k=3)
    assert len(components) <= 3
    assert all("frequency" in c and "amplitude" in c and "phase" in c for c in components)
    # Dominant component should have period near 24
    assert components[0]["amplitude"] > 0.1


def test_predict_productivity():
    model = CircadianRhythmModel()
    for day in range(7):
        for h in range(24):
            score = 0.5 + 0.4 * math.cos(2 * math.pi * (h - 10) / 24)
            model.record_activity(hour=h, productivity_score=max(0, min(1, score)))
    pred_10 = model.predict_productivity(hour=10)
    pred_3 = model.predict_productivity(hour=3)
    # Peak near 10 should be higher than trough near 3
    assert pred_10 > pred_3


def test_get_optimal_windows():
    model = CircadianRhythmModel()
    for day in range(7):
        for h in range(24):
            score = 0.5 + 0.4 * math.cos(2 * math.pi * (h - 10) / 24)
            model.record_activity(hour=h, productivity_score=max(0, min(1, score)))
    windows = model.get_optimal_windows(top_n=3)
    assert len(windows) <= 3
    assert all("hour" in w and "predicted_score" in w for w in windows)
    # Best window should be near hour 10
    assert abs(windows[0]["hour"] - 10) <= 2


def test_get_activity_rhythm():
    model = CircadianRhythmModel()
    model.record_activity(hour=9, productivity_score=0.8, activity_type="coding")
    model.record_activity(hour=9, productivity_score=0.7, activity_type="coding")
    model.record_activity(hour=14, productivity_score=0.6, activity_type="writing")
    rhythm = model.get_activity_rhythm()
    assert "coding" in rhythm
    assert 9 in rhythm["coding"]


def test_serialize_deserialize():
    model = CircadianRhythmModel()
    model.record_activity(hour=9, productivity_score=0.8, activity_type="coding")
    data = model.serialize()
    model2 = CircadianRhythmModel.deserialize(data)
    assert len(model2._hourly_buckets[9]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_neural/test_rhythm_model.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/homie_core/neural/rhythm_model.py
"""Fourier-based circadian rhythm modeling.

Uses Discrete Fourier Transform to decompose productivity signals into
frequency components, identifying daily/weekly cycles and predicting
optimal work windows.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional


class CircadianRhythmModel:
    """Models user productivity as a sum of sinusoidal components.

    Records productivity observations bucketed by hour-of-day,
    decomposes the signal via DFT, and reconstructs a continuous
    productivity curve for prediction.
    """

    def __init__(self, num_hours: int = 24):
        self._num_hours = num_hours
        self._hourly_buckets: dict[int, list[float]] = defaultdict(list)
        self._activity_buckets: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._fourier_cache: Optional[list[dict]] = None

    def record_activity(
        self,
        hour: int,
        productivity_score: float,
        activity_type: Optional[str] = None,
    ) -> None:
        """Record a productivity observation at a given hour."""
        hour = hour % self._num_hours
        score = max(0.0, min(1.0, productivity_score))
        self._hourly_buckets[hour].append(score)
        if activity_type:
            self._activity_buckets[activity_type][hour].append(score)
        self._fourier_cache = None  # invalidate

    def get_hourly_averages(self) -> list[float]:
        """Return average productivity for each hour slot."""
        result = []
        for h in range(self._num_hours):
            bucket = self._hourly_buckets[h]
            result.append(sum(bucket) / len(bucket) if bucket else 0.0)
        return result

    def _dft(self, signal: list[float]) -> list[tuple[float, float]]:
        """Compute Discrete Fourier Transform.

        Returns list of (real, imag) pairs for each frequency bin.
        """
        n = len(signal)
        result = []
        for k in range(n):
            re = 0.0
            im = 0.0
            for t in range(n):
                angle = 2 * math.pi * k * t / n
                re += signal[t] * math.cos(angle)
                im -= signal[t] * math.sin(angle)
            result.append((re, im))
        return result

    def fourier_decompose(self, top_k: int = 5) -> list[dict]:
        """Decompose hourly productivity into frequency components.

        Returns top_k components sorted by amplitude, each with:
        - frequency: cycles per day (for 24h signal)
        - amplitude: strength of this component
        - phase: phase offset in radians
        - period: period in hours
        """
        if self._fourier_cache is not None:
            return self._fourier_cache[:top_k]

        signal = self.get_hourly_averages()
        n = len(signal)
        if all(v == 0.0 for v in signal):
            return []

        dft = self._dft(signal)
        components = []
        for k in range(1, n // 2 + 1):  # skip DC component (k=0)
            re, im = dft[k]
            amplitude = 2.0 * math.sqrt(re * re + im * im) / n
            phase = math.atan2(-im, re)
            period = n / k  # in hours
            components.append({
                "frequency": k,
                "amplitude": amplitude,
                "phase": phase,
                "period": period,
            })

        components.sort(key=lambda c: c["amplitude"], reverse=True)
        self._fourier_cache = components
        return components[:top_k]

    def predict_productivity(self, hour: float) -> float:
        """Predict productivity at a fractional hour using Fourier reconstruction."""
        signal = self.get_hourly_averages()
        n = len(signal)
        if all(v == 0.0 for v in signal):
            return 0.5  # no data, return neutral

        # DC component (mean)
        dft = self._dft(signal)
        dc = dft[0][0] / n

        # Reconstruct from top components
        components = self.fourier_decompose(top_k=6)
        value = dc
        for comp in components:
            k = comp["frequency"]
            value += comp["amplitude"] * math.cos(
                2 * math.pi * k * hour / n + comp["phase"]
            )

        return max(0.0, min(1.0, value))

    def get_optimal_windows(self, top_n: int = 3) -> list[dict]:
        """Find the top_n most productive hours."""
        predictions = []
        for h in range(self._num_hours):
            predictions.append({
                "hour": h,
                "predicted_score": self.predict_productivity(h),
            })
        predictions.sort(key=lambda p: p["predicted_score"], reverse=True)
        return predictions[:top_n]

    def get_activity_rhythm(self) -> dict[str, dict[int, float]]:
        """Return per-activity hourly averages."""
        result = {}
        for activity, buckets in self._activity_buckets.items():
            hourly = {}
            for h, scores in buckets.items():
                if scores:
                    hourly[h] = sum(scores) / len(scores)
            result[activity] = hourly
        return result

    def serialize(self) -> dict:
        """Serialize model state."""
        return {
            "num_hours": self._num_hours,
            "hourly_buckets": {
                str(k): v for k, v in self._hourly_buckets.items()
            },
            "activity_buckets": {
                act: {str(h): scores for h, scores in buckets.items()}
                for act, buckets in self._activity_buckets.items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> CircadianRhythmModel:
        """Deserialize from dict."""
        model = cls(num_hours=data.get("num_hours", 24))
        for k, v in data.get("hourly_buckets", {}).items():
            model._hourly_buckets[int(k)] = v
        for act, buckets in data.get("activity_buckets", {}).items():
            for h, scores in buckets.items():
                model._activity_buckets[act][int(h)] = scores
        return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_neural/test_rhythm_model.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/neural/rhythm_model.py tests/unit/test_neural/test_rhythm_model.py
git commit -m "feat: add Fourier-based circadian rhythm model"
```

---

### Task 2: Behavioral Eigenvector Profile (PCA via Power Iteration)

**Files:**
- Create: `src/homie_core/neural/behavioral_profile.py`
- Test: `tests/unit/test_neural/test_behavioral_profile.py`

**Algorithm:** Power Iteration method to compute principal components of behavior embedding vectors without numpy. Builds a covariance matrix from observed activity embeddings, then extracts top-k eigenvectors to form a "behavioral DNA fingerprint." Includes cosine distance for comparing behavioral profiles.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_neural/test_behavioral_profile.py
import math
from homie_core.neural.behavioral_profile import BehavioralProfile


def _make_embedding(seed: float, dim: int = 8) -> list[float]:
    """Deterministic fake embedding."""
    return [math.sin(seed * (i + 1)) for i in range(dim)]


def test_observe_adds_sample():
    bp = BehavioralProfile(embed_dim=8)
    bp.observe(_make_embedding(1.0))
    assert bp.sample_count == 1


def test_compute_mean():
    bp = BehavioralProfile(embed_dim=4)
    bp.observe([1.0, 0.0, 0.0, 0.0])
    bp.observe([0.0, 1.0, 0.0, 0.0])
    mean = bp.get_mean_vector()
    assert abs(mean[0] - 0.5) < 1e-6
    assert abs(mean[1] - 0.5) < 1e-6


def test_covariance_matrix():
    bp = BehavioralProfile(embed_dim=2)
    bp.observe([1.0, 0.0])
    bp.observe([0.0, 1.0])
    cov = bp._compute_covariance()
    assert len(cov) == 2
    assert len(cov[0]) == 2


def test_power_iteration_finds_eigenvector():
    bp = BehavioralProfile(embed_dim=4)
    # Strong signal along dim 0
    for _ in range(20):
        bp.observe([1.0, 0.1, 0.0, 0.0])
        bp.observe([0.9, 0.05, 0.0, 0.0])
    eigenvecs = bp.compute_eigenvectors(top_k=1)
    assert len(eigenvecs) == 1
    # First eigenvector should be mostly along dim 0
    assert abs(eigenvecs[0][0]) > 0.5


def test_behavioral_fingerprint():
    bp = BehavioralProfile(embed_dim=4)
    for _ in range(20):
        bp.observe([1.0, 0.5, 0.0, 0.0])
    fingerprint = bp.get_fingerprint(top_k=2)
    assert "eigenvectors" in fingerprint
    assert "explained_variance" in fingerprint
    assert "sample_count" in fingerprint


def test_compare_profiles_identical():
    bp1 = BehavioralProfile(embed_dim=4)
    bp2 = BehavioralProfile(embed_dim=4)
    for _ in range(20):
        bp1.observe([1.0, 0.5, 0.0, 0.0])
        bp2.observe([1.0, 0.5, 0.0, 0.0])
    similarity = BehavioralProfile.compare(bp1, bp2, top_k=2)
    assert similarity > 0.8


def test_compare_profiles_different():
    bp1 = BehavioralProfile(embed_dim=4)
    bp2 = BehavioralProfile(embed_dim=4)
    for _ in range(20):
        bp1.observe([1.0, 0.0, 0.0, 0.0])
        bp2.observe([0.0, 0.0, 0.0, 1.0])
    similarity = BehavioralProfile.compare(bp1, bp2, top_k=1)
    assert similarity < 0.5


def test_serialize_deserialize():
    bp = BehavioralProfile(embed_dim=4)
    for _ in range(5):
        bp.observe([1.0, 0.5, 0.2, 0.1])
    data = bp.serialize()
    bp2 = BehavioralProfile.deserialize(data)
    assert bp2.sample_count == 5
    assert bp2.get_mean_vector() == bp.get_mean_vector()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_neural/test_behavioral_profile.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/homie_core/neural/behavioral_profile.py
"""Behavioral eigenvector decomposition via power iteration.

Computes principal components of activity embedding vectors to form a
'behavioral DNA fingerprint' — a compact representation of a user's
unique work patterns. Uses power iteration (no numpy required) to
extract eigenvectors from the covariance matrix.
"""
from __future__ import annotations

import math
import random
from homie_core.neural.utils import cosine_similarity


class BehavioralProfile:
    """Builds a behavioral fingerprint from activity embeddings.

    Accumulates embeddings, computes covariance matrix, and extracts
    principal components via power iteration to characterize the user's
    dominant behavioral modes.
    """

    def __init__(self, embed_dim: int = 384, max_samples: int = 5000):
        self._dim = embed_dim
        self._max_samples = max_samples
        self._samples: list[list[float]] = []
        self._sum: list[float] = [0.0] * embed_dim
        self._eigen_cache: list[list[float]] | None = None

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def observe(self, embedding: list[float]) -> None:
        """Record an activity embedding observation."""
        if len(embedding) != self._dim:
            raise ValueError(f"Expected dim {self._dim}, got {len(embedding)}")
        self._samples.append(list(embedding))
        for i in range(self._dim):
            self._sum[i] += embedding[i]
        # Evict oldest if over capacity
        if len(self._samples) > self._max_samples:
            old = self._samples.pop(0)
            for i in range(self._dim):
                self._sum[i] -= old[i]
        self._eigen_cache = None

    def get_mean_vector(self) -> list[float]:
        """Return mean of all observed embeddings."""
        n = len(self._samples)
        if n == 0:
            return [0.0] * self._dim
        return [self._sum[i] / n for i in range(self._dim)]

    def _compute_covariance(self) -> list[list[float]]:
        """Compute covariance matrix of centered samples."""
        n = len(self._samples)
        mean = self.get_mean_vector()
        d = self._dim

        # Center samples
        centered = []
        for s in self._samples:
            centered.append([s[i] - mean[i] for i in range(d)])

        # Cov = (1/n) * X^T X
        cov = [[0.0] * d for _ in range(d)]
        for s in centered:
            for i in range(d):
                if abs(s[i]) < 1e-12:
                    continue
                for j in range(i, d):
                    val = s[i] * s[j]
                    cov[i][j] += val
                    if i != j:
                        cov[j][i] += val

        if n > 1:
            for i in range(d):
                for j in range(d):
                    cov[i][j] /= (n - 1)

        return cov

    def _mat_vec(self, mat: list[list[float]], vec: list[float]) -> list[float]:
        """Matrix-vector multiplication."""
        d = len(vec)
        result = [0.0] * d
        for i in range(d):
            s = 0.0
            for j in range(d):
                s += mat[i][j] * vec[j]
            result[i] = s
        return result

    def _vec_norm(self, vec: list[float]) -> float:
        return math.sqrt(sum(v * v for v in vec))

    def _normalize(self, vec: list[float]) -> list[float]:
        norm = self._vec_norm(vec)
        if norm < 1e-12:
            return vec
        return [v / norm for v in vec]

    def compute_eigenvectors(
        self, top_k: int = 3, max_iterations: int = 200, tol: float = 1e-6
    ) -> list[list[float]]:
        """Extract top-k eigenvectors via deflated power iteration.

        Power iteration finds the dominant eigenvector by repeatedly
        multiplying a random vector by the covariance matrix and
        normalizing. Deflation removes found components to find the next.
        """
        if len(self._samples) < 2:
            return []

        cov = self._compute_covariance()
        d = self._dim
        eigenvectors = []
        eigenvalues = []

        for _ in range(min(top_k, d)):
            # Random initial vector (seeded for reproducibility)
            rng = random.Random(42 + len(eigenvectors))
            vec = self._normalize([rng.gauss(0, 1) for _ in range(d)])

            eigenvalue = 0.0
            for _it in range(max_iterations):
                new_vec = self._mat_vec(cov, vec)
                new_norm = self._vec_norm(new_vec)
                if new_norm < 1e-12:
                    break
                new_vec = [v / new_norm for v in new_vec]

                # Check convergence
                diff = sum((new_vec[i] - vec[i]) ** 2 for i in range(d))
                vec = new_vec
                eigenvalue = new_norm
                if diff < tol:
                    break

            eigenvectors.append(vec)
            eigenvalues.append(eigenvalue)

            # Deflation: remove this component from covariance
            for i in range(d):
                for j in range(d):
                    cov[i][j] -= eigenvalue * vec[i] * vec[j]

        self._eigen_cache = eigenvectors
        return eigenvectors

    def get_fingerprint(self, top_k: int = 3) -> dict:
        """Compute behavioral DNA fingerprint.

        Returns eigenvectors (principal behavioral modes), their
        explained variance ratios, and metadata.
        """
        eigenvecs = self.compute_eigenvectors(top_k=top_k)

        # Compute explained variance from eigenvalues
        cov = self._compute_covariance()
        total_var = sum(cov[i][i] for i in range(self._dim))

        variances = []
        for ev in eigenvecs:
            projected = self._mat_vec(self._compute_covariance(), ev)
            var = sum(ev[i] * projected[i] for i in range(self._dim))
            variances.append(var)

        explained = [v / total_var if total_var > 0 else 0 for v in variances]

        return {
            "eigenvectors": eigenvecs,
            "explained_variance": explained,
            "sample_count": self.sample_count,
            "mean_vector": self.get_mean_vector(),
        }

    @staticmethod
    def compare(
        profile1: BehavioralProfile,
        profile2: BehavioralProfile,
        top_k: int = 3,
    ) -> float:
        """Compare two behavioral profiles by subspace alignment.

        Computes average absolute cosine similarity between corresponding
        eigenvectors — high similarity means similar behavioral patterns.
        """
        ev1 = profile1.compute_eigenvectors(top_k=top_k)
        ev2 = profile2.compute_eigenvectors(top_k=top_k)
        if not ev1 or not ev2:
            return 0.0

        n = min(len(ev1), len(ev2))
        total_sim = 0.0
        for i in range(n):
            total_sim += abs(cosine_similarity(ev1[i], ev2[i]))
        return total_sim / n

    def serialize(self) -> dict:
        return {
            "dim": self._dim,
            "max_samples": self._max_samples,
            "samples": self._samples,
            "sum": self._sum,
        }

    @classmethod
    def deserialize(cls, data: dict) -> BehavioralProfile:
        bp = cls(embed_dim=data["dim"], max_samples=data.get("max_samples", 5000))
        bp._samples = data["samples"]
        bp._sum = data["sum"]
        return bp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_neural/test_behavioral_profile.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/neural/behavioral_profile.py tests/unit/test_neural/test_behavioral_profile.py
git commit -m "feat: add behavioral eigenvector profile with power iteration PCA"
```

---

### Task 3: Preference Engine with CUSUM Change-Point Detection

**Files:**
- Create: `src/homie_core/neural/preference_engine.py`
- Test: `tests/unit/test_neural/test_preference_engine.py`

**Algorithm:** Cumulative Sum (CUSUM) control chart for detecting when user preferences drift. Maintains running averages for preference dimensions (activity preferences, time preferences, tool preferences) and triggers change-point alerts when the cumulative deviation from the mean exceeds a threshold.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_neural/test_preference_engine.py
from homie_core.neural.preference_engine import PreferenceEngine


def test_record_preference():
    pe = PreferenceEngine()
    pe.record("tool", "vscode", 1.0)
    prefs = pe.get_preferences("tool")
    assert "vscode" in prefs
    assert prefs["vscode"] > 0


def test_preference_strength_increases():
    pe = PreferenceEngine()
    pe.record("tool", "vscode", 1.0)
    s1 = pe.get_preferences("tool")["vscode"]
    pe.record("tool", "vscode", 1.0)
    s2 = pe.get_preferences("tool")["vscode"]
    assert s2 > s1


def test_ema_decay():
    pe = PreferenceEngine(ema_alpha=0.3)
    pe.record("tool", "vim", 1.0)
    pe.record("tool", "vim", 0.0)
    pe.record("tool", "vim", 0.0)
    pref = pe.get_preferences("tool")["vim"]
    # Should have decayed from 1.0 towards 0
    assert pref < 0.5


def test_cusum_detects_shift():
    pe = PreferenceEngine(cusum_threshold=2.0)
    # Establish baseline: user prefers coding
    for _ in range(20):
        pe.record("activity", "coding", 1.0)
    # Sudden shift to writing
    for _ in range(10):
        pe.record("activity", "writing", 1.0)
        pe.record("activity", "coding", 0.0)
    shifts = pe.get_detected_shifts()
    assert len(shifts) > 0
    assert any(s["domain"] == "activity" for s in shifts)


def test_no_false_shift_on_stable():
    pe = PreferenceEngine(cusum_threshold=3.0)
    for _ in range(50):
        pe.record("activity", "coding", 1.0)
    shifts = pe.get_detected_shifts()
    # Should not detect shift in stable preferences
    coding_shifts = [s for s in shifts if s["key"] == "coding"]
    assert len(coding_shifts) == 0


def test_get_all_preferences():
    pe = PreferenceEngine()
    pe.record("tool", "vscode", 1.0)
    pe.record("time", "morning", 0.8)
    all_prefs = pe.get_all_preferences()
    assert "tool" in all_prefs
    assert "time" in all_prefs


def test_get_dominant_preference():
    pe = PreferenceEngine()
    pe.record("tool", "vscode", 1.0)
    pe.record("tool", "vscode", 1.0)
    pe.record("tool", "vim", 0.3)
    dominant = pe.get_dominant("tool")
    assert dominant == "vscode"


def test_serialize_deserialize():
    pe = PreferenceEngine()
    pe.record("tool", "vscode", 1.0)
    data = pe.serialize()
    pe2 = PreferenceEngine.deserialize(data)
    prefs = pe2.get_preferences("tool")
    assert "vscode" in prefs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_neural/test_preference_engine.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/homie_core/neural/preference_engine.py
"""Online preference learning with CUSUM change-point detection.

Tracks user preferences across domains (tools, activities, times) using
exponential moving averages. Detects preference drift using Cumulative Sum
(CUSUM) control charts — a statistical process control method that
accumulates deviations from the mean and signals when the cumulative
sum exceeds a threshold.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional


class _CUSUMTracker:
    """Tracks cumulative sum for one preference signal."""

    def __init__(self, threshold: float = 3.0, slack: float = 0.5):
        self._threshold = threshold
        self._slack = slack  # allowance parameter
        self._mean = 0.0
        self._count = 0
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._shift_detected = False

    def update(self, value: float) -> bool:
        """Update CUSUM with new observation. Returns True if shift detected."""
        self._count += 1
        if self._count <= 5:
            # Warmup: just update mean
            self._mean += (value - self._mean) / self._count
            return False

        deviation = value - self._mean
        self._cusum_pos = max(0, self._cusum_pos + deviation - self._slack)
        self._cusum_neg = max(0, self._cusum_neg - deviation - self._slack)

        if self._cusum_pos > self._threshold or self._cusum_neg > self._threshold:
            self._shift_detected = True
            # Reset after detection
            self._mean = value
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0
            return True

        # Slow mean update
        self._mean += 0.01 * (value - self._mean)
        return False

    @property
    def detected(self) -> bool:
        return self._shift_detected

    def serialize(self) -> dict:
        return {
            "threshold": self._threshold,
            "slack": self._slack,
            "mean": self._mean,
            "count": self._count,
            "cusum_pos": self._cusum_pos,
            "cusum_neg": self._cusum_neg,
            "shift_detected": self._shift_detected,
        }

    @classmethod
    def deserialize(cls, data: dict) -> _CUSUMTracker:
        t = cls(threshold=data["threshold"], slack=data["slack"])
        t._mean = data["mean"]
        t._count = data["count"]
        t._cusum_pos = data["cusum_pos"]
        t._cusum_neg = data["cusum_neg"]
        t._shift_detected = data["shift_detected"]
        return t


class PreferenceEngine:
    """Learns and tracks user preferences with drift detection.

    Maintains exponential moving averages (EMA) of preference scores
    per domain/key pair. Uses CUSUM change-point detection to identify
    when the user's preferences shift significantly.
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        cusum_threshold: float = 3.0,
    ):
        self._alpha = ema_alpha
        self._cusum_threshold = cusum_threshold
        # domain -> key -> EMA score
        self._preferences: dict[str, dict[str, float]] = defaultdict(dict)
        # domain -> key -> CUSUM tracker
        self._cusum: dict[str, dict[str, _CUSUMTracker]] = defaultdict(dict)
        # detected shifts
        self._shifts: list[dict] = []
        # observation counts
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record(self, domain: str, key: str, score: float) -> None:
        """Record a preference observation.

        Args:
            domain: Category (e.g., "tool", "activity", "time")
            key: Specific item (e.g., "vscode", "coding", "morning")
            score: Preference strength 0.0-1.0
        """
        score = max(0.0, min(1.0, score))
        prefs = self._preferences[domain]
        self._counts[domain][key] += 1

        if key in prefs:
            prefs[key] = (1 - self._alpha) * prefs[key] + self._alpha * score
        else:
            prefs[key] = score

        # CUSUM tracking
        if key not in self._cusum[domain]:
            self._cusum[domain][key] = _CUSUMTracker(
                threshold=self._cusum_threshold
            )
        shift = self._cusum[domain][key].update(score)
        if shift:
            self._shifts.append({
                "domain": domain,
                "key": key,
                "new_value": score,
                "observation": self._counts[domain][key],
            })

    def get_preferences(self, domain: str) -> dict[str, float]:
        """Get current preference scores for a domain."""
        return dict(self._preferences.get(domain, {}))

    def get_all_preferences(self) -> dict[str, dict[str, float]]:
        """Get all preferences across all domains."""
        return {d: dict(p) for d, p in self._preferences.items()}

    def get_dominant(self, domain: str) -> Optional[str]:
        """Get the highest-scoring preference in a domain."""
        prefs = self._preferences.get(domain, {})
        if not prefs:
            return None
        return max(prefs, key=prefs.get)

    def get_detected_shifts(self) -> list[dict]:
        """Return all detected preference shifts."""
        return list(self._shifts)

    def serialize(self) -> dict:
        return {
            "alpha": self._alpha,
            "cusum_threshold": self._cusum_threshold,
            "preferences": {
                d: dict(p) for d, p in self._preferences.items()
            },
            "counts": {
                d: dict(c) for d, c in self._counts.items()
            },
            "cusum": {
                d: {k: t.serialize() for k, t in trackers.items()}
                for d, trackers in self._cusum.items()
            },
            "shifts": self._shifts,
        }

    @classmethod
    def deserialize(cls, data: dict) -> PreferenceEngine:
        pe = cls(
            ema_alpha=data.get("alpha", 0.1),
            cusum_threshold=data.get("cusum_threshold", 3.0),
        )
        for d, prefs in data.get("preferences", {}).items():
            pe._preferences[d] = prefs
        for d, counts in data.get("counts", {}).items():
            for k, c in counts.items():
                pe._counts[d][k] = c
        for d, trackers in data.get("cusum", {}).items():
            for k, t_data in trackers.items():
                pe._cusum[d][k] = _CUSUMTracker.deserialize(t_data)
        pe._shifts = data.get("shifts", [])
        return pe
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_neural/test_preference_engine.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/neural/preference_engine.py tests/unit/test_neural/test_preference_engine.py
git commit -m "feat: add preference engine with CUSUM change-point detection"
```

---

## Chunk 2: Phase 3 — Predictive Intelligence

### Task 4: Sparse Markov Chain Workflow Predictor

**Files:**
- Create: `src/homie_core/intelligence/workflow_predictor.py`
- Test: `tests/unit/test_intelligence/test_workflow_predictor.py`

**Algorithm:** Builds a sparse transition probability matrix over activity states. Uses Laplace (add-k) Bayesian smoothing to handle unseen transitions gracefully. Predicts the next N most likely activities and their probabilities. Supports higher-order Markov chains (bigram context) for more accurate predictions.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_intelligence/test_workflow_predictor.py
from homie_core.intelligence.workflow_predictor import WorkflowPredictor


def test_observe_transition():
    wp = WorkflowPredictor()
    wp.observe("coding")
    wp.observe("researching")
    assert wp.transition_count("coding", "researching") == 1


def test_predict_next():
    wp = WorkflowPredictor()
    for _ in range(10):
        wp.observe("coding")
        wp.observe("researching")
        wp.observe("coding")
    predictions = wp.predict_next("coding", top_n=3)
    assert len(predictions) > 0
    assert predictions[0][0] == "researching"
    assert predictions[0][1] > 0.5


def test_laplace_smoothing():
    wp = WorkflowPredictor(smoothing_k=1.0)
    wp.observe("coding")
    wp.observe("researching")
    # Even unseen transitions should have nonzero probability
    predictions = wp.predict_next("coding", top_n=10)
    probs = {p[0]: p[1] for p in predictions}
    assert probs.get("researching", 0) > 0
    # All probabilities should sum to ~1
    total = sum(p[1] for p in predictions)
    assert abs(total - 1.0) < 0.01


def test_bigram_context():
    wp = WorkflowPredictor(order=2)
    sequence = ["coding", "researching", "writing", "coding", "researching", "writing"]
    for s in sequence:
        wp.observe(s)
    # After coding->researching, next should be writing
    predictions = wp.predict_next_with_context(["coding", "researching"], top_n=3)
    assert len(predictions) > 0
    assert predictions[0][0] == "writing"


def test_get_transition_matrix():
    wp = WorkflowPredictor()
    wp.observe("a")
    wp.observe("b")
    wp.observe("a")
    matrix = wp.get_transition_matrix()
    assert "a" in matrix
    assert "b" in matrix["a"]


def test_predict_sequence():
    wp = WorkflowPredictor()
    for _ in range(20):
        wp.observe("coding")
        wp.observe("testing")
        wp.observe("committing")
    seq = wp.predict_sequence("coding", length=3)
    assert len(seq) == 3
    assert seq[0] == "testing"


def test_get_stationary_distribution():
    wp = WorkflowPredictor()
    for _ in range(50):
        wp.observe("coding")
        wp.observe("testing")
    dist = wp.get_stationary_distribution(max_iterations=100)
    assert "coding" in dist
    assert "testing" in dist
    assert abs(sum(dist.values()) - 1.0) < 0.01


def test_serialize_deserialize():
    wp = WorkflowPredictor()
    wp.observe("coding")
    wp.observe("testing")
    data = wp.serialize()
    wp2 = WorkflowPredictor.deserialize(data)
    assert wp2.transition_count("coding", "testing") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_intelligence/test_workflow_predictor.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/homie_core/intelligence/workflow_predictor.py
"""Sparse Markov chain workflow predictor with Bayesian smoothing.

Builds a transition probability matrix over activity states from observed
sequences. Uses Laplace (add-k) smoothing for unseen transitions and
supports higher-order Markov chains (bigram context) for more accurate
multi-step workflow prediction.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional


class WorkflowPredictor:
    """Predicts next workflow activities using Markov chain transitions.

    Maintains sparse transition counts between activity states,
    computes smoothed probabilities, and predicts likely next
    activities or full sequences.
    """

    def __init__(self, smoothing_k: float = 0.1, order: int = 1):
        self._k = smoothing_k
        self._order = order
        # First-order: from_state -> to_state -> count
        self._transitions: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Higher-order: tuple(context) -> to_state -> count
        self._higher_transitions: dict[tuple, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._states: set[str] = set()
        self._history: list[str] = []

    def observe(self, state: str) -> None:
        """Record an observed activity state."""
        self._states.add(state)
        if self._history:
            prev = self._history[-1]
            self._transitions[prev][state] += 1

            # Higher-order transitions
            if self._order > 1 and len(self._history) >= self._order:
                context = tuple(self._history[-self._order:])
                self._higher_transitions[context][state] += 1

        self._history.append(state)

    def transition_count(self, from_state: str, to_state: str) -> int:
        """Get raw transition count."""
        return self._transitions.get(from_state, {}).get(to_state, 0)

    def _smoothed_probs(self, counts: dict[str, int]) -> dict[str, float]:
        """Compute Laplace-smoothed probabilities."""
        total = sum(counts.values()) + self._k * len(self._states)
        if total == 0:
            # Uniform
            n = len(self._states)
            return {s: 1.0 / n for s in self._states} if n > 0 else {}
        probs = {}
        for s in self._states:
            probs[s] = (counts.get(s, 0) + self._k) / total
        return probs

    def predict_next(
        self, current_state: str, top_n: int = 5
    ) -> list[tuple[str, float]]:
        """Predict next activities with probabilities."""
        counts = dict(self._transitions.get(current_state, {}))
        probs = self._smoothed_probs(counts)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:top_n]

    def predict_next_with_context(
        self, context: list[str], top_n: int = 5
    ) -> list[tuple[str, float]]:
        """Predict using higher-order context (bigram+)."""
        ctx = tuple(context[-self._order:])
        counts = dict(self._higher_transitions.get(ctx, {}))
        if not counts:
            # Fallback to first-order
            return self.predict_next(context[-1], top_n) if context else []
        probs = self._smoothed_probs(counts)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:top_n]

    def predict_sequence(
        self, start_state: str, length: int = 5
    ) -> list[str]:
        """Predict a sequence of future activities greedily."""
        sequence = []
        current = start_state
        for _ in range(length):
            preds = self.predict_next(current, top_n=1)
            if not preds:
                break
            next_state = preds[0][0]
            sequence.append(next_state)
            current = next_state
        return sequence

    def get_transition_matrix(self) -> dict[str, dict[str, float]]:
        """Get full smoothed transition probability matrix."""
        matrix = {}
        for state in self._states:
            counts = dict(self._transitions.get(state, {}))
            matrix[state] = self._smoothed_probs(counts)
        return matrix

    def get_stationary_distribution(
        self, max_iterations: int = 200, tol: float = 1e-8
    ) -> dict[str, float]:
        """Compute stationary distribution via power iteration.

        The stationary distribution represents the long-run fraction
        of time spent in each state.
        """
        states = sorted(self._states)
        n = len(states)
        if n == 0:
            return {}

        matrix = self.get_transition_matrix()
        # Start uniform
        dist = {s: 1.0 / n for s in states}

        for _ in range(max_iterations):
            new_dist = {s: 0.0 for s in states}
            for s_from in states:
                for s_to in states:
                    new_dist[s_to] += dist[s_from] * matrix.get(s_from, {}).get(s_to, 0)

            # Check convergence
            diff = sum(abs(new_dist[s] - dist[s]) for s in states)
            dist = new_dist
            if diff < tol:
                break

        return dist

    def serialize(self) -> dict:
        return {
            "k": self._k,
            "order": self._order,
            "states": sorted(self._states),
            "transitions": {
                k: dict(v) for k, v in self._transitions.items()
            },
            "higher_transitions": {
                "|".join(k): dict(v)
                for k, v in self._higher_transitions.items()
            },
            "history": self._history[-100:],  # keep last 100
        }

    @classmethod
    def deserialize(cls, data: dict) -> WorkflowPredictor:
        wp = cls(smoothing_k=data.get("k", 0.1), order=data.get("order", 1))
        wp._states = set(data.get("states", []))
        for k, v in data.get("transitions", {}).items():
            for k2, count in v.items():
                wp._transitions[k][k2] = count
        for ctx_str, v in data.get("higher_transitions", {}).items():
            ctx = tuple(ctx_str.split("|"))
            for k2, count in v.items():
                wp._higher_transitions[ctx][k2] = count
        wp._history = data.get("history", [])
        return wp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_intelligence/test_workflow_predictor.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/intelligence/workflow_predictor.py tests/unit/test_intelligence/test_workflow_predictor.py
git commit -m "feat: add sparse Markov chain workflow predictor with Bayesian smoothing"
```

---

### Task 5: Streaming Isolation Forest Anomaly Detector

**Files:**
- Create: `src/homie_core/intelligence/anomaly_detector.py`
- Test: `tests/unit/test_intelligence/test_anomaly_detector.py`

**Algorithm:** Isolation Forest — builds random binary trees that isolate observations. Anomalies have shorter average path lengths because they're easier to separate. Streaming variant maintains a forest of fixed-size trees that rotate as new data arrives. No training labels needed (unsupervised).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_intelligence/test_anomaly_detector.py
import random
from homie_core.intelligence.anomaly_detector import AnomalyDetector


def test_fit_and_score():
    ad = AnomalyDetector(n_trees=10, sample_size=32)
    # Normal data cluster
    data = [[float(i % 5), float(i % 3)] for i in range(50)]
    ad.fit(data)
    score = ad.score([2.0, 1.0])  # normal point
    assert 0.0 <= score <= 1.0


def test_anomaly_scores_higher():
    ad = AnomalyDetector(n_trees=50, sample_size=32)
    random.seed(42)
    # Normal cluster around (0, 0)
    normal = [[random.gauss(0, 0.5), random.gauss(0, 0.5)] for _ in range(100)]
    ad.fit(normal)
    normal_score = ad.score([0.1, -0.1])
    anomaly_score = ad.score([10.0, 10.0])
    # Anomaly should have higher score (closer to 1.0)
    assert anomaly_score > normal_score


def test_is_anomaly():
    ad = AnomalyDetector(n_trees=50, sample_size=32, threshold=0.6)
    random.seed(42)
    normal = [[random.gauss(0, 0.5), random.gauss(0, 0.5)] for _ in range(100)]
    ad.fit(normal)
    assert not ad.is_anomaly([0.0, 0.0])
    assert ad.is_anomaly([20.0, 20.0])


def test_streaming_update():
    ad = AnomalyDetector(n_trees=10, sample_size=16)
    data = [[float(i), float(i)] for i in range(20)]
    ad.fit(data)
    # Stream new data
    ad.stream_update([100.0, 100.0])
    # Should still function
    score = ad.score([0.0, 0.0])
    assert 0.0 <= score <= 1.0


def test_empty_forest_returns_neutral():
    ad = AnomalyDetector(n_trees=10, sample_size=16)
    score = ad.score([1.0, 2.0])
    assert score == 0.5  # neutral when no trees


def test_single_dimension():
    ad = AnomalyDetector(n_trees=20, sample_size=16)
    data = [[float(i)] for i in range(30)]
    ad.fit(data)
    score = ad.score([15.0])
    assert 0.0 <= score <= 1.0


def test_get_feature_importance():
    ad = AnomalyDetector(n_trees=30, sample_size=32)
    random.seed(42)
    # Dim 0 varies a lot, dim 1 is constant
    data = [[random.gauss(0, 5), 1.0] for _ in range(50)]
    ad.fit(data)
    importance = ad.get_feature_importance()
    assert len(importance) == 2
    # Dim 0 should be more important (more splits)
    assert importance[0] >= importance[1]


def test_serialize_deserialize():
    ad = AnomalyDetector(n_trees=10, sample_size=16)
    data = [[float(i), float(i * 2)] for i in range(20)]
    ad.fit(data)
    score_before = ad.score([5.0, 10.0])
    state = ad.serialize()
    ad2 = AnomalyDetector.deserialize(state)
    score_after = ad2.score([5.0, 10.0])
    assert abs(score_before - score_after) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_intelligence/test_anomaly_detector.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/homie_core/intelligence/anomaly_detector.py
"""Streaming isolation forest for anomaly detection.

Isolation Forest works by building random binary trees that recursively
partition data along random features at random split points. Anomalies
are isolated in fewer splits (shorter path lengths) because they differ
from the majority. This streaming variant rotates trees as new data arrives.

Reference: Liu, Ting & Zhou (2008) "Isolation Forest"
"""
from __future__ import annotations

import math
import random
from typing import Optional


def _avg_path_length(n: int) -> float:
    """Expected average path length in an unsuccessful BST search.

    This is the normalization constant c(n) from the Isolation Forest paper.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    h = math.log(n - 1) + 0.5772156649  # Euler-Mascheroni constant
    return 2.0 * h - 2.0 * (n - 1) / n


class _IsolationTree:
    """A single isolation tree (binary space partition)."""

    def __init__(
        self,
        split_feature: Optional[int] = None,
        split_value: Optional[float] = None,
        left: Optional[_IsolationTree] = None,
        right: Optional[_IsolationTree] = None,
        size: int = 0,
        depth: int = 0,
    ):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = size  # number of samples at this node (for external nodes)
        self.depth = depth

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def path_length(self, point: list[float], current_depth: int = 0) -> float:
        """Compute path length for a point in this tree."""
        if self.is_leaf:
            return current_depth + _avg_path_length(self.size)

        if self.split_feature is not None and self.split_value is not None:
            if point[self.split_feature] < self.split_value:
                if self.left:
                    return self.left.path_length(point, current_depth + 1)
            else:
                if self.right:
                    return self.right.path_length(point, current_depth + 1)

        return current_depth + _avg_path_length(self.size)

    def serialize(self) -> dict:
        result: dict = {"size": self.size, "depth": self.depth}
        if not self.is_leaf:
            result["split_feature"] = self.split_feature
            result["split_value"] = self.split_value
            if self.left:
                result["left"] = self.left.serialize()
            if self.right:
                result["right"] = self.right.serialize()
        return result

    @classmethod
    def deserialize(cls, data: dict) -> _IsolationTree:
        node = cls(
            split_feature=data.get("split_feature"),
            split_value=data.get("split_value"),
            size=data.get("size", 0),
            depth=data.get("depth", 0),
        )
        if "left" in data:
            node.left = cls.deserialize(data["left"])
        if "right" in data:
            node.right = cls.deserialize(data["right"])
        return node


def _build_tree(
    data: list[list[float]],
    current_depth: int,
    max_depth: int,
    rng: random.Random,
) -> _IsolationTree:
    """Recursively build an isolation tree."""
    n = len(data)
    if n <= 1 or current_depth >= max_depth:
        return _IsolationTree(size=n, depth=current_depth)

    n_features = len(data[0])
    feature = rng.randint(0, n_features - 1)

    # Get min/max for this feature
    values = [row[feature] for row in data]
    min_val = min(values)
    max_val = max(values)

    if abs(max_val - min_val) < 1e-12:
        return _IsolationTree(size=n, depth=current_depth)

    split_value = rng.uniform(min_val, max_val)

    left_data = [row for row in data if row[feature] < split_value]
    right_data = [row for row in data if row[feature] >= split_value]

    if not left_data or not right_data:
        return _IsolationTree(size=n, depth=current_depth)

    return _IsolationTree(
        split_feature=feature,
        split_value=split_value,
        left=_build_tree(left_data, current_depth + 1, max_depth, rng),
        right=_build_tree(right_data, current_depth + 1, max_depth, rng),
        size=n,
        depth=current_depth,
    )


class AnomalyDetector:
    """Streaming Isolation Forest anomaly detector.

    Builds an ensemble of random isolation trees. Points with shorter
    average path lengths are scored as more anomalous. The streaming
    variant maintains a buffer and periodically rebuilds trees.
    """

    def __init__(
        self,
        n_trees: int = 100,
        sample_size: int = 256,
        threshold: float = 0.6,
        seed: int = 42,
    ):
        self._n_trees = n_trees
        self._sample_size = sample_size
        self._threshold = threshold
        self._seed = seed
        self._trees: list[_IsolationTree] = []
        self._max_depth: int = 0
        self._data_buffer: list[list[float]] = []
        self._feature_split_counts: dict[int, int] = {}

    def fit(self, data: list[list[float]]) -> None:
        """Build the isolation forest from data."""
        if not data:
            return
        self._data_buffer = list(data)
        n = min(len(data), self._sample_size)
        self._max_depth = max(1, int(math.ceil(math.log2(max(n, 2)))))
        self._trees = []
        self._feature_split_counts = {}

        rng = random.Random(self._seed)
        for i in range(self._n_trees):
            sample = rng.sample(data, min(n, len(data)))
            tree = _build_tree(sample, 0, self._max_depth, random.Random(self._seed + i))
            self._trees.append(tree)
            self._count_splits(tree)

    def _count_splits(self, tree: _IsolationTree) -> None:
        """Count feature splits for importance calculation."""
        if tree.is_leaf:
            return
        if tree.split_feature is not None:
            self._feature_split_counts[tree.split_feature] = (
                self._feature_split_counts.get(tree.split_feature, 0) + 1
            )
        if tree.left:
            self._count_splits(tree.left)
        if tree.right:
            self._count_splits(tree.right)

    def score(self, point: list[float]) -> float:
        """Compute anomaly score for a point (0=normal, 1=anomaly)."""
        if not self._trees:
            return 0.5

        avg_path = sum(t.path_length(point) for t in self._trees) / len(self._trees)
        c = _avg_path_length(self._sample_size)
        if c == 0:
            return 0.5

        # Anomaly score formula from the paper
        return 2.0 ** (-avg_path / c)

    def is_anomaly(self, point: list[float]) -> bool:
        """Check if a point is anomalous."""
        return self.score(point) > self._threshold

    def stream_update(self, point: list[float]) -> None:
        """Add a streaming observation, rebuilding oldest tree."""
        self._data_buffer.append(point)
        if len(self._data_buffer) > self._sample_size * 4:
            self._data_buffer = self._data_buffer[-self._sample_size * 2:]

        # Rebuild one tree with updated data
        if self._trees:
            rng = random.Random(self._seed + len(self._data_buffer))
            n = min(len(self._data_buffer), self._sample_size)
            sample = rng.sample(self._data_buffer, n)
            new_tree = _build_tree(sample, 0, self._max_depth, rng)
            # Replace oldest tree
            self._trees.pop(0)
            self._trees.append(new_tree)

    def get_feature_importance(self) -> list[float]:
        """Get feature importance based on split frequency."""
        if not self._feature_split_counts:
            return []
        max_feat = max(self._feature_split_counts.keys()) + 1
        total = sum(self._feature_split_counts.values())
        if total == 0:
            return [0.0] * max_feat
        return [
            self._feature_split_counts.get(i, 0) / total
            for i in range(max_feat)
        ]

    def serialize(self) -> dict:
        return {
            "n_trees": self._n_trees,
            "sample_size": self._sample_size,
            "threshold": self._threshold,
            "seed": self._seed,
            "max_depth": self._max_depth,
            "trees": [t.serialize() for t in self._trees],
            "feature_split_counts": self._feature_split_counts,
        }

    @classmethod
    def deserialize(cls, data: dict) -> AnomalyDetector:
        ad = cls(
            n_trees=data["n_trees"],
            sample_size=data["sample_size"],
            threshold=data.get("threshold", 0.6),
            seed=data.get("seed", 42),
        )
        ad._max_depth = data.get("max_depth", 8)
        ad._trees = [_IsolationTree.deserialize(t) for t in data.get("trees", [])]
        ad._feature_split_counts = {
            int(k): v for k, v in data.get("feature_split_counts", {}).items()
        }
        return ad
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_intelligence/test_anomaly_detector.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/intelligence/anomaly_detector.py tests/unit/test_intelligence/test_anomaly_detector.py
git commit -m "feat: add streaming isolation forest anomaly detector"
```

---

### Task 6: Shannon Entropy Flow State Detector

**Files:**
- Create: `src/homie_core/intelligence/flow_detector.py`
- Test: `tests/unit/test_intelligence/test_flow_detector.py`

**Algorithm:** Uses Shannon entropy of activity switches over a sliding window to detect flow state. Low entropy (few different activities, long durations) indicates deep focus. High entropy (many switches) indicates scattered attention. Combines with switch rate and duration metrics for a composite flow score.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_intelligence/test_flow_detector.py
from homie_core.intelligence.flow_detector import FlowDetector


def test_record_activity():
    fd = FlowDetector(window_size=10)
    fd.record_activity("coding")
    assert fd._window == ["coding"]


def test_entropy_single_activity():
    fd = FlowDetector(window_size=10)
    for _ in range(10):
        fd.record_activity("coding")
    entropy = fd.compute_entropy()
    assert entropy == 0.0  # no uncertainty


def test_entropy_uniform():
    fd = FlowDetector(window_size=10)
    activities = ["coding", "writing", "browsing", "researching", "testing"]
    for i in range(10):
        fd.record_activity(activities[i % len(activities)])
    entropy = fd.compute_entropy()
    assert entropy > 1.0  # high entropy


def test_flow_score_deep_focus():
    fd = FlowDetector(window_size=20)
    for _ in range(20):
        fd.record_activity("coding")
    score = fd.get_flow_score()
    assert score > 0.7  # deep focus = high flow


def test_flow_score_scattered():
    fd = FlowDetector(window_size=20)
    activities = ["coding", "email", "browsing", "slack", "docs"]
    for i in range(20):
        fd.record_activity(activities[i % len(activities)])
    score = fd.get_flow_score()
    assert score < 0.4  # scattered = low flow


def test_is_in_flow():
    fd = FlowDetector(window_size=10, flow_threshold=0.7)
    for _ in range(10):
        fd.record_activity("coding")
    assert fd.is_in_flow()


def test_get_focus_report():
    fd = FlowDetector(window_size=10)
    for _ in range(5):
        fd.record_activity("coding")
    for _ in range(5):
        fd.record_activity("writing")
    report = fd.get_focus_report()
    assert "entropy" in report
    assert "flow_score" in report
    assert "dominant_activity" in report
    assert "switch_rate" in report


def test_switch_rate():
    fd = FlowDetector(window_size=10)
    fd.record_activity("coding")
    fd.record_activity("coding")
    fd.record_activity("writing")  # switch
    fd.record_activity("coding")   # switch
    rate = fd.get_switch_rate()
    assert abs(rate - 0.5) < 0.1  # 2 switches out of 3 transitions


def test_empty_detector():
    fd = FlowDetector(window_size=10)
    assert fd.compute_entropy() == 0.0
    assert fd.get_flow_score() == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_intelligence/test_flow_detector.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/homie_core/intelligence/flow_detector.py
"""Shannon entropy-based flow state detection.

Measures the entropy (information-theoretic uncertainty) of activity
switches over a sliding window. Low entropy means the user is deeply
focused on one activity (flow state). High entropy means scattered
attention across many activities.

Shannon entropy: H = -sum(p_i * log2(p_i)) for each activity proportion p_i
"""
from __future__ import annotations

import math
from collections import Counter, deque
from typing import Optional


class FlowDetector:
    """Detects user flow state using activity switch entropy.

    Maintains a sliding window of recent activities and computes
    Shannon entropy, switch rate, and a composite flow score.
    """

    def __init__(
        self,
        window_size: int = 30,
        flow_threshold: float = 0.7,
    ):
        self._window_size = window_size
        self._flow_threshold = flow_threshold
        self._window: deque[str] = deque(maxlen=window_size)

    def record_activity(self, activity: str) -> None:
        """Record an observed activity."""
        self._window.append(activity)

    def compute_entropy(self) -> float:
        """Compute Shannon entropy of activity distribution in window.

        Returns 0 for single-activity (perfect focus) and increases
        with more diverse activities (max = log2(n_unique)).
        """
        if len(self._window) == 0:
            return 0.0

        counts = Counter(self._window)
        total = len(self._window)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def get_switch_rate(self) -> float:
        """Fraction of consecutive activity pairs that differ.

        0.0 = no switches (single activity), 1.0 = every pair differs.
        """
        if len(self._window) < 2:
            return 0.0
        switches = sum(
            1 for i in range(1, len(self._window))
            if self._window[i] != self._window[i - 1]
        )
        return switches / (len(self._window) - 1)

    def get_flow_score(self) -> float:
        """Composite flow score from 0 (scattered) to 1 (deep focus).

        Combines normalized entropy (inverted) and switch rate (inverted)
        with equal weighting.
        """
        if len(self._window) == 0:
            return 0.5  # neutral

        n_unique = len(set(self._window))
        max_entropy = math.log2(n_unique) if n_unique > 1 else 1.0
        entropy = self.compute_entropy()

        # Invert and normalize entropy: 0 entropy -> 1.0 flow, max entropy -> 0.0 flow
        entropy_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

        # Invert switch rate: 0 switches -> 1.0 flow, all switches -> 0.0 flow
        switch_score = 1.0 - self.get_switch_rate()

        # Composite with equal weights
        return 0.5 * entropy_score + 0.5 * switch_score

    def is_in_flow(self) -> bool:
        """Check if user is currently in flow state."""
        return self.get_flow_score() >= self._flow_threshold

    def get_focus_report(self) -> dict:
        """Generate a focus/flow report."""
        counts = Counter(self._window)
        dominant = counts.most_common(1)[0][0] if counts else None

        return {
            "entropy": round(self.compute_entropy(), 4),
            "flow_score": round(self.get_flow_score(), 4),
            "switch_rate": round(self.get_switch_rate(), 4),
            "dominant_activity": dominant,
            "unique_activities": len(set(self._window)),
            "window_fill": len(self._window) / self._window_size,
            "in_flow": self.is_in_flow(),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_intelligence/test_flow_detector.py -v`
Expected: PASS (all 9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/intelligence/flow_detector.py tests/unit/test_intelligence/test_flow_detector.py
git commit -m "feat: add Shannon entropy-based flow state detector"
```

---

## Chunk 3: Phase 4 — Autonomous Agent Brain

### Task 7: Hierarchical Task Network (HTN) Planner

**Files:**
- Create: `src/homie_core/intelligence/planner.py`
- Test: `tests/unit/test_intelligence/test_planner.py`

**Algorithm:** HTN planning decomposes abstract goals into concrete primitive tasks using decomposition rules. Each rule maps an abstract task to a sequence of subtasks (which may themselves be abstract). Plans are generated by recursively decomposing until only primitive (executable) tasks remain.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_intelligence/test_planner.py
from homie_core.intelligence.planner import HTNPlanner, Task, DecompositionRule


def test_add_rule():
    planner = HTNPlanner()
    rule = DecompositionRule(
        abstract_task="write_feature",
        subtasks=["write_test", "implement", "run_tests"],
        preconditions={"has_spec": True},
    )
    planner.add_rule(rule)
    assert len(planner.get_rules("write_feature")) == 1


def test_primitive_task():
    t = Task(name="open_editor", is_primitive=True)
    assert t.is_primitive


def test_decompose_single_level():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="write_feature",
        subtasks=["write_test", "implement", "run_tests"],
    ))
    planner.mark_primitive("write_test")
    planner.mark_primitive("implement")
    planner.mark_primitive("run_tests")

    plan = planner.plan("write_feature")
    assert plan is not None
    assert [t.name for t in plan] == ["write_test", "implement", "run_tests"]


def test_decompose_multi_level():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="ship_feature",
        subtasks=["write_code", "review", "deploy"],
    ))
    planner.add_rule(DecompositionRule(
        abstract_task="write_code",
        subtasks=["design", "implement", "test"],
    ))
    for p in ["design", "implement", "test", "review", "deploy"]:
        planner.mark_primitive(p)

    plan = planner.plan("ship_feature")
    assert plan is not None
    names = [t.name for t in plan]
    assert names == ["design", "implement", "test", "review", "deploy"]


def test_precondition_check():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="deploy",
        subtasks=["push", "monitor"],
        preconditions={"tests_passing": True},
    ))
    planner.mark_primitive("push")
    planner.mark_primitive("monitor")

    # Without precondition met
    plan = planner.plan("deploy", state={"tests_passing": False})
    assert plan is None

    # With precondition met
    plan = planner.plan("deploy", state={"tests_passing": True})
    assert plan is not None


def test_multiple_rules_picks_first_valid():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="fix_bug",
        subtasks=["hotfix", "deploy"],
        preconditions={"is_critical": True},
    ))
    planner.add_rule(DecompositionRule(
        abstract_task="fix_bug",
        subtasks=["investigate", "fix", "test", "deploy"],
    ))
    for p in ["hotfix", "investigate", "fix", "test", "deploy"]:
        planner.mark_primitive(p)

    # Non-critical bug uses second rule
    plan = planner.plan("fix_bug", state={"is_critical": False})
    assert plan is not None
    assert len(plan) == 4

    # Critical bug uses first rule
    plan = planner.plan("fix_bug", state={"is_critical": True})
    assert plan is not None
    assert len(plan) == 2


def test_max_depth_prevents_infinite():
    planner = HTNPlanner(max_depth=5)
    # Circular decomposition
    planner.add_rule(DecompositionRule(
        abstract_task="loop_a",
        subtasks=["loop_b"],
    ))
    planner.add_rule(DecompositionRule(
        abstract_task="loop_b",
        subtasks=["loop_a"],
    ))
    plan = planner.plan("loop_a")
    assert plan is None  # Should fail gracefully


def test_estimate_cost():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="feature",
        subtasks=["code", "test"],
    ))
    planner.mark_primitive("code", cost=5.0)
    planner.mark_primitive("test", cost=3.0)
    plan = planner.plan("feature")
    total = planner.estimate_cost(plan)
    assert abs(total - 8.0) < 1e-6


def test_serialize_deserialize():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="task",
        subtasks=["a", "b"],
    ))
    planner.mark_primitive("a")
    planner.mark_primitive("b")
    data = planner.serialize()
    planner2 = HTNPlanner.deserialize(data)
    plan = planner2.plan("task")
    assert plan is not None
    assert len(plan) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_intelligence/test_planner.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/homie_core/intelligence/planner.py
"""Hierarchical Task Network (HTN) planner for goal decomposition.

HTN planning decomposes abstract goals into sequences of primitive
(executable) tasks using decomposition rules. Each rule specifies how
an abstract task breaks down into subtasks, with optional preconditions.
Recursion continues until all tasks are primitive.

This enables Homie to reason about complex multi-step goals like
'ship a feature' by breaking them into concrete actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    """A task node in the plan hierarchy."""
    name: str
    is_primitive: bool = False
    cost: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class DecompositionRule:
    """Rule for decomposing an abstract task into subtasks."""
    abstract_task: str
    subtasks: list[str]
    preconditions: dict = field(default_factory=dict)
    priority: int = 0  # higher = tried first


class HTNPlanner:
    """Hierarchical Task Network planner.

    Maintains decomposition rules and primitive task definitions.
    Generates linear plans by recursively decomposing abstract tasks
    into primitive ones.
    """

    def __init__(self, max_depth: int = 20):
        self._max_depth = max_depth
        self._rules: dict[str, list[DecompositionRule]] = {}
        self._primitives: dict[str, Task] = {}

    def add_rule(self, rule: DecompositionRule) -> None:
        """Register a decomposition rule."""
        if rule.abstract_task not in self._rules:
            self._rules[rule.abstract_task] = []
        self._rules[rule.abstract_task].append(rule)
        # Sort by priority (highest first)
        self._rules[rule.abstract_task].sort(
            key=lambda r: r.priority, reverse=True
        )

    def get_rules(self, task_name: str) -> list[DecompositionRule]:
        """Get decomposition rules for a task."""
        return self._rules.get(task_name, [])

    def mark_primitive(self, name: str, cost: float = 1.0, **metadata) -> None:
        """Register a primitive (executable) task."""
        self._primitives[name] = Task(
            name=name, is_primitive=True, cost=cost, metadata=metadata
        )

    def _check_preconditions(
        self, rule: DecompositionRule, state: dict
    ) -> bool:
        """Check if all preconditions of a rule are met."""
        for key, expected in rule.preconditions.items():
            if state.get(key) != expected:
                return False
        return True

    def _decompose(
        self, task_name: str, state: dict, depth: int
    ) -> Optional[list[Task]]:
        """Recursively decompose a task into primitives."""
        if depth > self._max_depth:
            return None

        # If it's primitive, return it directly
        if task_name in self._primitives:
            return [self._primitives[task_name]]

        # Try each decomposition rule
        rules = self._rules.get(task_name, [])
        if not rules:
            return None  # No rules and not primitive

        for rule in rules:
            if not self._check_preconditions(rule, state):
                continue

            # Try to decompose all subtasks
            plan: list[Task] = []
            success = True
            for subtask in rule.subtasks:
                sub_plan = self._decompose(subtask, state, depth + 1)
                if sub_plan is None:
                    success = False
                    break
                plan.extend(sub_plan)

            if success:
                return plan

        return None  # No valid decomposition found

    def plan(
        self, goal: str, state: Optional[dict] = None
    ) -> Optional[list[Task]]:
        """Generate a plan to achieve a goal.

        Returns a list of primitive tasks in execution order,
        or None if no valid plan exists.
        """
        if state is None:
            state = {}
        return self._decompose(goal, state, depth=0)

    def estimate_cost(self, plan: Optional[list[Task]]) -> float:
        """Estimate total cost of a plan."""
        if not plan:
            return 0.0
        return sum(t.cost for t in plan)

    def serialize(self) -> dict:
        return {
            "max_depth": self._max_depth,
            "rules": {
                name: [
                    {
                        "abstract_task": r.abstract_task,
                        "subtasks": r.subtasks,
                        "preconditions": r.preconditions,
                        "priority": r.priority,
                    }
                    for r in rules
                ]
                for name, rules in self._rules.items()
            },
            "primitives": {
                name: {"cost": t.cost, "metadata": t.metadata}
                for name, t in self._primitives.items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> HTNPlanner:
        planner = cls(max_depth=data.get("max_depth", 20))
        for name, rules in data.get("rules", {}).items():
            for r in rules:
                planner.add_rule(DecompositionRule(**r))
        for name, info in data.get("primitives", {}).items():
            planner.mark_primitive(name, cost=info.get("cost", 1.0), **info.get("metadata", {}))
        return planner
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_intelligence/test_planner.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/intelligence/planner.py tests/unit/test_intelligence/test_planner.py
git commit -m "feat: add Hierarchical Task Network (HTN) planner"
```

---

### Task 8: Monte Carlo Tree Search (MCTS/UCT) Action Selector

**Files:**
- Create: `src/homie_core/intelligence/action_selector.py`
- Test: `tests/unit/test_intelligence/test_action_selector.py`

**Algorithm:** Upper Confidence Bound for Trees (UCT) — a variant of Monte Carlo Tree Search. Balances exploration vs exploitation when selecting actions. Each node tracks visit count and reward. The UCB1 formula selects which action to explore: `Q/N + C * sqrt(ln(parent_N) / N)`. Random rollouts estimate action quality.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_intelligence/test_action_selector.py
import math
from homie_core.intelligence.action_selector import MCTSActionSelector, ActionNode


def test_create_root():
    selector = MCTSActionSelector()
    root = selector.create_root(available_actions=["a", "b", "c"])
    assert len(root.children) == 0
    assert root.untried_actions == ["a", "b", "c"]


def test_ucb1_formula():
    node = ActionNode(action="test")
    node.visits = 10
    node.total_reward = 7.0
    parent_visits = 100
    ucb = node.ucb1(parent_visits, exploration_weight=1.414)
    expected_exploit = 7.0 / 10
    expected_explore = 1.414 * math.sqrt(math.log(100) / 10)
    assert abs(ucb - (expected_exploit + expected_explore)) < 1e-4


def test_select_expands_untried():
    selector = MCTSActionSelector(exploration_weight=1.414)
    root = selector.create_root(["a", "b", "c"])
    # First selection should expand an untried action
    selected = selector.select(root)
    assert selected.action in ["a", "b", "c"]
    assert len(root.children) == 1


def test_simulate_returns_reward():
    selector = MCTSActionSelector()

    def reward_fn(action: str, context: dict) -> float:
        return 1.0 if action == "good" else 0.0

    reward = selector.simulate("good", {}, reward_fn)
    assert reward == 1.0


def test_backpropagate():
    parent = ActionNode(action="root")
    child = ActionNode(action="a", parent=parent)
    parent.children.append(child)

    MCTSActionSelector.backpropagate(child, reward=0.8)
    assert child.visits == 1
    assert child.total_reward == 0.8
    assert parent.visits == 1
    assert parent.total_reward == 0.8


def test_best_action_after_search():
    selector = MCTSActionSelector(exploration_weight=1.0, n_iterations=100)

    def reward_fn(action: str, context: dict) -> float:
        rewards = {"good": 0.9, "ok": 0.5, "bad": 0.1}
        return rewards.get(action, 0.0)

    best = selector.search(
        available_actions=["good", "ok", "bad"],
        context={},
        reward_fn=reward_fn,
    )
    assert best == "good"


def test_get_action_scores():
    selector = MCTSActionSelector(n_iterations=50)

    def reward_fn(action: str, context: dict) -> float:
        return 0.8 if action == "x" else 0.2

    selector.search(["x", "y"], {}, reward_fn)
    scores = selector.get_action_scores()
    assert "x" in scores
    assert scores["x"] > scores["y"]


def test_empty_actions():
    selector = MCTSActionSelector()
    best = selector.search([], {}, lambda a, c: 0.0)
    assert best is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_intelligence/test_action_selector.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/homie_core/intelligence/action_selector.py
"""Monte Carlo Tree Search (MCTS) with UCT for action selection.

Uses Upper Confidence Bound for Trees (UCT) to balance exploration
vs exploitation when selecting actions. Each action is evaluated
through random rollouts that estimate its quality. The UCB1 formula
guides the search: Q/N + C * sqrt(ln(parent_N) / N).

This enables Homie to intelligently choose among multiple possible
actions (e.g., which suggestion to surface, which task to prioritize)
by simulating outcomes.

Reference: Kocsis & Szepesvari (2006) "Bandit based Monte-Carlo Planning"
"""
from __future__ import annotations

import math
import random
from typing import Callable, Optional


class ActionNode:
    """A node in the MCTS search tree."""

    def __init__(
        self,
        action: Optional[str] = None,
        parent: Optional[ActionNode] = None,
    ):
        self.action = action
        self.parent = parent
        self.children: list[ActionNode] = []
        self.untried_actions: list[str] = []
        self.visits: int = 0
        self.total_reward: float = 0.0

    @property
    def avg_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1(self, parent_visits: int, exploration_weight: float = 1.414) -> float:
        """Upper Confidence Bound formula.

        Balances exploitation (avg reward) with exploration
        (bonus for less-visited nodes).
        """
        if self.visits == 0:
            return float("inf")
        exploit = self.total_reward / self.visits
        explore = exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        return exploit + explore

    def best_child(self, exploration_weight: float) -> ActionNode:
        """Select child with highest UCB1 score."""
        return max(
            self.children,
            key=lambda c: c.ucb1(self.visits, exploration_weight),
        )


class MCTSActionSelector:
    """Selects best action using Monte Carlo Tree Search.

    Performs n_iterations of select-expand-simulate-backpropagate
    to evaluate available actions and returns the one with highest
    estimated reward.
    """

    def __init__(
        self,
        exploration_weight: float = 1.414,
        n_iterations: int = 200,
        seed: int = 42,
    ):
        self._c = exploration_weight
        self._n_iter = n_iterations
        self._rng = random.Random(seed)
        self._root: Optional[ActionNode] = None

    def create_root(self, available_actions: list[str]) -> ActionNode:
        """Create root node with available actions."""
        root = ActionNode(action="root")
        root.untried_actions = list(available_actions)
        self._root = root
        return root

    def select(self, node: ActionNode) -> ActionNode:
        """Selection phase: traverse tree using UCB1 until expandable node."""
        current = node
        while not current.untried_actions and current.children:
            current = current.best_child(self._c)
        # Expansion: if there are untried actions, expand one
        if current.untried_actions:
            action = current.untried_actions.pop(
                self._rng.randint(0, len(current.untried_actions) - 1)
            )
            child = ActionNode(action=action, parent=current)
            current.children.append(child)
            return child
        return current

    def simulate(
        self,
        action: str,
        context: dict,
        reward_fn: Callable[[str, dict], float],
    ) -> float:
        """Simulation phase: estimate reward via the reward function."""
        return reward_fn(action, context)

    @staticmethod
    def backpropagate(node: ActionNode, reward: float) -> None:
        """Backpropagation phase: update all ancestors with reward."""
        current: Optional[ActionNode] = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def search(
        self,
        available_actions: list[str],
        context: dict,
        reward_fn: Callable[[str, dict], float],
    ) -> Optional[str]:
        """Run MCTS search and return best action.

        Args:
            available_actions: Actions to choose from
            context: Current state context
            reward_fn: Function(action, context) -> reward in [0, 1]

        Returns:
            Best action string, or None if no actions available
        """
        if not available_actions:
            return None

        root = self.create_root(available_actions)

        for _ in range(self._n_iter):
            # Select & expand
            node = self.select(root)
            # Simulate
            if node.action and node.action != "root":
                reward = self.simulate(node.action, context, reward_fn)
                # Backpropagate
                self.backpropagate(node, reward)

        # Return most-visited child (robust selection)
        if not root.children:
            return available_actions[0] if available_actions else None

        best = max(root.children, key=lambda c: c.visits)
        return best.action

    def get_action_scores(self) -> dict[str, float]:
        """Get average reward scores for all explored actions."""
        if not self._root:
            return {}
        return {
            child.action: child.avg_reward
            for child in self._root.children
            if child.action
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_intelligence/test_action_selector.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/intelligence/action_selector.py tests/unit/test_intelligence/test_action_selector.py
git commit -m "feat: add Monte Carlo Tree Search (UCT) action selector"
```

---

### Task 9: Reflective Self-Correction with Platt Scaling

**Files:**
- Create: `src/homie_core/intelligence/self_reflection.py`
- Test: `tests/unit/test_intelligence/test_self_reflection.py`

**Algorithm:** Agent evaluates its own outputs using multiple scoring dimensions (relevance, helpfulness, confidence). Uses Platt scaling (logistic calibration) to convert raw scores into well-calibrated probabilities. Learns from user feedback to improve calibration over time.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_intelligence/test_self_reflection.py
from homie_core.intelligence.self_reflection import SelfReflection, ReflectionResult


def test_score_action():
    sr = SelfReflection()
    result = sr.score_action(
        action="suggest_break",
        context={"flow_score": 0.3, "hours_worked": 4.0},
        features={"relevance": 0.8, "helpfulness": 0.7, "urgency": 0.5},
    )
    assert isinstance(result, ReflectionResult)
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.calibrated_confidence <= 1.0


def test_platt_scaling():
    sr = SelfReflection()
    # Raw score of 0.0 should map to ~0.5 with default params
    calibrated = sr._platt_scale(0.0)
    assert abs(calibrated - 0.5) < 0.2

    # Higher raw scores should give higher calibrated
    low = sr._platt_scale(-2.0)
    high = sr._platt_scale(2.0)
    assert high > low


def test_should_act():
    sr = SelfReflection(action_threshold=0.6)
    result_high = sr.score_action(
        "suggest",
        {},
        {"relevance": 0.9, "helpfulness": 0.9, "urgency": 0.9},
    )
    result_low = sr.score_action(
        "suggest",
        {},
        {"relevance": 0.1, "helpfulness": 0.1, "urgency": 0.1},
    )
    # High-scoring action should pass threshold more easily
    assert result_high.confidence > result_low.confidence


def test_record_feedback_updates_calibration():
    sr = SelfReflection()
    # Record several correct high-confidence predictions
    for _ in range(20):
        sr.record_feedback(predicted_score=0.9, was_correct=True)
    # Record several incorrect low-confidence predictions
    for _ in range(20):
        sr.record_feedback(predicted_score=0.1, was_correct=False)
    # Platt parameters should have shifted
    assert sr._platt_a != 0.0 or sr._platt_b != 0.0


def test_get_calibration_stats():
    sr = SelfReflection()
    sr.record_feedback(0.8, True)
    sr.record_feedback(0.3, False)
    stats = sr.get_calibration_stats()
    assert "total_feedback" in stats
    assert stats["total_feedback"] == 2


def test_reflection_result_fields():
    result = ReflectionResult(
        action="test",
        raw_score=0.7,
        confidence=0.65,
        calibrated_confidence=0.68,
        reasoning={"relevance": 0.8},
    )
    assert result.action == "test"
    assert result.raw_score == 0.7


def test_multi_dimension_scoring():
    sr = SelfReflection(dimension_weights={
        "relevance": 0.5,
        "helpfulness": 0.3,
        "urgency": 0.2,
    })
    result = sr.score_action(
        "notify",
        {},
        {"relevance": 1.0, "helpfulness": 0.0, "urgency": 0.0},
    )
    # Should be weighted towards relevance
    assert result.raw_score > 0.4


def test_serialize_deserialize():
    sr = SelfReflection()
    sr.record_feedback(0.8, True)
    data = sr.serialize()
    sr2 = SelfReflection.deserialize(data)
    assert sr2.get_calibration_stats()["total_feedback"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_intelligence/test_self_reflection.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/homie_core/intelligence/self_reflection.py
"""Reflective self-scoring with Platt-scaled confidence calibration.

The agent evaluates its own actions across multiple scoring dimensions
(relevance, helpfulness, urgency) and converts raw scores into
well-calibrated probability estimates using Platt scaling — a logistic
regression technique that maps raw classifier outputs to true
probabilities.

Platt scaling: P(correct) = 1 / (1 + exp(A*f + B))
where f is the raw score, and A, B are learned from feedback.

Reference: Platt (1999) "Probabilistic Outputs for SVMs"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReflectionResult:
    """Result of self-reflection scoring."""
    action: str
    raw_score: float
    confidence: float
    calibrated_confidence: float
    reasoning: dict = field(default_factory=dict)


class SelfReflection:
    """Self-evaluating agent with calibrated confidence.

    Scores proposed actions across multiple dimensions, combines them
    with configurable weights, and applies Platt scaling to produce
    well-calibrated confidence estimates. Learns from user feedback
    to improve calibration over time.
    """

    def __init__(
        self,
        dimension_weights: Optional[dict[str, float]] = None,
        action_threshold: float = 0.5,
        learning_rate: float = 0.01,
    ):
        self._weights = dimension_weights or {
            "relevance": 0.4,
            "helpfulness": 0.35,
            "urgency": 0.25,
        }
        self._threshold = action_threshold
        self._lr = learning_rate

        # Platt scaling parameters: P = 1/(1 + exp(A*f + B))
        # Start with identity mapping (A=-1, B=0 gives sigmoid)
        self._platt_a: float = -1.0
        self._platt_b: float = 0.0

        # Feedback history for calibration
        self._feedback: list[tuple[float, bool]] = []

    def _platt_scale(self, raw_score: float) -> float:
        """Apply Platt scaling to convert raw score to calibrated probability."""
        logit = self._platt_a * raw_score + self._platt_b
        # Clamp to prevent overflow
        logit = max(-20.0, min(20.0, logit))
        return 1.0 / (1.0 + math.exp(logit))

    def score_action(
        self,
        action: str,
        context: dict,
        features: dict[str, float],
    ) -> ReflectionResult:
        """Score a proposed action across multiple dimensions.

        Args:
            action: Name of the proposed action
            context: Current state context
            features: Scoring dimensions (e.g., relevance, helpfulness, urgency)
                     Each value in [0, 1]

        Returns:
            ReflectionResult with raw, confidence, and calibrated scores
        """
        # Weighted combination of feature scores
        total_weight = 0.0
        raw_score = 0.0
        for dim, weight in self._weights.items():
            if dim in features:
                raw_score += weight * features[dim]
                total_weight += weight

        if total_weight > 0:
            raw_score /= total_weight  # normalize by actual weights used

        # Raw confidence (simple sigmoid of centered score)
        confidence = 1.0 / (1.0 + math.exp(-5.0 * (raw_score - 0.5)))

        # Calibrated confidence via Platt scaling
        calibrated = self._platt_scale(raw_score)

        return ReflectionResult(
            action=action,
            raw_score=round(raw_score, 4),
            confidence=round(confidence, 4),
            calibrated_confidence=round(calibrated, 4),
            reasoning=features,
        )

    def should_act(self, result: ReflectionResult) -> bool:
        """Decide whether to take an action based on calibrated confidence."""
        return result.calibrated_confidence >= self._threshold

    def record_feedback(
        self, predicted_score: float, was_correct: bool
    ) -> None:
        """Record outcome feedback to improve Platt calibration.

        Uses online gradient descent on the logistic loss to update
        Platt scaling parameters A and B.
        """
        self._feedback.append((predicted_score, was_correct))
        target = 1.0 if was_correct else 0.0

        # Current prediction
        p = self._platt_scale(predicted_score)

        # Gradient of log-loss w.r.t. A and B
        # Loss = -[t*log(p) + (1-t)*log(1-p)]
        # dL/dA = (p - t) * f, dL/dB = (p - t)
        error = p - target
        self._platt_a -= self._lr * error * predicted_score
        self._platt_b -= self._lr * error

    def get_calibration_stats(self) -> dict:
        """Get calibration statistics from feedback."""
        if not self._feedback:
            return {"total_feedback": 0, "platt_a": self._platt_a, "platt_b": self._platt_b}

        correct = sum(1 for _, c in self._feedback if c)
        return {
            "total_feedback": len(self._feedback),
            "accuracy": correct / len(self._feedback),
            "platt_a": round(self._platt_a, 4),
            "platt_b": round(self._platt_b, 4),
        }

    def serialize(self) -> dict:
        return {
            "weights": self._weights,
            "threshold": self._threshold,
            "lr": self._lr,
            "platt_a": self._platt_a,
            "platt_b": self._platt_b,
            "feedback": self._feedback,
        }

    @classmethod
    def deserialize(cls, data: dict) -> SelfReflection:
        sr = cls(
            dimension_weights=data.get("weights"),
            action_threshold=data.get("threshold", 0.5),
            learning_rate=data.get("lr", 0.01),
        )
        sr._platt_a = data.get("platt_a", -1.0)
        sr._platt_b = data.get("platt_b", 0.0)
        sr._feedback = [tuple(f) for f in data.get("feedback", [])]
        return sr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_intelligence/test_self_reflection.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/intelligence/self_reflection.py tests/unit/test_intelligence/test_self_reflection.py
git commit -m "feat: add reflective self-correction with Platt-scaled confidence"
```

---

## Chunk 4: Integration & Final Verification

### Task 10: Integration — Wire Phases 2-4 into ObserverLoop and Integration Tests

**Files:**
- Modify: `src/homie_core/intelligence/observer_loop.py`
- Modify: `tests/unit/test_neural/test_integration.py`

- [ ] **Step 1: Write the integration test**

```python
# Add to tests/unit/test_neural/test_integration.py

from homie_core.neural.rhythm_model import CircadianRhythmModel
from homie_core.neural.behavioral_profile import BehavioralProfile
from homie_core.neural.preference_engine import PreferenceEngine
from homie_core.intelligence.workflow_predictor import WorkflowPredictor
from homie_core.intelligence.anomaly_detector import AnomalyDetector
from homie_core.intelligence.flow_detector import FlowDetector
from homie_core.intelligence.planner import HTNPlanner, DecompositionRule
from homie_core.intelligence.action_selector import MCTSActionSelector
from homie_core.intelligence.self_reflection import SelfReflection


def test_full_neural_pipeline():
    """End-to-end test: Phase 1 + 2 + 3 components work together."""
    wm = WorkingMemory()
    tg = TaskGraph()
    context_engine = SemanticContextEngine(embed_fn=_fake_embed, embed_dim=4)
    classifier = ActivityClassifier(embed_fn=_fake_embed, embed_dim=4)
    classifier._init_prototypes()
    rhythm = CircadianRhythmModel()
    profile = BehavioralProfile(embed_dim=4)
    prefs = PreferenceEngine()
    workflow = WorkflowPredictor()
    flow = FlowDetector(window_size=10)

    loop = ObserverLoop(
        working_memory=wm,
        task_graph=tg,
        context_engine=context_engine,
        activity_classifier=classifier,
        rhythm_model=rhythm,
        behavioral_profile=profile,
        preference_engine=prefs,
        workflow_predictor=workflow,
        flow_detector=flow,
    )

    window = WindowInfo(
        title="engine.py - Homie",
        process_name="Code.exe",
        pid=1234,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    loop._handle_window_change(window)

    # Phase 1 components should be updated
    assert any(v != 0.0 for v in context_engine.get_context_vector())
    assert wm.get("activity_type") is not None

    # Phase 2 components should be updated
    assert rhythm._hourly_buckets  # has recorded activity
    assert profile.sample_count > 0
    assert prefs.get_preferences("activity")

    # Phase 3 components should be updated
    assert flow._window  # has recorded activity
    report = flow.get_focus_report()
    assert "flow_score" in report


def test_phase4_autonomous_pipeline():
    """Phase 4 components work together for decision-making."""
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="help_user",
        subtasks=["assess", "suggest", "verify"],
    ))
    for p in ["assess", "suggest", "verify"]:
        planner.mark_primitive(p, cost=1.0)

    plan = planner.plan("help_user")
    assert plan is not None
    assert len(plan) == 3

    # MCTS selects best action
    selector = MCTSActionSelector(n_iterations=50)
    best = selector.search(
        ["suggest_break", "suggest_resource", "stay_quiet"],
        {"flow_score": 0.3},
        lambda a, c: 0.8 if a == "suggest_break" and c.get("flow_score", 1) < 0.5 else 0.2,
    )
    assert best == "suggest_break"

    # Self-reflection evaluates the action
    reflection = SelfReflection()
    result = reflection.score_action(
        best,
        {"flow_score": 0.3},
        {"relevance": 0.9, "helpfulness": 0.8, "urgency": 0.6},
    )
    assert result.calibrated_confidence > 0.5
```

- [ ] **Step 2: Modify ObserverLoop to accept Phase 2+3 components**

Add the following optional parameters to `ObserverLoop.__init__()` and wire them into `_handle_window_change()`:

```python
# Add to imports (TYPE_CHECKING block):
from homie_core.neural.rhythm_model import CircadianRhythmModel
from homie_core.neural.behavioral_profile import BehavioralProfile
from homie_core.neural.preference_engine import PreferenceEngine
from homie_core.intelligence.workflow_predictor import WorkflowPredictor
from homie_core.intelligence.flow_detector import FlowDetector

# Add to __init__ params:
rhythm_model: Optional[CircadianRhythmModel] = None,
behavioral_profile: Optional[BehavioralProfile] = None,
preference_engine: Optional[PreferenceEngine] = None,
workflow_predictor: Optional[WorkflowPredictor] = None,
flow_detector: Optional[FlowDetector] = None,

# Store as instance vars:
self._rhythm = rhythm_model
self._profile = behavioral_profile
self._prefs = preference_engine
self._workflow = workflow_predictor
self._flow = flow_detector

# Add to end of _handle_window_change():
# Phase 2: Personal Neural Profile
if self._rhythm:
    from datetime import datetime
    hour = datetime.now().hour
    self._rhythm.record_activity(
        hour=hour,
        productivity_score=0.7,  # default; refined by activity scores
        activity_type=top if self._activity_classifier else None,
    )
if self._profile and self._context_engine:
    vec = self._context_engine.get_context_vector()
    if any(v != 0.0 for v in vec):
        self._profile.observe(vec)
if self._prefs and self._activity_classifier:
    scores = self._activity_classifier.classify(window.process_name, window.title)
    top = max(scores, key=scores.get)
    self._prefs.record("activity", top, scores[top])
    self._prefs.record("tool", window.process_name, 1.0)

# Phase 3: Predictive Intelligence
if self._workflow:
    activity = self._wm.get("activity_type", "unknown")
    self._workflow.observe(activity)
if self._flow:
    activity = self._wm.get("activity_type", "unknown")
    self._flow.record_activity(activity)
    self._wm.update("flow_score", self._flow.get_flow_score())
    self._wm.update("in_flow", self._flow.is_in_flow())
```

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/homie_core/intelligence/observer_loop.py tests/unit/test_neural/test_integration.py
git commit -m "feat: integrate Phase 2-4 components into ObserverLoop pipeline"
```
