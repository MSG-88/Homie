"""Behavioral eigenvector profile using power iteration PCA."""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any

from homie_core.neural.utils import cosine_similarity


class BehavioralProfile:
    """Builds a behavioral fingerprint from activity embeddings using
    power iteration PCA with deflation to extract principal components."""

    MAX_SAMPLES = 5000

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim
        self._samples: deque[list[float]] = deque(maxlen=self.MAX_SAMPLES)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, embedding: list[float]) -> None:
        """Record a single activity embedding."""
        if len(embedding) != self.embed_dim:
            raise ValueError(
                f"Expected embedding of dim {self.embed_dim}, got {len(embedding)}"
            )
        self._samples.append(list(embedding))

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_mean_vector(self) -> list[float]:
        """Return the element-wise mean of all observed embeddings."""
        n = len(self._samples)
        if n == 0:
            return [0.0] * self.embed_dim
        mean = [0.0] * self.embed_dim
        for sample in self._samples:
            for i in range(self.embed_dim):
                mean[i] += sample[i]
        return [m / n for m in mean]

    def _compute_covariance(self) -> list[list[float]]:
        """Compute the covariance matrix of the centered samples."""
        n = len(self._samples)
        d = self.embed_dim
        if n == 0:
            return [[0.0] * d for _ in range(d)]

        mean = self.get_mean_vector()

        # Center the samples
        centered = []
        for sample in self._samples:
            centered.append([sample[i] - mean[i] for i in range(d)])

        # Covariance: (1/n) * X^T X
        cov = [[0.0] * d for _ in range(d)]
        for row in centered:
            for i in range(d):
                for j in range(d):
                    cov[i][j] += row[i] * row[j]
        for i in range(d):
            for j in range(d):
                cov[i][j] /= n
        return cov

    # ------------------------------------------------------------------
    # Power iteration PCA
    # ------------------------------------------------------------------

    @staticmethod
    def _mat_vec(matrix: list[list[float]], vec: list[float]) -> list[float]:
        """Multiply a square matrix by a vector."""
        d = len(vec)
        result = [0.0] * d
        for i in range(d):
            s = 0.0
            for j in range(d):
                s += matrix[i][j] * vec[j]
            result[i] = s
        return result

    @staticmethod
    def _normalize(vec: list[float]) -> tuple[list[float], float]:
        """Return (unit_vector, norm)."""
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec, 0.0
        return [x / norm for x in vec], norm

    def _power_iteration(
        self, matrix: list[list[float]], max_iter: int = 200, tol: float = 1e-8
    ) -> tuple[list[float], float]:
        """Find the dominant eigenvector/eigenvalue of *matrix* via power iteration."""
        d = len(matrix)
        # Initial guess – use [1, 1, …] normalised
        vec = [1.0] * d
        vec, _ = self._normalize(vec)

        eigenvalue = 0.0
        for _ in range(max_iter):
            new_vec = self._mat_vec(matrix, vec)
            new_vec, eigenvalue = self._normalize(new_vec)
            # Check convergence (absolute cosine similarity ~ 1)
            dot = sum(a * b for a, b in zip(vec, new_vec))
            if abs(abs(dot) - 1.0) < tol:
                vec = new_vec
                break
            vec = new_vec

        # Compute the actual eigenvalue: v^T A v
        av = self._mat_vec(matrix, vec)
        eigenvalue = sum(v * a for v, a in zip(vec, av))
        return vec, eigenvalue

    @staticmethod
    def _deflate(
        matrix: list[list[float]], eigenvec: list[float], eigenvalue: float
    ) -> list[list[float]]:
        """Remove the component of *eigenvec* from *matrix* (deflation)."""
        d = len(matrix)
        deflated = [row[:] for row in matrix]
        for i in range(d):
            for j in range(d):
                deflated[i][j] -= eigenvalue * eigenvec[i] * eigenvec[j]
        return deflated

    def compute_eigenvectors(self, top_k: int = 3) -> list[list[float]]:
        """Return the top-k eigenvectors via power iteration + deflation."""
        cov = self._compute_covariance()
        eigenvectors: list[list[float]] = []
        eigenvalues: list[float] = []

        matrix = cov
        for _ in range(min(top_k, self.embed_dim)):
            vec, val = self._power_iteration(matrix)
            eigenvectors.append(vec)
            eigenvalues.append(val)
            matrix = self._deflate(matrix, vec, val)

        return eigenvectors

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------

    def get_fingerprint(self, top_k: int = 3) -> dict[str, Any]:
        """Return a behavioural fingerprint dictionary."""
        cov = self._compute_covariance()
        eigenvectors: list[list[float]] = []
        eigenvalues: list[float] = []

        matrix = cov
        for _ in range(min(top_k, self.embed_dim)):
            vec, val = self._power_iteration(matrix)
            eigenvectors.append(vec)
            eigenvalues.append(val)
            matrix = self._deflate(matrix, vec, val)

        total_variance = sum(eigenvalues)
        if total_variance > 0:
            explained = [v / total_variance for v in eigenvalues]
        else:
            explained = [0.0] * len(eigenvalues)

        return {
            "eigenvectors": eigenvectors,
            "explained_variance": explained,
            "sample_count": self.sample_count,
            "mean_vector": self.get_mean_vector(),
        }

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare(
        profile1: BehavioralProfile,
        profile2: BehavioralProfile,
        top_k: int = 3,
    ) -> float:
        """Average absolute cosine similarity between top-k eigenvectors.

        When variance is negligible (e.g. all observations identical),
        falls back to cosine similarity of the mean vectors.
        """
        vecs1 = profile1.compute_eigenvectors(top_k)
        vecs2 = profile2.compute_eigenvectors(top_k)

        k = min(len(vecs1), len(vecs2))
        if k == 0:
            return 0.0

        # Check if eigenvalues are near-zero (degenerate case).
        # In that case eigenvectors are arbitrary; compare means instead.
        cov1 = profile1._compute_covariance()
        cov2 = profile2._compute_covariance()
        trace1 = sum(cov1[i][i] for i in range(profile1.embed_dim))
        trace2 = sum(cov2[i][i] for i in range(profile2.embed_dim))
        if trace1 < 1e-12 and trace2 < 1e-12:
            mean1 = profile1.get_mean_vector()
            mean2 = profile2.get_mean_vector()
            norm1 = math.sqrt(sum(x * x for x in mean1))
            norm2 = math.sqrt(sum(x * x for x in mean2))
            if norm1 < 1e-12 or norm2 < 1e-12:
                return 0.0
            return abs(cosine_similarity(mean1, mean2))

        total = 0.0
        for i in range(k):
            sim = cosine_similarity(vecs1[i], vecs2[i])
            total += abs(sim)
        return total / k

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """Serialize the profile to a plain dict."""
        return {
            "embed_dim": self.embed_dim,
            "samples": [list(s) for s in self._samples],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> BehavioralProfile:
        """Reconstruct a profile from a serialized dict."""
        bp = cls(embed_dim=data["embed_dim"])
        for sample in data["samples"]:
            bp.observe(sample)
        return bp
