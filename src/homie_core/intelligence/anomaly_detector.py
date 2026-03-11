from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any


_EULER = 0.5772156649


def _harmonic(n: int) -> float:
    """Approximate harmonic number via ln(n) + Euler-Mascheroni constant."""
    if n <= 0:
        return 0.0
    return math.log(n) + _EULER


def _avg_path_length(n: int) -> float:
    """Expected average path length of unsuccessful search in a BST.

    Formula: c(n) = 2*H(n-1) - 2*(n-1)/n
    Reference: Liu, Ting & Zhou (2008).
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * _harmonic(n - 1) - 2.0 * (n - 1) / n


@dataclass
class _IsolationTree:
    """A single node in an isolation tree."""

    split_feature: int | None = None
    split_value: float | None = None
    left: _IsolationTree | None = None
    right: _IsolationTree | None = None
    size: int = 0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def path_length(self, point: list[float], depth: int = 0) -> float:
        """Return the path length for *point* through this tree."""
        if self.is_leaf():
            return float(depth) + _avg_path_length(self.size)
        if self.split_feature is None:
            return float(depth) + _avg_path_length(self.size)
        val = point[self.split_feature]
        if val < self.split_value:  # type: ignore[operator]
            return self.left.path_length(point, depth + 1)  # type: ignore[union-attr]
        return self.right.path_length(point, depth + 1)  # type: ignore[union-attr]

    def serialize(self) -> dict[str, Any]:
        return {
            "sf": self.split_feature,
            "sv": self.split_value,
            "sz": self.size,
            "l": self.left.serialize() if self.left else None,
            "r": self.right.serialize() if self.right else None,
        }

    @classmethod
    def deserialize(cls, d: dict[str, Any]) -> _IsolationTree:
        node = cls(
            split_feature=d["sf"],
            split_value=d["sv"],
            size=d["sz"],
        )
        if d["l"] is not None:
            node.left = cls.deserialize(d["l"])
        if d["r"] is not None:
            node.right = cls.deserialize(d["r"])
        return node


def _build_tree(
    data: list[list[float]],
    depth: int,
    max_depth: int,
    rng: random.Random,
) -> _IsolationTree:
    """Recursively build an isolation tree."""
    n = len(data)
    if n <= 1 or depth >= max_depth:
        return _IsolationTree(size=n)

    n_features = len(data[0])
    feat = rng.randint(0, n_features - 1)

    col_vals = [row[feat] for row in data]
    lo, hi = min(col_vals), max(col_vals)

    if lo == hi:
        return _IsolationTree(size=n)

    split = rng.uniform(lo, hi)

    left_data = [row for row in data if row[feat] < split]
    right_data = [row for row in data if row[feat] >= split]

    # Edge-case: if split didn't actually partition, return leaf.
    if not left_data or not right_data:
        return _IsolationTree(size=n)

    node = _IsolationTree(split_feature=feat, split_value=split, size=n)
    node.left = _build_tree(left_data, depth + 1, max_depth, rng)
    node.right = _build_tree(right_data, depth + 1, max_depth, rng)
    return node


class AnomalyDetector:
    """Streaming isolation-forest anomaly detector.

    Reference: Liu, Ting & Zhou (2008) "Isolation Forest".
    Score formula: s(x, n) = 2^(-E(h(x)) / c(n))
    """

    def __init__(
        self,
        n_trees: int = 100,
        sample_size: int = 256,
        threshold: float = 0.6,
        seed: int | None = None,
    ) -> None:
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.threshold = threshold
        self._rng = random.Random(seed)
        self._trees: list[_IsolationTree] = []
        self._buffer: list[list[float]] = []
        self._stream_idx = 0  # round-robin index for tree replacement

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: list[list[float]]) -> None:
        """Build the forest from *data*."""
        self._buffer = list(data)
        max_depth = math.ceil(math.log2(max(self.sample_size, 2)))
        self._trees = []
        for _ in range(self.n_trees):
            if len(data) <= self.sample_size:
                sample = list(data)
            else:
                sample = self._rng.sample(data, self.sample_size)
            tree = _build_tree(sample, 0, max_depth, self._rng)
            self._trees.append(tree)

    def score(self, point: list[float]) -> float:
        """Return anomaly score in [0, 1]. Higher = more anomalous."""
        if not self._trees:
            return 0.5  # neutral when no forest exists

        c = _avg_path_length(self.sample_size)
        if c == 0.0:
            return 0.5

        avg_pl = sum(t.path_length(point) for t in self._trees) / len(self._trees)
        return 2.0 ** (-avg_pl / c)

    def is_anomaly(self, point: list[float]) -> bool:
        """Return True when the anomaly score exceeds the threshold."""
        return self.score(point) > self.threshold

    def stream_update(self, point: list[float]) -> None:
        """Incorporate a new data point and rebuild one tree (round-robin)."""
        self._buffer.append(point)
        if len(self._buffer) > self.sample_size * 4:
            self._buffer = self._buffer[-self.sample_size * 4 :]

        max_depth = math.ceil(math.log2(max(self.sample_size, 2)))
        if len(self._buffer) <= self.sample_size:
            sample = list(self._buffer)
        else:
            sample = self._rng.sample(self._buffer, self.sample_size)

        new_tree = _build_tree(sample, 0, max_depth, self._rng)

        if self._trees:
            idx = self._stream_idx % len(self._trees)
            self._trees[idx] = new_tree
            self._stream_idx += 1
        else:
            self._trees.append(new_tree)

    def get_feature_importance(self) -> list[float]:
        """Return per-feature importance based on split frequency."""
        if not self._trees:
            return []

        # Determine dimensionality from a tree.
        n_features = self._infer_n_features()
        if n_features == 0:
            return []

        counts = [0.0] * n_features
        for tree in self._trees:
            self._count_splits(tree, counts)

        total = sum(counts)
        if total == 0:
            return [1.0 / n_features] * n_features
        return [c / total for c in counts]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> str:
        """Serialize the detector to a JSON string."""
        return json.dumps({
            "n_trees": self.n_trees,
            "sample_size": self.sample_size,
            "threshold": self.threshold,
            "stream_idx": self._stream_idx,
            "trees": [t.serialize() for t in self._trees],
        })

    @classmethod
    def deserialize(cls, state: str) -> AnomalyDetector:
        """Reconstruct a detector from a serialized JSON string."""
        d = json.loads(state)
        ad = cls(
            n_trees=d["n_trees"],
            sample_size=d["sample_size"],
            threshold=d["threshold"],
        )
        ad._stream_idx = d["stream_idx"]
        ad._trees = [_IsolationTree.deserialize(t) for t in d["trees"]]
        return ad

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _infer_n_features(self) -> int:
        """Infer dimensionality from buffer data or tree splits."""
        if self._buffer:
            return len(self._buffer[0])
        max_feat = -1
        for tree in self._trees:
            m = self._max_feature(tree)
            if m > max_feat:
                max_feat = m
        return max_feat + 1 if max_feat >= 0 else 0

    @staticmethod
    def _max_feature(node: _IsolationTree) -> int:
        best = -1
        if node.split_feature is not None:
            best = node.split_feature
        if node.left:
            best = max(best, AnomalyDetector._max_feature(node.left))
        if node.right:
            best = max(best, AnomalyDetector._max_feature(node.right))
        return best

    @staticmethod
    def _count_splits(node: _IsolationTree, counts: list[float]) -> None:
        if node.split_feature is not None:
            counts[node.split_feature] += 1.0
        if node.left:
            AnomalyDetector._count_splits(node.left, counts)
        if node.right:
            AnomalyDetector._count_splits(node.right, counts)
