from __future__ import annotations

from collections import defaultdict
from typing import Any


class WorkflowPredictor:
    """Sparse Markov chain workflow predictor.

    Maintains a sparse transition matrix (first-order and optionally higher-order)
    and provides smoothed probability predictions for next workflow states.
    """

    def __init__(self, order: int = 1, smoothing_k: float = 0.0):
        self._order = order
        self._smoothing_k = smoothing_k

        # First-order transitions: from_state -> to_state -> count
        self._transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Higher-order transitions: tuple(context) -> to_state -> count
        self._higher_transitions: dict[tuple[str, ...], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # All observed states
        self._states: set[str] = set()

        # History buffer for building higher-order contexts
        self._history: list[str] = []

    def observe(self, state: str) -> None:
        """Record a state observation and update transition counts."""
        self._states.add(state)

        if self._history:
            prev = self._history[-1]
            self._transitions[prev][state] += 1

            # Higher-order transitions
            for n in range(2, self._order + 1):
                if len(self._history) >= n:
                    context = tuple(self._history[-n:])
                    self._higher_transitions[context][state] += 1

        self._history.append(state)

    def transition_count(self, from_state: str, to_state: str) -> int:
        """Return raw transition count from from_state to to_state."""
        return self._transitions[from_state][to_state]

    def _smoothed_probs(self, counts: dict[str, int]) -> list[tuple[str, float]]:
        """Apply Laplace smoothing and return sorted (state, prob) pairs."""
        k = self._smoothing_k
        n_states = len(self._states) if self._states else 1
        total = sum(counts.values())

        result: list[tuple[str, float]] = []

        if k > 0:
            # Include all known states for smoothing
            for state in self._states:
                count = counts.get(state, 0)
                prob = (count + k) / (total + k * n_states)
                result.append((state, prob))
        else:
            # No smoothing: only states with counts > 0
            for state, count in counts.items():
                if count > 0:
                    prob = count / total if total > 0 else 0.0
                    result.append((state, prob))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def predict_next(self, current: str, top_n: int = 5) -> list[tuple[str, float]]:
        """Predict next states from current state using smoothed probabilities."""
        counts = dict(self._transitions.get(current, {}))
        probs = self._smoothed_probs(counts)
        return probs[:top_n]

    def predict_next_with_context(
        self, context: list[str], top_n: int = 5
    ) -> list[tuple[str, float]]:
        """Predict next state using higher-order context with first-order fallback."""
        ctx = tuple(context)

        # Try higher-order match
        if ctx in self._higher_transitions:
            counts = dict(self._higher_transitions[ctx])
            probs = self._smoothed_probs(counts)
            if probs:
                return probs[:top_n]

        # Fallback to first-order using last element of context
        if context:
            return self.predict_next(context[-1], top_n=top_n)

        return []

    def predict_sequence(self, start: str, length: int = 5) -> list[str]:
        """Greedily predict a sequence of states starting from start."""
        seq: list[str] = []
        current = start
        for _ in range(length):
            preds = self.predict_next(current, top_n=1)
            if not preds:
                break
            next_state = preds[0][0]
            seq.append(next_state)
            current = next_state
        return seq

    def get_transition_matrix(self) -> dict[str, dict[str, float]]:
        """Return the full smoothed transition matrix."""
        matrix: dict[str, dict[str, float]] = {}
        for from_state in self._states:
            counts = dict(self._transitions.get(from_state, {}))
            probs = self._smoothed_probs(counts)
            matrix[from_state] = {s: p for s, p in probs}
        return matrix

    def get_stationary_distribution(
        self, max_iterations: int = 1000, tol: float = 1e-8
    ) -> dict[str, float]:
        """Compute stationary distribution via power iteration."""
        states = sorted(self._states)
        n = len(states)
        if n == 0:
            return {}

        state_idx = {s: i for i, s in enumerate(states)}
        matrix = self.get_transition_matrix()

        # Build row-stochastic matrix as list of lists
        mat: list[list[float]] = []
        for s in states:
            row = [0.0] * n
            row_data = matrix.get(s, {})
            total = sum(row_data.values())
            if total > 0:
                for to_state, prob in row_data.items():
                    row[state_idx[to_state]] = prob
            else:
                # Uniform for absorbing states
                for j in range(n):
                    row[j] = 1.0 / n
            mat.append(row)

        # Power iteration: pi = pi * P
        pi = [1.0 / n] * n
        for _ in range(max_iterations):
            new_pi = [0.0] * n
            for j in range(n):
                for i in range(n):
                    new_pi[j] += pi[i] * mat[i][j]
            # Check convergence
            diff = sum(abs(new_pi[i] - pi[i]) for i in range(n))
            pi = new_pi
            if diff < tol:
                break

        return {states[i]: pi[i] for i in range(n)}

    def serialize(self) -> dict[str, Any]:
        """Serialize predictor state to a dict."""
        transitions: dict[str, dict[str, int]] = {}
        for from_s, to_counts in self._transitions.items():
            transitions[from_s] = dict(to_counts)

        higher: dict[str, dict[str, int]] = {}
        for ctx, to_counts in self._higher_transitions.items():
            key = "|".join(ctx)
            higher[key] = dict(to_counts)

        return {
            "order": self._order,
            "smoothing_k": self._smoothing_k,
            "states": sorted(self._states),
            "transitions": transitions,
            "higher_transitions": higher,
            "history": self._history,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> WorkflowPredictor:
        """Restore predictor from serialized dict."""
        wp = cls(order=data.get("order", 1), smoothing_k=data.get("smoothing_k", 0.0))
        wp._states = set(data.get("states", []))
        wp._history = data.get("history", [])

        for from_s, to_counts in data.get("transitions", {}).items():
            for to_s, count in to_counts.items():
                wp._transitions[from_s][to_s] = count

        for key, to_counts in data.get("higher_transitions", {}).items():
            ctx = tuple(key.split("|"))
            for to_s, count in to_counts.items():
                wp._higher_transitions[ctx][to_s] = count

        return wp
