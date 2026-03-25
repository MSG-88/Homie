"""Quality filter for scoring synthetic training data.

Uses an inference function to rate assistant responses across three
dimensions and filters out low-quality examples.
"""

from __future__ import annotations

import json
from typing import Callable

_SCORE_PROMPT = """Rate this assistant response on three dimensions (1-5 each).
Return ONLY a JSON object: {{"relevance": N, "correctness": N, "naturalness": N}}

System context: {system}
User message: {user}
Assistant response: {response}"""


class QualityFilter:
    """Score and filter synthetic training examples via LLM inference.

    Parameters
    ----------
    inference_fn:
        Callable invoked as ``inference_fn(prompt=..., max_tokens=100, temperature=0.0)``.
        Must return a string containing a JSON object with relevance,
        correctness, and naturalness scores.
    min_score:
        Minimum score (the *minimum* across the three dimensions) required
        for a sample to pass the filter.  Defaults to 4.
    """

    def __init__(
        self,
        inference_fn: Callable[..., str],
        min_score: int = 4,
    ) -> None:
        self._inference_fn = inference_fn
        self._min_score = min_score

    def score(self, system: str, user: str, response: str) -> int:
        """Score a response by calling the inference function.

        Returns the *minimum* of the three dimension scores, or ``0`` on
        any error (malformed JSON, missing keys, inference failure, etc.).
        """
        try:
            prompt = _SCORE_PROMPT.format(
                system=system, user=user, response=response
            )
            raw = self._inference_fn(
                prompt=prompt, max_tokens=100, temperature=0.0
            )
            data = json.loads(raw)
            return int(
                min(
                    data["relevance"],
                    data["correctness"],
                    data["naturalness"],
                )
            )
        except Exception:  # noqa: BLE001
            return 0

    def passes(self, system: str, user: str, response: str) -> bool:
        """Return ``True`` if the response meets the minimum quality bar."""
        return self.score(system, user, response) >= self._min_score
