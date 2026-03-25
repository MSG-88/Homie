"""Cloud judge for open-ended response scoring."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """Rate this assistant response from 1-5 on the following criteria:
{criteria}

Response to evaluate:
{response}

Return ONLY a single number from 1 to 5."""


class Judge:
    def __init__(self, inference_fn):
        self._inference_fn = inference_fn

    def score(self, response: str, criteria: str) -> float:
        """Call cloud to rate 1-5. Parse number from response. Return 2.5 on error."""
        try:
            prompt = _JUDGE_PROMPT.format(criteria=criteria, response=response)
            raw = self._inference_fn(prompt=prompt, max_tokens=20, temperature=0.0)
            match = re.search(r"[1-5]", raw)
            if match:
                return float(match.group())
            logger.warning("Could not parse score from judge response: %s", raw)
            return 2.5
        except Exception:
            logger.exception("Judge scoring failed")
            return 2.5

    @staticmethod
    def normalize(score: float) -> float:
        """Map 1-5 to 0.0-1.0: (score - 1) / 4"""
        return (score - 1.0) / 4.0
