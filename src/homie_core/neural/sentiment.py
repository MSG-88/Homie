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
