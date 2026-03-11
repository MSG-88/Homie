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
