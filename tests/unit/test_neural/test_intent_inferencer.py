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
