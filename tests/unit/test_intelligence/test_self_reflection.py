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
    calibrated = sr._platt_scale(0.0)
    assert abs(calibrated - 0.5) < 0.2
    low = sr._platt_scale(-2.0)
    high = sr._platt_scale(2.0)
    assert high > low


def test_should_act():
    sr = SelfReflection(action_threshold=0.6)
    result_high = sr.score_action(
        "suggest", {},
        {"relevance": 0.9, "helpfulness": 0.9, "urgency": 0.9},
    )
    result_low = sr.score_action(
        "suggest", {},
        {"relevance": 0.1, "helpfulness": 0.1, "urgency": 0.1},
    )
    assert result_high.confidence > result_low.confidence


def test_record_feedback_updates_calibration():
    sr = SelfReflection()
    for _ in range(20):
        sr.record_feedback(predicted_score=0.9, was_correct=True)
    for _ in range(20):
        sr.record_feedback(predicted_score=0.1, was_correct=False)
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
        "notify", {},
        {"relevance": 1.0, "helpfulness": 0.0, "urgency": 0.0},
    )
    assert result.raw_score > 0.4


def test_serialize_deserialize():
    sr = SelfReflection()
    sr.record_feedback(0.8, True)
    data = sr.serialize()
    sr2 = SelfReflection.deserialize(data)
    assert sr2.get_calibration_stats()["total_feedback"] == 1
