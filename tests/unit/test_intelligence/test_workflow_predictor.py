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
    predictions = wp.predict_next("coding", top_n=10)
    probs = {p[0]: p[1] for p in predictions}
    assert probs.get("researching", 0) > 0
    total = sum(p[1] for p in predictions)
    assert abs(total - 1.0) < 0.01


def test_bigram_context():
    wp = WorkflowPredictor(order=2)
    sequence = ["coding", "researching", "writing", "coding", "researching", "writing"]
    for s in sequence:
        wp.observe(s)
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
