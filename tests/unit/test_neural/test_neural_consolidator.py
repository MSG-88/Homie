from homie_core.neural.consolidator import NeuralConsolidator


def _fake_embed(text):
    val = hash(text) % 1000 / 1000.0
    return [val, 1.0 - val, val * 0.5, (1.0 - val) * 0.5]


def test_compute_relevance():
    consolidator = NeuralConsolidator(embed_fn=_fake_embed)
    context = _fake_embed("coding python project")

    mem_relevant = {"summary": "coding python project", "embedding": _fake_embed("coding python project")}
    mem_irrelevant = {"summary": "grocery shopping", "embedding": _fake_embed("grocery shopping list")}

    rel_score = consolidator.compute_relevance(mem_relevant, context)
    irr_score = consolidator.compute_relevance(mem_irrelevant, context)

    assert 0.0 <= rel_score <= 1.0
    assert 0.0 <= irr_score <= 1.0


def test_find_patterns_empty():
    consolidator = NeuralConsolidator(embed_fn=_fake_embed)
    patterns = consolidator.find_patterns([])
    assert patterns == []


def test_find_patterns_groups_similar():
    consolidator = NeuralConsolidator(embed_fn=_fake_embed, similarity_threshold=0.8)

    episodes = [
        {"summary": "coding session A", "embedding": [1.0, 0.0, 0.0, 0.0]},
        {"summary": "coding session B", "embedding": [0.95, 0.05, 0.0, 0.0]},
        {"summary": "meeting with team", "embedding": [0.0, 1.0, 0.0, 0.0]},
    ]

    patterns = consolidator.find_patterns(episodes)
    assert isinstance(patterns, list)


def test_consolidate():
    consolidator = NeuralConsolidator(embed_fn=_fake_embed)
    context = _fake_embed("coding")

    episodes = [
        {"summary": "wrote code for feature", "embedding": _fake_embed("wrote code")},
        {"summary": "debugged test failure", "embedding": _fake_embed("debug test")},
    ]

    result = consolidator.consolidate(episodes, context)
    assert "relevant" in result
    assert "clusters" in result
    assert isinstance(result["relevant"], list)


def test_consolidate_empty():
    consolidator = NeuralConsolidator(embed_fn=_fake_embed)
    result = consolidator.consolidate([], [0.0] * 4)
    assert result["relevant"] == []
    assert result["clusters"] == []
