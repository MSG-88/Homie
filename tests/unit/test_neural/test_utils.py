import math

from homie_core.neural.utils import cosine_similarity, weighted_average, top_k_similar


def test_cosine_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6


def test_cosine_zero_vector_returns_zero():
    a = [0.0, 0.0]
    b = [1.0, 2.0]
    assert cosine_similarity(a, b) == 0.0


def test_weighted_average_equal_weights():
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    weights = [1.0, 1.0]
    result = weighted_average(vecs, weights)
    assert len(result) == 2
    assert abs(result[0] - 0.5) < 1e-6
    assert abs(result[1] - 0.5) < 1e-6


def test_weighted_average_single_vector():
    vecs = [[3.0, 4.0]]
    weights = [1.0]
    result = weighted_average(vecs, weights)
    assert abs(result[0] - 3.0) < 1e-6
    assert abs(result[1] - 4.0) < 1e-6


def test_weighted_average_empty_returns_empty():
    result = weighted_average([], [])
    assert result == []


def test_top_k_similar():
    query = [1.0, 0.0]
    candidates = [
        [1.0, 0.0],   # identical
        [0.0, 1.0],   # orthogonal
        [0.7, 0.7],   # partial match
    ]
    results = top_k_similar(query, candidates, k=2)
    assert len(results) == 2
    assert results[0][0] == 0  # index of most similar
    assert results[0][1] > 0.9  # high similarity


def test_top_k_similar_k_exceeds_candidates():
    query = [1.0, 0.0]
    candidates = [[0.5, 0.5]]
    results = top_k_similar(query, candidates, k=5)
    assert len(results) == 1
