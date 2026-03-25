"""Tests for the quality filter module."""

from __future__ import annotations

import json

import pytest

from homie_core.finetune.synthetic.quality_filter import QualityFilter


class TestQualityFilter:
    """Tests for QualityFilter."""

    def test_score_parses_json_response(self) -> None:
        def mock_inference(**kwargs: object) -> str:
            return json.dumps({"relevance": 5, "correctness": 4, "naturalness": 3})

        qf = QualityFilter(inference_fn=mock_inference)
        score = qf.score(system="sys", user="hi", response="hello")
        assert score == 3  # min of 5, 4, 3

    def test_passes_when_above_threshold(self) -> None:
        def mock_inference(**kwargs: object) -> str:
            return json.dumps({"relevance": 5, "correctness": 5, "naturalness": 5})

        qf = QualityFilter(inference_fn=mock_inference, min_score=4)
        assert qf.passes(system="sys", user="hi", response="hello") is True

    def test_fails_when_below_threshold(self) -> None:
        def mock_inference(**kwargs: object) -> str:
            return json.dumps({"relevance": 3, "correctness": 2, "naturalness": 1})

        qf = QualityFilter(inference_fn=mock_inference, min_score=4)
        assert qf.passes(system="sys", user="hi", response="hello") is False

    def test_handles_malformed_json(self) -> None:
        def mock_inference(**kwargs: object) -> str:
            return "not valid json at all"

        qf = QualityFilter(inference_fn=mock_inference)
        assert qf.score(system="sys", user="hi", response="hello") == 0

    def test_handles_inference_error(self) -> None:
        def mock_inference(**kwargs: object) -> str:
            raise RuntimeError("model unavailable")

        qf = QualityFilter(inference_fn=mock_inference)
        assert qf.score(system="sys", user="hi", response="hello") == 0
