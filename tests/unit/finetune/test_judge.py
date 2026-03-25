"""Tests for the cloud judge."""

from __future__ import annotations

from unittest.mock import MagicMock

from homie_core.finetune.evaluation.judge import Judge


class TestJudge:
    def test_score_parses_response(self):
        inference_fn = MagicMock(return_value="4")
        judge = Judge(inference_fn)
        score = judge.score("some response", "helpfulness")
        assert score == 4.0

    def test_score_handles_verbose_response(self):
        inference_fn = MagicMock(return_value="I'd rate this a 3 out of 5.")
        judge = Judge(inference_fn)
        score = judge.score("some response", "helpfulness")
        assert score == 3.0

    def test_score_handles_error(self):
        inference_fn = MagicMock(side_effect=RuntimeError("API error"))
        judge = Judge(inference_fn)
        score = judge.score("some response", "helpfulness")
        assert score == 2.5

    def test_normalize(self):
        assert Judge.normalize(5.0) == 1.0
        assert Judge.normalize(1.0) == 0.0
        assert Judge.normalize(3.0) == 0.5
