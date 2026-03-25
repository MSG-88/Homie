"""Tests for the 30-test benchmark suite."""

from __future__ import annotations

from unittest.mock import MagicMock

from homie_core.finetune.evaluation.benchmark import (
    DOMAIN_WEIGHTS,
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkSuite,
)
from homie_core.finetune.synthetic.templates import Domain


class TestBenchmarkSuite:
    def test_has_30_cases(self):
        suite = BenchmarkSuite(inference_fn=MagicMock(), judge_fn=MagicMock())
        assert len(suite.cases) == 30

    def test_domain_distribution(self):
        suite = BenchmarkSuite(inference_fn=MagicMock(), judge_fn=MagicMock())
        counts = {}
        for case in suite.cases:
            counts[case.domain] = counts.get(case.domain, 0) + 1
        assert counts[Domain.INTENT] == 8
        assert counts[Domain.CONTEXT] == 6
        assert counts[Domain.CONVERSATIONAL] == 5
        assert counts[Domain.ORCHESTRATION] == 4
        assert counts[Domain.SELF_AWARENESS] == 4
        assert counts[Domain.SAFETY] == 3

    def test_domain_weights_sum_to_1(self):
        total = sum(DOMAIN_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_run_returns_result(self):
        inference_fn = MagicMock(return_value="Hello, I'm Homie, your desktop assistant.")
        judge_fn = MagicMock(return_value=0.8)
        suite = BenchmarkSuite(inference_fn=inference_fn, judge_fn=judge_fn)
        result = suite.run()
        assert isinstance(result, BenchmarkResult)
        assert 0.0 <= result.overall_score <= 1.0
        assert len(result.case_results) == 30
        assert len(result.domain_scores) == len(Domain)

    def test_case_structure(self):
        suite = BenchmarkSuite(inference_fn=MagicMock(), judge_fn=MagicMock())
        for case in suite.cases:
            assert isinstance(case, BenchmarkCase)
            assert isinstance(case.domain, Domain)
            assert isinstance(case.name, str) and len(case.name) > 0
            assert isinstance(case.system_prompt, str) and len(case.system_prompt) > 0
            assert isinstance(case.user_prompt, str) and len(case.user_prompt) > 0
            assert isinstance(case.automated_checks, list) and len(case.automated_checks) > 0
            assert isinstance(case.judge_criteria, str) and len(case.judge_criteria) > 0
            for check in case.automated_checks:
                assert "type" in check
                assert check["type"] in ("contains", "not_contains", "regex", "min_length")
                assert "value" in check
