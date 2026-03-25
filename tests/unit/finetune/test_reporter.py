"""Tests for the eval reporter with plateau detection."""

from __future__ import annotations

from homie_core.finetune.evaluation.benchmark import BenchmarkResult
from homie_core.finetune.evaluation.reporter import EvalReporter
from homie_core.finetune.synthetic.templates import Domain


def _make_result(overall: float, safety: float = 0.9) -> BenchmarkResult:
    scores = {d: overall for d in Domain}
    scores[Domain.SAFETY] = safety
    return BenchmarkResult(domain_scores=scores, overall_score=overall, case_results=[])


class TestEvalReporter:
    def test_should_promote_when_improved(self):
        reporter = EvalReporter()
        current = _make_result(0.70)
        candidate = _make_result(0.75)
        assert reporter.should_promote(current, candidate) is True

    def test_reject_when_not_improved_enough(self):
        reporter = EvalReporter()
        current = _make_result(0.70)
        candidate = _make_result(0.71)  # only 0.01 improvement, below 0.02 threshold
        assert reporter.should_promote(current, candidate) is False

    def test_reject_when_safety_below_floor(self):
        reporter = EvalReporter()
        current = _make_result(0.70)
        candidate = _make_result(0.80, safety=0.80)  # safety below 0.85 floor
        assert reporter.should_promote(current, candidate) is False

    def test_reject_when_domain_regresses(self):
        reporter = EvalReporter(max_regression=0.05)
        current = _make_result(0.70)
        # Create candidate with one domain regressed more than max_regression
        candidate = _make_result(0.75)
        candidate.domain_scores[Domain.CONTEXT] = 0.60  # regressed 0.10 > 0.05
        assert reporter.should_promote(current, candidate) is False

    def test_plateau_detection(self):
        reporter = EvalReporter(promotion_threshold=0.02, plateau_cycles=3)
        # Record 3 cycles with negligible improvement
        reporter.record_cycle(1, 0.70)
        reporter.record_cycle(2, 0.71)
        reporter.record_cycle(3, 0.715)
        assert reporter.is_plateau() is True

    def test_no_plateau_when_improving(self):
        reporter = EvalReporter(promotion_threshold=0.02, plateau_cycles=3)
        reporter.record_cycle(1, 0.60)
        reporter.record_cycle(2, 0.65)
        reporter.record_cycle(3, 0.70)
        assert reporter.is_plateau() is False

    def test_weakest_domain(self):
        result = _make_result(0.80)
        result.domain_scores[Domain.ORCHESTRATION] = 0.50
        reporter = EvalReporter()
        assert reporter.weakest_domain(result) == Domain.ORCHESTRATION

    def test_save_and_load(self, tmp_path):
        reporter = EvalReporter(promotion_threshold=0.03, safety_floor=0.90)
        reporter.record_cycle(1, 0.65)
        reporter.record_cycle(2, 0.70)
        path = tmp_path / "reporter.json"
        reporter.save(path)
        loaded = EvalReporter.load(path)
        assert loaded._promotion_threshold == 0.03
        assert loaded._safety_floor == 0.90
        assert loaded._cycle_scores == {1: 0.65, 2: 0.70}
