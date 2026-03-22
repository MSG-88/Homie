# tests/unit/knowledge_evolution/test_value_scorer.py
import pytest
from homie_core.adaptive_learning.knowledge.intake.value_scorer import ValueScorer


class TestValueScorer:
    def test_score_by_import_count(self):
        scorer = ValueScorer()
        extractions = [
            {"file": "core.py", "imports": [], "classes": ["Core"], "functions": []},
            {"file": "utils.py", "imports": ["core"], "classes": [], "functions": ["helper"]},
            {"file": "main.py", "imports": ["core", "utils"], "classes": [], "functions": ["main"]},
        ]
        # Build import reference counts
        import_counts = {"core": 2, "utils": 1}
        scores = scorer.score_files(extractions, import_counts=import_counts)
        # core.py should score highest (most referenced)
        assert scores["core.py"] > scores["utils.py"]

    def test_score_by_size(self):
        scorer = ValueScorer()
        extractions = [
            {"file": "big.py", "line_count": 500, "classes": [], "functions": [], "imports": []},
            {"file": "small.py", "line_count": 10, "classes": [], "functions": [], "imports": []},
        ]
        scores = scorer.score_files(extractions)
        assert scores["big.py"] > scores["small.py"]

    def test_select_top_percent(self):
        scorer = ValueScorer(top_percent=50)
        extractions = [
            {"file": f"file{i}.py", "line_count": i * 10, "classes": [], "functions": [], "imports": []}
            for i in range(10)
        ]
        selected = scorer.select_for_deep_pass(extractions)
        assert len(selected) == 5  # top 50%

    def test_empty_extractions(self):
        scorer = ValueScorer()
        assert scorer.score_files([]) == {}
        assert scorer.select_for_deep_pass([]) == []
