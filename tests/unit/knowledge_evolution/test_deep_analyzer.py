# tests/unit/knowledge_evolution/test_deep_analyzer.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.knowledge.intake.deep_analyzer import DeepAnalyzer


class TestDeepAnalyzer:
    def test_analyze_returns_entities_and_relationships(self):
        inference_fn = MagicMock(return_value='{"entities": [{"name": "UserService", "type": "class"}], "relationships": [{"subject": "UserService", "predicate": "handles", "object": "user operations"}]}')
        analyzer = DeepAnalyzer(inference_fn=inference_fn)
        result = analyzer.analyze("class UserService:\n    pass", file_path="service.py")
        assert "entities" in result
        assert "relationships" in result

    def test_analyze_without_llm_returns_empty(self):
        analyzer = DeepAnalyzer(inference_fn=None)
        result = analyzer.analyze("some code", file_path="test.py")
        assert result["entities"] == []
        assert result["relationships"] == []

    def test_handles_malformed_llm_response(self):
        inference_fn = MagicMock(return_value="not valid json")
        analyzer = DeepAnalyzer(inference_fn=inference_fn)
        result = analyzer.analyze("code", file_path="test.py")
        assert result["entities"] == []

    def test_truncates_long_content(self):
        inference_fn = MagicMock(return_value='{"entities": [], "relationships": []}')
        analyzer = DeepAnalyzer(inference_fn=inference_fn, max_content_chars=100)
        long_content = "x" * 500
        analyzer.analyze(long_content, file_path="test.py")
        # Should have been called with truncated content
        call_args = inference_fn.call_args[0][0]
        assert len(call_args) < 500
