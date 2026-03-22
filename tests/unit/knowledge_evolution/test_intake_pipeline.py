# tests/unit/knowledge_evolution/test_intake_pipeline.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.knowledge.intake.pipeline import IntakePipeline


class TestIntakePipeline:
    def test_ingest_directory(self, tmp_path):
        (tmp_path / "main.py").write_text("class App:\n    pass\n\ndef run():\n    pass\n")
        (tmp_path / "utils.py").write_text("import os\ndef helper():\n    pass\n")
        graph_store = MagicMock()
        graph_store.find_entity_by_name.return_value = None
        graph_store.add_entity.return_value = "eid-123"
        pipeline = IntakePipeline(graph_store=graph_store, inference_fn=None)
        result = pipeline.ingest(tmp_path)
        assert result["files_scanned"] >= 2
        assert result["entities_created"] >= 0
        assert graph_store.add_entity.called

    def test_ingest_single_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("class Service:\n    pass\n")
        graph_store = MagicMock()
        graph_store.find_entity_by_name.return_value = None
        graph_store.add_entity.return_value = "eid"
        pipeline = IntakePipeline(graph_store=graph_store, inference_fn=None)
        result = pipeline.ingest(f)
        assert result["files_scanned"] == 1

    def test_ingest_empty_directory(self, tmp_path):
        graph_store = MagicMock()
        pipeline = IntakePipeline(graph_store=graph_store, inference_fn=None)
        result = pipeline.ingest(tmp_path)
        assert result["files_scanned"] == 0

    def test_reports_deep_pass_count(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"class C{i}:\n    pass\n" * 20)
        graph_store = MagicMock()
        graph_store.find_entity_by_name.return_value = None
        graph_store.add_entity.return_value = "eid"
        # No inference_fn = no deep pass
        pipeline = IntakePipeline(graph_store=graph_store, inference_fn=None, deep_pass_top_percent=20)
        result = pipeline.ingest(tmp_path)
        assert result["deep_analyzed"] == 0  # no LLM available
