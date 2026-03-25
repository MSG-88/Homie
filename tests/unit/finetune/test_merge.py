"""Tests for the ModelMerger (LoRA merge + GGUF quantization)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from homie_core.finetune.training.merge import ModelMerger


class TestModelMerger:
    """ModelMerger unit tests."""

    def test_build_modelfile_content(self):
        merger = ModelMerger(base_model="lfm2")
        content = merger.build_modelfile("/models/homie.gguf", system_prompt="You are Homie.")
        assert "FROM /models/homie.gguf" in content
        assert 'SYSTEM """You are Homie."""' in content

    def test_build_modelfile_no_system_prompt(self):
        merger = ModelMerger(base_model="lfm2")
        content = merger.build_modelfile("/models/homie.gguf")
        assert "FROM /models/homie.gguf" in content
        assert "SYSTEM" not in content

    @patch("homie_core.finetune.training.merge.subprocess")
    def test_quantize_calls_llama_cpp(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=0)
        merger = ModelMerger(base_model="lfm2")
        result = merger.quantize("/model/path", "/output/path", "Q4_K_M")
        assert result is True
        args = mock_sub.run.call_args[0][0]
        assert args[0] == "llama-quantize"
        assert "/model/path" in args
        assert "/output/path" in args
        assert "Q4_K_M" in args

    @patch("homie_core.finetune.training.merge.subprocess")
    def test_import_to_ollama_uses_candidate_tag(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=0)
        merger = ModelMerger(base_model="lfm2")
        result = merger.import_to_ollama("/tmp/Modelfile")
        assert result is True
        args = mock_sub.run.call_args[0][0]
        assert "PyMasters/Homie:candidate" in args

    @patch("homie_core.finetune.training.merge.subprocess")
    def test_promote_swaps_candidate_to_latest(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=0)
        merger = ModelMerger(base_model="lfm2")
        result = merger.promote_candidate()
        assert result is True
        args = mock_sub.run.call_args[0][0]
        assert "PyMasters/Homie:candidate" in args
        assert "PyMasters/Homie:latest" in args

    @patch("homie_core.finetune.training.merge.subprocess")
    def test_quantize_failure_returns_false(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=1)
        merger = ModelMerger(base_model="lfm2")
        result = merger.quantize("/bad/path", "/output", "Q4_K_M")
        assert result is False
