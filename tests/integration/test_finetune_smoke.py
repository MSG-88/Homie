"""End-to-end smoke test for the recursive finetuning pipeline.

Mocks all GPU/training operations but exercises the actual pipeline orchestration.
"""

import json

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from homie_core.finetune.pipeline import RecursiveFinetuneLoop
from homie_core.finetune.config import FinetuneConfig


class TestFinetuneSmoke:
    @patch("homie_core.finetune.training.qlora_trainer.QLoRATrainer.train_sft")
    @patch("homie_core.finetune.training.qlora_trainer.QLoRATrainer.train_dpo")
    @patch("homie_core.finetune.training.merge.ModelMerger.merge_lora")
    @patch("homie_core.finetune.training.merge.ModelMerger.quantize")
    @patch("homie_core.finetune.training.merge.ModelMerger.import_to_ollama")
    @patch("homie_core.finetune.training.merge.ModelMerger.promote_candidate")
    @patch("time.sleep", return_value=None)
    def test_single_cycle_e2e(
        self,
        mock_sleep,
        mock_promote,
        mock_import,
        mock_quant,
        mock_merge,
        mock_dpo,
        mock_sft,
        tmp_path,
    ):
        mock_sft.return_value = {"loss": 0.5}
        mock_dpo.return_value = {"loss": 0.3}
        mock_merge.return_value = True
        mock_quant.return_value = True
        mock_import.return_value = True
        mock_promote.return_value = True

        def mock_inference(**kwargs):
            prompt = kwargs.get("prompt", "")
            if "relevance" in prompt.lower():
                return '{"relevance": 5, "correctness": 5, "naturalness": 5}'
            if "rate" in prompt.lower() or "1 to 5" in prompt.lower():
                return "5"
            if "chosen" in prompt.lower():
                return '{"chosen": "Good response.", "rejected": "Bad response."}'
            return "This is a helpful response."

        cfg = FinetuneConfig()
        cfg.limits.max_cycles = 1
        cfg.data.sft_per_cycle = 10
        cfg.data.dpo_per_cycle = 5

        pipe = RecursiveFinetuneLoop(
            config=cfg,
            inference_fn=mock_inference,
            ollama_manager=MagicMock(),
            model_registry=MagicMock(),
            base_dir=tmp_path,
        )
        result = pipe.run()
        assert result["cycles_completed"] >= 1
        assert (tmp_path / "state.json").exists()
