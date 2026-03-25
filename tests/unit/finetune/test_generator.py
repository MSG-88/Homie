"""Tests for the synthetic data generator."""

from __future__ import annotations

from homie_core.finetune.synthetic.generator import SyntheticDataGenerator
from homie_core.finetune.synthetic.templates import Domain


class TestSyntheticDataGenerator:
    def _mock_inference(self, **kwargs):
        prompt = kwargs.get("prompt", "")
        if "chosen" in prompt.lower() or "rejected" in prompt.lower():
            return '{"chosen": "Good response here.", "rejected": "Bad response."}'
        if "relevance" in prompt.lower():
            return '{"relevance": 5, "correctness": 5, "naturalness": 5}'
        return "This is a high-quality assistant response."

    def test_generate_sft_batch(self):
        gen = SyntheticDataGenerator(inference_fn=self._mock_inference, seed=42, min_quality=1, rate_limit=0)
        examples = gen.generate_sft(count=5, domain=Domain.INTENT)
        assert len(examples) == 5
        for ex in examples:
            assert "system" in ex and "user" in ex and "assistant" in ex

    def test_generate_dpo_batch(self):
        gen = SyntheticDataGenerator(inference_fn=self._mock_inference, seed=42, min_quality=1, rate_limit=0)
        pairs = gen.generate_dpo(count=3, domain=Domain.SAFETY)
        assert len(pairs) == 3
        for pair in pairs:
            assert "system" in pair and "user" in pair and "chosen" in pair and "rejected" in pair

    def test_domain_allocation(self):
        gen = SyntheticDataGenerator(inference_fn=self._mock_inference, seed=42, min_quality=1, rate_limit=0)
        alloc = gen.compute_allocation(total_sft=100, total_dpo=50, weak_domain=Domain.SAFETY, boost=0.4)
        assert alloc[Domain.SAFETY]["sft"] > 10  # boosted above base 10%
        total_sft = sum(a["sft"] for a in alloc.values())
        assert total_sft == 100

    def test_export_jsonl(self, tmp_path):
        gen = SyntheticDataGenerator(inference_fn=self._mock_inference, seed=42, min_quality=1, rate_limit=0)
        examples = gen.generate_sft(count=3, domain=Domain.INTENT)
        out = tmp_path / "sft.jsonl"
        gen.export_jsonl(examples, out)
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_format_adapter_alpaca_to_chatml(self):
        alpaca = {"instruction": "Help me", "input": "with this", "output": "Sure!"}
        chatml = SyntheticDataGenerator.format_adapter(alpaca, from_format="alpaca")
        assert chatml["user"] == "Help me\nwith this" or "Help me" in chatml["user"]
        assert chatml["assistant"] == "Sure!"
