# tests/unit/self_optimizer/test_prompt_optimizer.py
import pytest
from homie_core.adaptive_learning.performance.self_optimizer.prompt_optimizer import PromptOptimizer


class TestPromptOptimizer:
    def test_truncates_history_by_complexity(self):
        opt = PromptOptimizer()
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        trimmed = opt.trim_history(history, complexity="simple")
        assert len(trimmed) <= 5

    def test_preserves_all_history_for_deep(self):
        opt = PromptOptimizer()
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        trimmed = opt.trim_history(history, complexity="deep")
        assert len(trimmed) == 20

    def test_dedup_removes_similar_facts(self):
        opt = PromptOptimizer()
        facts = [
            "User works at Google",
            "The user is employed at Google",
            "Python is a programming language",
        ]
        deduped = opt.deduplicate_facts(facts)
        assert len(deduped) < len(facts)

    def test_dedup_preserves_unique_facts(self):
        opt = PromptOptimizer()
        facts = [
            "User works at Google",
            "Python is a programming language",
            "Homie uses ChromaDB",
        ]
        deduped = opt.deduplicate_facts(facts)
        assert len(deduped) == 3

    def test_compress_prompt_reduces_length(self):
        opt = PromptOptimizer()
        long_prompt = "You are Homie.\n" + "\n".join([f"Fact {i}: user likes thing {i}" for i in range(50)])
        compressed = opt.compress(long_prompt, complexity="simple", max_chars=500)
        assert len(compressed) <= len(long_prompt)

    def test_compress_respects_complexity_budget(self):
        opt = PromptOptimizer()
        prompt = "x" * 10000
        compressed_simple = opt.compress(prompt, complexity="simple", max_chars=1500)
        compressed_deep = opt.compress(prompt, complexity="deep", max_chars=12000)
        assert len(compressed_simple) <= 1500
        assert len(compressed_deep) <= 12000

    def test_middleware_interface(self):
        opt = PromptOptimizer()
        assert hasattr(opt, "modify_prompt")
        assert hasattr(opt, "name")
        assert opt.name == "prompt_optimizer"
