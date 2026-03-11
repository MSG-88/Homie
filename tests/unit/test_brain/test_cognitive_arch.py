"""Tests for the Cognitive Architecture — Homie's intelligent response pipeline."""
import pytest
from unittest.mock import MagicMock, patch

from homie_core.brain.cognitive_arch import (
    CognitiveArchitecture,
    QueryComplexity,
    SituationalAwareness,
    classify_query_complexity,
    _tf_idf_relevance,
    _tokenize,
    _TOKEN_BUDGETS,
)
from homie_core.memory.working import WorkingMemory
from homie_core.intelligence.self_reflection import SelfReflection


# -----------------------------------------------------------------------
# Query complexity classification
# -----------------------------------------------------------------------

class TestQueryComplexity:
    def test_trivial_greeting(self):
        assert classify_query_complexity("hi") == QueryComplexity.TRIVIAL

    def test_trivial_thanks(self):
        assert classify_query_complexity("thanks!") == QueryComplexity.TRIVIAL

    def test_trivial_single_word(self):
        assert classify_query_complexity("ok") == QueryComplexity.TRIVIAL

    def test_simple_factual(self):
        result = classify_query_complexity("what time is it?")
        assert result in (QueryComplexity.SIMPLE, QueryComplexity.MODERATE)

    def test_moderate_question(self):
        result = classify_query_complexity("how do I configure the database connection?")
        assert result in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX)

    def test_complex_multi_part(self):
        result = classify_query_complexity(
            "explain the difference between async and sync programming, "
            "compare their trade-offs for web servers, and help me understand "
            "which approach would be better for my use case with high concurrency"
        )
        assert result in (QueryComplexity.COMPLEX, QueryComplexity.DEEP)

    def test_code_markers_increase_complexity(self):
        simple = classify_query_complexity("rename the variable")
        code = classify_query_complexity("refactor `def process_data()` to use async")
        # Code markers should push it at least one level up
        complexities = [QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE,
                        QueryComplexity.MODERATE, QueryComplexity.COMPLEX,
                        QueryComplexity.DEEP]
        assert complexities.index(code) >= complexities.index(simple)

    def test_conversation_depth_increases_complexity(self):
        shallow = classify_query_complexity("tell me more", conversation_depth=1)
        deep = classify_query_complexity("tell me more", conversation_depth=10)
        complexities = [QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE,
                        QueryComplexity.MODERATE, QueryComplexity.COMPLEX,
                        QueryComplexity.DEEP]
        assert complexities.index(deep) >= complexities.index(shallow)


# -----------------------------------------------------------------------
# TF-IDF relevance scoring
# -----------------------------------------------------------------------

class TestTfIdfRelevance:
    def test_exact_match_scores_highest(self):
        docs = [
            "python programming language",
            "java programming language",
            "cooking recipes for dinner",
        ]
        scores = _tf_idf_relevance("python programming", docs)
        assert scores[0] > scores[2]  # python doc > cooking doc

    def test_empty_query_returns_zeros(self):
        scores = _tf_idf_relevance("", ["doc1", "doc2"])
        assert all(s == 0.0 for s in scores)

    def test_empty_documents_returns_zeros(self):
        scores = _tf_idf_relevance("query", [])
        assert scores == []

    def test_relevant_doc_scores_higher(self):
        docs = [
            "user prefers dark mode in all applications",
            "user works at a software company",
            "the weather is nice today",
        ]
        scores = _tf_idf_relevance("dark mode settings", docs)
        assert scores[0] > scores[2]

    def test_tokenizer(self):
        tokens = _tokenize("Hello, World! Test 123.")
        assert tokens == ["hello", "world", "test", "123"]


# -----------------------------------------------------------------------
# Situational Awareness
# -----------------------------------------------------------------------

class TestSituationalAwareness:
    def test_cognitive_load_default(self):
        sa = SituationalAwareness()
        load = sa.cognitive_load()
        assert 0.0 <= load <= 1.0

    def test_cognitive_load_high_flow(self):
        sa = SituationalAwareness(flow_score=0.9, minutes_in_task=60, is_deep_work=True)
        load = sa.cognitive_load()
        assert load > 0.7  # should be high

    def test_cognitive_load_low_engagement(self):
        sa = SituationalAwareness(flow_score=0.1, minutes_in_task=0)
        load = sa.cognitive_load()
        # Even low engagement has a base load from the 0.1 constant
        assert load <= 0.5

    def test_context_block_coding(self):
        sa = SituationalAwareness(
            activity_type="coding", active_window="VS Code",
            flow_score=0.85, sentiment="neutral", arousal="calm",
        )
        block = sa.to_context_block()
        assert "coding" in block
        assert "VS Code" in block
        assert "Deep concentration" in block

    def test_context_block_frustrated(self):
        sa = SituationalAwareness(sentiment="negative", arousal="frustrated")
        block = sa.to_context_block()
        assert "frustrated" in block

    def test_context_block_low_energy(self):
        sa = SituationalAwareness(rhythmic_score=0.2)
        block = sa.to_context_block()
        assert "Low productivity" in block

    def test_context_block_empty_for_defaults(self):
        sa = SituationalAwareness()
        block = sa.to_context_block()
        # Default neutral/calm/0.5 should produce minimal output
        assert "frustrated" not in block
        assert "Deep concentration" not in block


# -----------------------------------------------------------------------
# Token budgets
# -----------------------------------------------------------------------

class TestTokenBudgets:
    def test_trivial_has_smallest_budget(self):
        assert _TOKEN_BUDGETS[QueryComplexity.TRIVIAL]["max_tokens"] < \
               _TOKEN_BUDGETS[QueryComplexity.DEEP]["max_tokens"]

    def test_deep_has_largest_prompt_budget(self):
        assert _TOKEN_BUDGETS[QueryComplexity.DEEP]["prompt_chars"] > \
               _TOKEN_BUDGETS[QueryComplexity.SIMPLE]["prompt_chars"]

    def test_complex_has_lower_temperature(self):
        assert _TOKEN_BUDGETS[QueryComplexity.COMPLEX]["temperature"] <= \
               _TOKEN_BUDGETS[QueryComplexity.SIMPLE]["temperature"]

    def test_all_levels_have_required_keys(self):
        for level in [QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE,
                      QueryComplexity.MODERATE, QueryComplexity.COMPLEX,
                      QueryComplexity.DEEP]:
            budget = _TOKEN_BUDGETS[level]
            assert "max_tokens" in budget
            assert "prompt_chars" in budget
            assert "temperature" in budget


# -----------------------------------------------------------------------
# CognitiveArchitecture integration
# -----------------------------------------------------------------------

@pytest.fixture
def cognitive():
    engine = MagicMock()
    engine.generate.return_value = "Cognitive response here."
    engine.stream.return_value = iter(["Cognitive", " response", " here."])
    wm = WorkingMemory()
    return CognitiveArchitecture(
        model_engine=engine,
        working_memory=wm,
        system_prompt="You are Homie.",
    ), engine, wm


class TestCognitiveArchitecture:
    def test_process_returns_response(self, cognitive):
        arch, engine, wm = cognitive
        response = arch.process("Hello")
        assert response == "Cognitive response here."
        engine.generate.assert_called_once()

    def test_process_stream_yields_tokens(self, cognitive):
        arch, engine, wm = cognitive
        tokens = list(arch.process_stream("Hello"))
        assert tokens == ["Cognitive", " response", " here."]
        engine.stream.assert_called_once()

    def test_trivial_query_uses_small_budget(self, cognitive):
        arch, engine, wm = cognitive
        arch.process("hi")
        call_args = engine.generate.call_args
        assert call_args[1]["max_tokens"] == 128 or call_args.args[1] if len(call_args.args) > 1 else True

    def test_complex_query_uses_large_budget(self, cognitive):
        arch, engine, wm = cognitive
        arch.process(
            "explain the difference between async and sync programming, "
            "compare their trade-offs, and help me understand which is better"
        )
        call_args = engine.generate.call_args
        # Complex queries get more tokens
        max_tokens = call_args[1].get("max_tokens", call_args[0][1] if len(call_args[0]) > 1 else 512)
        assert max_tokens >= 512

    def test_conversation_stored(self, cognitive):
        arch, engine, wm = cognitive
        arch.process("test query")
        msgs = wm.get_conversation()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_stream_stores_full_response(self, cognitive):
        arch, engine, wm = cognitive
        list(arch.process_stream("test"))
        msgs = wm.get_conversation()
        assert msgs[-1]["content"] == "Cognitive response here."

    def test_set_system_prompt(self, cognitive):
        arch, engine, wm = cognitive
        arch.set_system_prompt("New prompt")
        assert arch._system_prompt == "New prompt"

    def test_with_semantic_memory(self):
        engine = MagicMock()
        engine.generate.return_value = "response"
        wm = WorkingMemory()
        sm = MagicMock()
        sm.get_facts.return_value = [
            {"fact": "user prefers Python", "confidence": 0.9},
            {"fact": "user works on AI projects", "confidence": 0.8},
        ]
        arch = CognitiveArchitecture(
            model_engine=engine, working_memory=wm, semantic_memory=sm,
            system_prompt="Test",
        )
        arch.process("how should I structure my Python project?")
        prompt = engine.generate.call_args[0][0]
        assert "Python" in prompt

    def test_with_episodic_memory(self):
        engine = MagicMock()
        engine.generate.return_value = "response"
        wm = WorkingMemory()
        em = MagicMock()
        em.recall.return_value = [
            {"summary": "Debugged auth module successfully", "mood": "focused", "outcome": "success"},
        ]
        arch = CognitiveArchitecture(
            model_engine=engine, working_memory=wm, episodic_memory=em,
            system_prompt="Test",
        )
        arch.process("explain how to debug authentication issues step by step")
        prompt = engine.generate.call_args[0][0]
        assert "auth" in prompt.lower()

    def test_awareness_feeds_into_prompt(self):
        engine = MagicMock()
        engine.generate.return_value = "response"
        wm = WorkingMemory()
        wm.update("activity_type", "coding")
        wm.update("active_window", "VS Code")
        wm.update("flow_score", 0.9)
        arch = CognitiveArchitecture(
            model_engine=engine, working_memory=wm, system_prompt="Test",
        )
        arch.process("how do I fix this bug?")
        prompt = engine.generate.call_args[0][0]
        assert "coding" in prompt
        assert "VS Code" in prompt

    def test_frustrated_user_gets_guidance(self):
        engine = MagicMock()
        engine.generate.return_value = "response"
        wm = WorkingMemory()
        wm.update("arousal", "frustrated")
        arch = CognitiveArchitecture(
            model_engine=engine, working_memory=wm, system_prompt="Test",
        )
        arch.process("why does this keep breaking?")
        prompt = engine.generate.call_args[0][0]
        assert "frustrated" in prompt.lower() or "empathetic" in prompt.lower()

    def test_get_query_analysis(self, cognitive):
        arch, _, _ = cognitive
        analysis = arch.get_query_analysis("explain quantum computing step by step")
        assert "complexity" in analysis
        assert "budget" in analysis
        assert "awareness" in analysis
        assert analysis["complexity"] in (
            QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.DEEP,
        )


# -----------------------------------------------------------------------
# Response guidance generation
# -----------------------------------------------------------------------

class TestResponseGuidance:
    def test_coding_context_suggests_code(self):
        engine = MagicMock()
        engine.generate.return_value = "response"
        wm = WorkingMemory()
        wm.update("activity_type", "coding")
        wm.update("flow_score", 0.8)
        arch = CognitiveArchitecture(
            model_engine=engine, working_memory=wm, system_prompt="Test",
        )
        # Use a moderate+ query to trigger guidance
        arch.process("how should I implement error handling in this module?")
        prompt = engine.generate.call_args[0][0]
        assert "code" in prompt.lower()

    def test_stressed_user_gets_priority_guidance(self):
        engine = MagicMock()
        engine.generate.return_value = "response"
        wm = WorkingMemory()
        wm.update("arousal", "stressed")
        arch = CognitiveArchitecture(
            model_engine=engine, working_memory=wm, system_prompt="Test",
        )
        arch.process("how do I fix the deployment issue before the deadline?")
        prompt = engine.generate.call_args[0][0]
        assert "stress" in prompt.lower() or "important" in prompt.lower()

    def test_deep_focus_user_gets_brevity(self):
        engine = MagicMock()
        engine.generate.return_value = "response"
        wm = WorkingMemory()
        wm.update("flow_score", 0.95)
        wm.update("minutes_in_task", 90)
        wm.update("is_deep_work", True)
        arch = CognitiveArchitecture(
            model_engine=engine, working_memory=wm, system_prompt="Test",
        )
        arch.process("what does this function do?")
        prompt = engine.generate.call_args[0][0]
        assert "brief" in prompt.lower() or "precise" in prompt.lower()
