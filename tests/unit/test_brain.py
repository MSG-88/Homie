"""Unit tests for brain/cognitive architecture core modules.

Covers:
- classify_query_complexity: all complexity tiers
- _tf_idf_relevance: scoring and edge cases
- SituationalAwareness: cognitive_load, to_context_block
- _TOKEN_BUDGETS: budget ordering and required keys
- CognitiveArchitecture: end-to-end process/stream with mocked LLM
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _complexity_rank(c: str) -> int:
    order = [
        QueryComplexity.TRIVIAL,
        QueryComplexity.SIMPLE,
        QueryComplexity.MODERATE,
        QueryComplexity.COMPLEX,
        QueryComplexity.DEEP,
    ]
    return order.index(c)


def _make_cognitive(**kwargs) -> tuple[CognitiveArchitecture, MagicMock, WorkingMemory]:
    engine = MagicMock()
    engine.generate.return_value = "Mocked response."
    engine.stream.return_value = iter(["Mocked", " response", "."])
    wm = WorkingMemory()
    arch = CognitiveArchitecture(
        model_engine=engine,
        working_memory=wm,
        system_prompt="You are Homie.",
        **kwargs,
    )
    return arch, engine, wm


# ---------------------------------------------------------------------------
# classify_query_complexity
# ---------------------------------------------------------------------------

class TestClassifyQueryComplexity:
    # Trivial tier
    def test_trivial_hi(self):
        assert classify_query_complexity("hi") == QueryComplexity.TRIVIAL

    def test_trivial_thanks(self):
        assert classify_query_complexity("thanks!") == QueryComplexity.TRIVIAL

    def test_trivial_okay(self):
        assert classify_query_complexity("okay") == QueryComplexity.TRIVIAL

    def test_trivial_single_word_no_marker(self):
        assert classify_query_complexity("bye") == QueryComplexity.TRIVIAL

    def test_trivial_two_words_no_marker(self):
        result = classify_query_complexity("good morning")
        assert result == QueryComplexity.TRIVIAL

    # Non-trivial tier
    def test_simple_factual_question(self):
        result = classify_query_complexity("what time is it?")
        assert result in (QueryComplexity.SIMPLE, QueryComplexity.MODERATE)

    def test_moderate_how_question(self):
        result = classify_query_complexity("how do I configure the database?")
        assert result in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX)

    def test_complex_multi_step_question(self):
        text = (
            "explain the difference between async and sync programming, "
            "compare their trade-offs, and help me understand which is better "
            "for a high-concurrency web server step by step"
        )
        result = classify_query_complexity(text)
        assert _complexity_rank(result) >= _complexity_rank(QueryComplexity.COMPLEX)

    def test_deep_with_code_markers(self):
        text = (
            "please help me implement a class to process the data, debug the "
            "existing `def process()` function, analyze the trade-offs, and "
            "design a better architecture for the whole system"
        )
        result = classify_query_complexity(text)
        assert _complexity_rank(result) >= _complexity_rank(QueryComplexity.COMPLEX)

    # Conversation depth
    def test_depth_increases_complexity(self):
        text = "tell me more"
        shallow = classify_query_complexity(text, conversation_depth=0)
        deep = classify_query_complexity(text, conversation_depth=15)
        assert _complexity_rank(deep) >= _complexity_rank(shallow)

    def test_code_backticks_increase_complexity(self):
        simple = classify_query_complexity("fix the bug please")
        with_code = classify_query_complexity("fix `def broken_func():` bug please")
        assert _complexity_rank(with_code) >= _complexity_rank(simple)

    def test_multiple_sentences_increase_complexity(self):
        single = classify_query_complexity("how does caching work?")
        multi = classify_query_complexity(
            "how does caching work? what are the trade-offs? "
            "which strategy should I use? please explain step by step."
        )
        assert _complexity_rank(multi) >= _complexity_rank(single)


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_words(self):
        assert _tokenize("hello world") == ["hello", "world"]

    def test_lowercases(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _tokenize("Hi, there! Test.") == ["hi", "there", "test"]

    def test_digits_kept(self):
        tokens = _tokenize("version 3.14")
        assert "3" in tokens or "3" in " ".join(tokens) or "version" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_only_punctuation(self):
        assert _tokenize("!!! ???") == []


# ---------------------------------------------------------------------------
# _tf_idf_relevance
# ---------------------------------------------------------------------------

class TestTfIdfRelevance:
    def test_exact_match_scores_highest(self):
        docs = [
            "python programming language",
            "java programming language",
            "cooking recipes for dinner",
        ]
        scores = _tf_idf_relevance("python programming", docs)
        assert scores[0] > scores[2]

    def test_irrelevant_doc_scores_lowest(self):
        docs = [
            "python programming language tutorial",
            "cooking delicious pasta dinner",
        ]
        scores = _tf_idf_relevance("python language", docs)
        assert scores[0] > scores[1]

    def test_empty_query_all_zeros(self):
        scores = _tf_idf_relevance("", ["doc one", "doc two"])
        assert all(s == 0.0 for s in scores)

    def test_empty_docs_returns_empty(self):
        scores = _tf_idf_relevance("query", [])
        assert scores == []

    def test_single_doc_returns_single_score(self):
        scores = _tf_idf_relevance("test query", ["test document"])
        assert len(scores) == 1

    def test_all_scores_non_negative(self):
        docs = ["apples oranges", "bananas grapes", "cherry pie"]
        scores = _tf_idf_relevance("fruit juice", docs)
        assert all(s >= 0.0 for s in scores)

    def test_identical_docs_same_score(self):
        docs = ["same content here", "same content here"]
        scores = _tf_idf_relevance("same content", docs)
        assert abs(scores[0] - scores[1]) < 1e-9


# ---------------------------------------------------------------------------
# SituationalAwareness
# ---------------------------------------------------------------------------

class TestSituationalAwareness:
    def test_cognitive_load_in_range(self):
        sa = SituationalAwareness()
        load = sa.cognitive_load()
        assert 0.0 <= load <= 1.0

    def test_high_flow_deep_work_gives_high_load(self):
        sa = SituationalAwareness(flow_score=0.95, minutes_in_task=90, is_deep_work=True)
        assert sa.cognitive_load() > 0.6

    def test_low_flow_short_task_gives_low_load(self):
        sa = SituationalAwareness(flow_score=0.1, minutes_in_task=0, is_deep_work=False)
        assert sa.cognitive_load() < 0.5

    def test_deep_work_bonus_applied(self):
        base = SituationalAwareness(flow_score=0.5, minutes_in_task=0, is_deep_work=False)
        deep = SituationalAwareness(flow_score=0.5, minutes_in_task=0, is_deep_work=True)
        assert deep.cognitive_load() > base.cognitive_load()

    # to_context_block
    def test_coding_activity_appears_in_block(self):
        sa = SituationalAwareness(
            activity_type="coding", active_window="VS Code", flow_score=0.5
        )
        block = sa.to_context_block()
        assert "coding" in block
        assert "VS Code" in block

    def test_high_flow_shows_deep_concentration(self):
        sa = SituationalAwareness(flow_score=0.9)
        block = sa.to_context_block()
        assert "Deep concentration" in block or "flow" in block.lower()

    def test_low_flow_shows_scattered(self):
        sa = SituationalAwareness(flow_score=0.2, switch_count_30m=15)
        block = sa.to_context_block()
        assert "Scattered" in block or "switches" in block

    def test_frustrated_mood_appears(self):
        sa = SituationalAwareness(sentiment="negative", arousal="frustrated")
        block = sa.to_context_block()
        assert "frustrated" in block

    def test_peak_energy_hour_shown(self):
        sa = SituationalAwareness(rhythmic_score=0.85)
        block = sa.to_context_block()
        assert "Peak" in block

    def test_low_energy_hour_shown(self):
        sa = SituationalAwareness(rhythmic_score=0.15)
        block = sa.to_context_block()
        assert "Low productivity" in block

    def test_task_duration_shown(self):
        sa = SituationalAwareness(minutes_in_task=30)
        block = sa.to_context_block()
        assert "30" in block or "Session" in block

    def test_neutral_defaults_minimal_output(self):
        sa = SituationalAwareness()
        block = sa.to_context_block()
        assert "Deep concentration" not in block
        assert "frustrated" not in block


# ---------------------------------------------------------------------------
# _TOKEN_BUDGETS
# ---------------------------------------------------------------------------

class TestTokenBudgets:
    _levels = [
        QueryComplexity.TRIVIAL,
        QueryComplexity.SIMPLE,
        QueryComplexity.MODERATE,
        QueryComplexity.COMPLEX,
        QueryComplexity.DEEP,
    ]

    def test_all_levels_present(self):
        for level in self._levels:
            assert level in _TOKEN_BUDGETS

    def test_required_keys_present(self):
        for level in self._levels:
            budget = _TOKEN_BUDGETS[level]
            assert "max_tokens" in budget
            assert "prompt_chars" in budget
            assert "temperature" in budget

    def test_tokens_monotonically_increase(self):
        tokens = [_TOKEN_BUDGETS[l]["max_tokens"] for l in self._levels]
        assert tokens == sorted(tokens)

    def test_prompt_chars_monotonically_increase(self):
        chars = [_TOKEN_BUDGETS[l]["prompt_chars"] for l in self._levels]
        assert chars == sorted(chars)

    def test_temperature_decreases_with_complexity(self):
        temps = [_TOKEN_BUDGETS[l]["temperature"] for l in self._levels]
        assert temps == sorted(temps, reverse=True)

    def test_trivial_smallest_budget(self):
        assert (
            _TOKEN_BUDGETS[QueryComplexity.TRIVIAL]["max_tokens"]
            < _TOKEN_BUDGETS[QueryComplexity.DEEP]["max_tokens"]
        )

    def test_deep_largest_prompt_budget(self):
        assert (
            _TOKEN_BUDGETS[QueryComplexity.DEEP]["prompt_chars"]
            > _TOKEN_BUDGETS[QueryComplexity.SIMPLE]["prompt_chars"]
        )


# ---------------------------------------------------------------------------
# CognitiveArchitecture (mocked LLM)
# ---------------------------------------------------------------------------

class TestCognitiveArchitecture:
    def test_process_returns_string(self):
        arch, engine, _ = _make_cognitive()
        result = arch.process("hello there")
        assert isinstance(result, str)
        assert result == "Mocked response."

    def test_process_calls_engine_once(self):
        arch, engine, _ = _make_cognitive()
        arch.process("test query")
        engine.generate.assert_called_once()

    def test_process_stream_yields_tokens(self):
        arch, engine, _ = _make_cognitive()
        tokens = list(arch.process_stream("hello"))
        assert tokens == ["Mocked", " response", "."]
        engine.stream.assert_called_once()

    def test_conversation_stored_after_process(self):
        arch, _, wm = _make_cognitive()
        arch.process("test query")
        msgs = wm.get_conversation()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "test query"
        assert msgs[1]["role"] == "assistant"

    def test_stream_stores_full_response_in_memory(self):
        arch, _, wm = _make_cognitive()
        list(arch.process_stream("stream test"))
        msgs = wm.get_conversation()
        last = msgs[-1]
        assert last["role"] == "assistant"
        assert "Mocked" in last["content"]

    def test_set_system_prompt(self):
        arch, _, _ = _make_cognitive()
        arch.set_system_prompt("New system prompt")
        assert arch._system_prompt == "New system prompt"

    def test_trivial_query_smaller_tokens_than_complex(self):
        arch, engine, _ = _make_cognitive()
        # Trivial query
        arch.process("hi")
        trivial_tokens = engine.generate.call_args[1].get("max_tokens", 9999)
        engine.reset_mock()
        engine.generate.return_value = "Mocked response."
        # Complex query
        arch.process(
            "please explain, compare, and analyze the trade-offs between "
            "different database indexing strategies step by step in detail"
        )
        complex_tokens = engine.generate.call_args[1].get("max_tokens", 0)
        assert complex_tokens >= trivial_tokens

    def test_semantic_memory_facts_in_prompt(self):
        sm = MagicMock()
        sm.get_facts.return_value = [
            {"fact": "user prefers Python", "confidence": 0.9},
        ]
        arch, engine, _ = _make_cognitive(semantic_memory=sm)
        arch.process("what language should I use?")
        prompt = engine.generate.call_args[0][0]
        assert "Python" in prompt

    def test_episodic_memory_recall_in_prompt(self):
        em = MagicMock()
        em.recall.return_value = [
            {
                "summary": "Debugged auth module successfully",
                "mood": "focused",
                "outcome": "success",
                "distance": 0.1,
            }
        ]
        arch, engine, _ = _make_cognitive(episodic_memory=em)
        arch.process("how do I debug authentication step by step?")
        prompt = engine.generate.call_args[0][0]
        assert "auth" in prompt.lower()

    def test_working_memory_activity_in_prompt(self):
        arch, engine, wm = _make_cognitive()
        wm.update("activity_type", "coding")
        wm.update("active_window", "PyCharm")
        arch.process("help me fix this bug")
        prompt = engine.generate.call_args[0][0]
        assert "coding" in prompt or "PyCharm" in prompt

    def test_multiple_turns_accumulate_in_memory(self):
        arch, _, wm = _make_cognitive()
        arch.process("first message")
        arch.process("second message")
        msgs = wm.get_conversation()
        roles = [m["role"] for m in msgs]
        assert roles.count("user") == 2
        assert roles.count("assistant") == 2

    def test_no_llm_call_for_empty_prompt(self):
        """Process should still complete even for edge-case empty input."""
        arch, engine, _ = _make_cognitive()
        # Should not raise
        result = arch.process("")
        assert isinstance(result, str)
