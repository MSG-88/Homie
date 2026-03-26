"""Tests for response conciseness optimizer."""
import pytest
from homie_core.brain.conciseness import detect_depth, ResponseDepth, get_depth_instruction, optimize_prompt_for_conciseness


class TestDetectDepth:
    def test_greeting_is_brief(self):
        assert detect_depth("Hey Homie!") == ResponseDepth.BRIEF
        assert detect_depth("Good morning") == ResponseDepth.BRIEF
        assert detect_depth("Thanks!") == ResponseDepth.BRIEF

    def test_time_question_is_brief(self):
        assert detect_depth("What time is it?") == ResponseDepth.BRIEF

    def test_simple_question_is_brief(self):
        assert detect_depth("How are you?") == ResponseDepth.BRIEF

    def test_explanation_is_detailed(self):
        assert detect_depth("Explain how async/await works in Python with examples") == ResponseDepth.DETAILED

    def test_code_request_is_detailed(self):
        assert detect_depth("Write a Python function that validates email addresses") == ResponseDepth.DETAILED

    def test_planning_is_detailed(self):
        assert detect_depth("Help me prepare a technical design doc outline for a REST API") == ResponseDepth.DETAILED

    def test_medium_question_is_moderate(self):
        assert detect_depth("What emails need my attention?") == ResponseDepth.MODERATE

    def test_short_question_defaults_brief(self):
        assert detect_depth("Why?") == ResponseDepth.BRIEF

    def test_long_question_defaults_detailed(self):
        assert detect_depth("I need you to help me think through the architecture for a new microservices system that handles user authentication and payment processing") == ResponseDepth.DETAILED


class TestGetDepthInstruction:
    def test_brief_instruction(self):
        inst = get_depth_instruction(ResponseDepth.BRIEF)
        assert "1-3 sentences" in inst

    def test_detailed_instruction(self):
        inst = get_depth_instruction(ResponseDepth.DETAILED)
        assert "thorough" in inst.lower() or "structured" in inst.lower()


class TestOptimize:
    def test_injects_guidance(self):
        result = optimize_prompt_for_conciseness("Hi!", "You are Homie.")
        assert "Response guidance" in result
        assert "1-3 sentences" in result

    def test_preserves_system_prompt(self):
        result = optimize_prompt_for_conciseness("Hi!", "You are Homie.")
        assert "You are Homie." in result
