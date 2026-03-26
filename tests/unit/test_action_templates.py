"""Tests for action-first response templates."""
import pytest
from homie_core.brain.action_templates import (
    ACTION_TEMPLATES, detect_intent, get_action_template,
    get_intent_keywords, ActionTemplate,
)


class TestActionTemplates:
    def test_all_templates_have_required_fields(self):
        for name, tmpl in ACTION_TEMPLATES.items():
            assert tmpl.intent, f"{name} missing intent"
            assert tmpl.description, f"{name} missing description"
            assert tmpl.required_tools, f"{name} missing tools"
            assert tmpl.tool_sequence, f"{name} missing sequence"

    def test_detect_email_intent(self):
        assert detect_intent("Check my emails please") == "check_email"
        assert detect_intent("Any new emails?") == "check_email"

    def test_detect_draft_email(self):
        assert detect_intent("Send an email to John") == "draft_email"
        assert detect_intent("Draft email to the team") == "draft_email"

    def test_detect_weather(self):
        assert detect_intent("What's the weather like?") == "check_weather"

    def test_detect_git(self):
        assert detect_intent("What's the git status?") == "git_status"

    def test_detect_remember(self):
        assert detect_intent("Remember that I prefer dark mode") == "remember_fact"

    def test_detect_recall(self):
        assert detect_intent("What do you know about my projects?") == "recall_info"

    def test_detect_no_match(self):
        assert detect_intent("Hello there!") is None
        assert detect_intent("Tell me a joke") is None

    def test_get_template(self):
        tmpl = get_action_template("check_email")
        assert tmpl is not None
        assert "email" in tmpl.required_tools[0]

    def test_confirmation_for_destructive_actions(self):
        draft = get_action_template("draft_email")
        assert draft.confirmation_required is True
        check = get_action_template("check_email")
        assert check.confirmation_required is False

    def test_keyword_coverage(self):
        keywords = get_intent_keywords()
        # Every template should have keywords
        for intent in ACTION_TEMPLATES:
            assert intent in keywords, f"No keywords for {intent}"
