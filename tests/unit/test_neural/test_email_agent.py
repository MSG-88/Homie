"""Unit tests for EmailAgent."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from homie_core.neural.agents.email_agent import EmailAgent
from homie_core.neural.communication.agent_bus import AgentBus, AgentMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bus():
    bus = AgentBus()
    yield bus
    bus.shutdown()


@pytest.fixture()
def bus():
    b = AgentBus()
    yield b
    b.shutdown()


def _inference_fn(response):
    """Return an inference_fn that always returns *response*."""
    def fn(prompt: str, **kw) -> str:
        return response
    return fn


def _json_inference(obj):
    """Return an inference_fn that returns a JSON-encoded object."""
    return _inference_fn(json.dumps(obj))


def _make_email_service(unread=None):
    svc = MagicMock()
    svc.get_unread.return_value = unread or {"high": [], "medium": [], "low": []}
    return svc


def _sample_email():
    return {
        "id": "msg-1",
        "sender": "alice@example.com",
        "subject": "Invoice #1234 Due Tomorrow",
        "snippet": "Please pay the invoice of $500 by March 15.",
        "body": "Dear User, please pay invoice #1234 for $500 by March 15. Regards, Alice.",
        "priority": "high",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassifyPriority:
    def test_returns_valid_priority(self, bus):
        agent = EmailAgent(bus, _json_inference({"priority": "critical"}))
        assert agent.classify_priority(_sample_email()) == "critical"

    def test_returns_normal_on_empty_email(self, bus):
        agent = EmailAgent(bus, _json_inference({"priority": "high"}))
        assert agent.classify_priority({}) == "normal"

    def test_returns_normal_on_invalid_llm_output(self, bus):
        agent = EmailAgent(bus, _inference_fn("not json at all"))
        assert agent.classify_priority(_sample_email()) == "normal"

    def test_returns_normal_on_inference_exception(self, bus):
        def failing_fn(prompt, **kw):
            raise RuntimeError("LLM down")
        agent = EmailAgent(bus, failing_fn)
        assert agent.classify_priority(_sample_email()) == "normal"


class TestExtractKnowledge:
    def test_extracts_structured_knowledge(self, bus):
        knowledge = {
            "facts": ["Invoice #1234 is due"],
            "action_items": ["Pay $500"],
            "deadlines": ["March 15"],
            "people": ["Alice"],
            "amounts": ["$500"],
            "relationships": [],
        }
        agent = EmailAgent(bus, _json_inference(knowledge))
        result = agent.extract_knowledge("Please pay invoice #1234 for $500 by March 15.")
        assert result["action_items"] == ["Pay $500"]
        assert result["deadlines"] == ["March 15"]

    def test_returns_empty_on_empty_content(self, bus):
        agent = EmailAgent(bus, _json_inference({}))
        result = agent.extract_knowledge("")
        assert result["facts"] == []
        assert result["action_items"] == []

    def test_fills_missing_keys(self, bus):
        agent = EmailAgent(bus, _json_inference({"facts": ["something"]}))
        result = agent.extract_knowledge("some content")
        assert "action_items" in result
        assert "deadlines" in result
        assert result["facts"] == ["something"]

    def test_returns_empty_on_llm_failure(self, bus):
        agent = EmailAgent(bus, _inference_fn("garbage output"))
        result = agent.extract_knowledge("some content")
        assert result == EmailAgent._empty_knowledge()


class TestGenerateDigest:
    def test_generates_digest_from_emails(self, bus):
        agent = EmailAgent(bus, _inference_fn("## Action Required\n- Pay invoice"))
        emails = [_sample_email(), {**_sample_email(), "id": "msg-2", "subject": "Meeting"}]
        digest = agent.generate_digest(emails)
        assert "Action Required" in digest

    def test_returns_message_on_empty_list(self, bus):
        agent = EmailAgent(bus, _inference_fn("should not be called"))
        assert agent.generate_digest([]) == "No emails to summarize."


class TestDraftReply:
    def test_drafts_reply(self, bus):
        agent = EmailAgent(bus, _inference_fn("Thank you for the invoice. I will pay by March 15."))
        reply = agent.draft_reply(_sample_email())
        assert "invoice" in reply.lower() or "March" in reply

    def test_drafts_reply_with_instructions(self, bus):
        agent = EmailAgent(bus, _inference_fn("I need a 2-day extension."))
        reply = agent.draft_reply(_sample_email(), instructions="Ask for a 2-day extension")
        assert "extension" in reply.lower()

    def test_returns_empty_on_empty_email(self, bus):
        agent = EmailAgent(bus, _inference_fn("should not be called"))
        assert agent.draft_reply({}) == ""


class TestProcessInbox:
    def test_no_email_service_degrades_gracefully(self, bus):
        agent = EmailAgent(bus, _json_inference({"priority": "normal"}), email_service=None)
        result = agent.process_inbox()
        assert result["status"] == "no_email_service"
        assert result["processed"] == 0

    def test_empty_inbox(self, bus):
        svc = _make_email_service({"high": [], "medium": [], "low": []})
        agent = EmailAgent(bus, _json_inference({"priority": "normal"}), email_service=svc)
        result = agent.process_inbox()
        assert result["status"] == "empty"

    def test_processes_unread_emails(self, bus):
        knowledge_resp = json.dumps({
            "facts": [], "action_items": ["Pay invoice"],
            "deadlines": ["March 15"], "people": [], "amounts": ["$500"],
            "relationships": [],
        })
        classify_resp = json.dumps({"priority": "high"})
        call_count = {"n": 0}

        def multi_inference(prompt, **kw):
            call_count["n"] += 1
            if "Classify" in prompt:
                return classify_resp
            return knowledge_resp

        svc = _make_email_service({
            "high": [_sample_email()],
            "medium": [],
            "low": [],
        })
        agent = EmailAgent(bus, multi_inference, email_service=svc)
        result = agent.process_inbox()
        assert result["status"] == "ok"
        assert result["processed"] == 1
        assert len(result["action_items"]) >= 1

    def test_handles_email_service_error(self, bus):
        svc = MagicMock()
        svc.get_unread.side_effect = RuntimeError("Connection failed")
        agent = EmailAgent(bus, _json_inference({}), email_service=svc)
        result = agent.process_inbox()
        assert result["status"] == "error"


class TestProcessMessage:
    def test_process_routes_classify(self, bus):
        agent = EmailAgent(bus, _json_inference({"priority": "high"}))
        msg = AgentMessage(
            from_agent="meta", to_agent="email",
            message_type="goal",
            content={"action": "classify_priority", "email": _sample_email()},
        )
        result = asyncio.get_event_loop().run_until_complete(agent.process(msg))
        assert result.content["priority"] == "high"

    def test_process_routes_unknown_action(self, bus):
        agent = EmailAgent(bus, _json_inference({}))
        msg = AgentMessage(
            from_agent="meta", to_agent="email",
            message_type="goal",
            content={"action": "unknown_action"},
        )
        result = asyncio.get_event_loop().run_until_complete(agent.process(msg))
        assert "error" in result.content
