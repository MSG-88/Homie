"""Tests for proactive email intelligence."""
import pytest
from unittest.mock import MagicMock, patch
from homie_core.email.proactive_intelligence import (
    ProactiveEmailIntelligence, EmailInsight, DailyBriefing,
)


class TestEmailInsight:
    def test_basic_creation(self):
        insight = EmailInsight(
            category="action_required", priority="high",
            subject="Review PR", detail="Please review the PR",
            source_email_id="123", source_subject="PR Review",
            source_sender="Alice", suggested_action="Review the PR",
        )
        assert insight.category == "action_required"
        assert insight.priority == "high"


class TestDailyBriefing:
    def test_empty_briefing(self):
        b = DailyBriefing(timestamp=0, total_unread=0, high_priority_count=0)
        assert b.to_prompt_context() == ""

    def test_briefing_with_insights(self):
        b = DailyBriefing(
            timestamp=0, total_unread=5, high_priority_count=2,
            insights=[
                EmailInsight("action_required", "urgent", "Fix server",
                            "Server is down", "1", "Server Down", "DevOps",
                            "Check server status"),
            ],
            summary="5 unread, 1 urgent",
            suggested_focus="Handle urgent: Fix server",
        )
        ctx = b.to_prompt_context()
        assert "URGENT" in ctx
        assert "Fix server" in ctx
        assert "5 unread" in ctx


class TestProactiveEmailIntelligence:
    def _mock_email_service(self, unread_emails=None):
        svc = MagicMock()
        cache = MagicMock()
        if unread_emails:
            cache.execute.return_value.fetchall.return_value = [
                (e["id"], e["subject"], e["sender"], e["snippet"],
                 e["priority"], e.get("categories", ""), e.get("date", 0), e.get("labels", ""))
                for e in unread_emails
            ]
        else:
            cache.execute.return_value.fetchall.return_value = []
        svc._cache = cache
        return svc

    def test_empty_inbox(self):
        svc = self._mock_email_service()
        intel = ProactiveEmailIntelligence(svc)
        briefing = intel.generate_briefing()
        assert briefing.total_unread == 0
        assert briefing.insights == []

    def test_detects_action_required(self):
        svc = self._mock_email_service([{
            "id": "1", "subject": "Please review the document",
            "sender": "Alice <alice@co.com>", "snippet": "Can you please review this?",
            "priority": "high",
        }])
        intel = ProactiveEmailIntelligence(svc)
        briefing = intel.generate_briefing(force=True)
        assert briefing.total_unread == 1
        assert any(i.category == "action_required" for i in briefing.insights)

    def test_detects_deadline(self):
        svc = self._mock_email_service([{
            "id": "2", "subject": "Report due by Friday",
            "sender": "Boss <boss@co.com>", "snippet": "The deadline for the Q4 report is Friday",
            "priority": "high",
        }])
        intel = ProactiveEmailIntelligence(svc)
        briefing = intel.generate_briefing(force=True)
        assert any(i.category == "deadline" for i in briefing.insights)

    def test_detects_urgent(self):
        svc = self._mock_email_service([{
            "id": "3", "subject": "ASAP: Server outage",
            "sender": "Ops <ops@co.com>", "snippet": "We need you to fix this ASAP",
            "priority": "high",
        }])
        intel = ProactiveEmailIntelligence(svc)
        briefing = intel.generate_briefing(force=True)
        urgent = [i for i in briefing.insights if i.priority == "urgent"]
        assert len(urgent) > 0

    def test_detects_financial(self):
        svc = self._mock_email_service([{
            "id": "4", "subject": "Your electricity bill",
            "sender": "Power Co <bill@power.com>", "snippet": "Your bill for March",
            "priority": "medium", "categories": "bill",
        }])
        intel = ProactiveEmailIntelligence(svc)
        briefing = intel.generate_briefing(force=True)
        assert any(i.category == "financial" for i in briefing.insights)

    def test_summary_generation(self):
        svc = self._mock_email_service([
            {"id": "1", "subject": "Please confirm", "sender": "A <a@b.com>",
             "snippet": "Please confirm the order", "priority": "high"},
            {"id": "2", "subject": "Normal email", "sender": "B <b@b.com>",
             "snippet": "Hey there", "priority": "low"},
        ])
        intel = ProactiveEmailIntelligence(svc)
        briefing = intel.generate_briefing(force=True)
        assert briefing.summary
        assert "unread" in briefing.summary

    def test_prompt_context_format(self):
        svc = self._mock_email_service([{
            "id": "1", "subject": "Urgent task",
            "sender": "Boss <boss@co.com>", "snippet": "Please respond ASAP",
            "priority": "high",
        }])
        intel = ProactiveEmailIntelligence(svc)
        ctx = intel.get_context_for_prompt()
        assert isinstance(ctx, str)
        assert "Email Briefing" in ctx

    def test_caching(self):
        svc = self._mock_email_service([{
            "id": "1", "subject": "Test", "sender": "a@b.com",
            "snippet": "test", "priority": "low",
        }])
        intel = ProactiveEmailIntelligence(svc)
        b1 = intel.generate_briefing(force=True)
        b2 = intel.generate_briefing()  # should return cached
        assert b1 is b2

    def test_llm_analysis(self):
        svc = self._mock_email_service([{
            "id": "5", "subject": "Important matter",
            "sender": "CEO <ceo@co.com>", "snippet": "We need to discuss strategy",
            "priority": "high",
        }])
        mock_inference = MagicMock(return_value='{"needs_action": true, "category": "action_required", "priority": "high", "summary": "CEO wants strategy discussion", "suggested_action": "Schedule meeting with CEO"}')
        intel = ProactiveEmailIntelligence(svc, inference_fn=mock_inference)
        briefing = intel.generate_briefing(force=True)
        assert any("CEO" in i.subject or "strategy" in i.subject for i in briefing.insights)
