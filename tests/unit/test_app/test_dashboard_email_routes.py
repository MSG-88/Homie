"""Tests for email-related dashboard API routes."""
from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import MagicMock

import pytest

from homie_app.tray.dashboard import create_dashboard_app


@pytest.fixture
def mock_email_service():
    svc = MagicMock()
    svc.get_summary.return_value = {
        "total": 42,
        "unread": 5,
        "high_priority": [
            {"subject": "Contract Review", "sender": "alice@acme.com"},
            {"subject": "Urgent: Deploy blocked", "sender": "bob@ops.io"},
        ],
    }
    svc.get_unread.return_value = {
        "high": [{"id": "m1", "subject": "Contract Review", "sender": "alice@acme.com", "snippet": "Please review..."}],
        "medium": [{"id": "m2", "subject": "Weekly sync", "sender": "carol@team.io", "snippet": "Agenda for..."}],
        "low": [],
    }
    svc.triage.return_value = {
        "status": "Triaged 5 emails",
        "action_needed": [{"id": "m1", "subject": "Contract Review", "action_needed": True}],
        "important": [{"id": "m2", "subject": "Weekly sync"}],
        "likely_spam": [],
        "all": [],
    }
    svc.get_intelligent_digest.return_value = "You have 5 unread emails. 2 need attention."
    return svc


@pytest.fixture
def client(mock_email_service):
    from fastapi.testclient import TestClient
    app = create_dashboard_app(
        email_service=mock_email_service,
        session_token="test-token-123",
    )
    return TestClient(app)


def test_email_summary_requires_auth(client):
    resp = client.get("/api/email/summary")
    assert resp.status_code == 401


def test_email_summary_with_valid_token(client):
    resp = client.get("/api/email/summary", cookies={"homie_session": "test-token-123"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["unread"] == 5
    assert len(data["high_priority"]) == 2


def test_email_unread(client):
    resp = client.get("/api/email/unread", cookies={"homie_session": "test-token-123"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["high"]) == 1
    assert data["high"][0]["subject"] == "Contract Review"


def test_email_triage(client):
    resp = client.post("/api/email/triage", cookies={"homie_session": "test-token-123"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Triaged 5 emails"
    assert len(data["action_needed"]) == 1


def test_email_digest(client):
    resp = client.get("/api/email/digest", cookies={"homie_session": "test-token-123"})
    assert resp.status_code == 200
    data = resp.json()
    assert "digest" in data


def test_briefing_page_returns_html(client):
    resp = client.get("/briefing", cookies={"homie_session": "test-token-123"})
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<html" in resp.text


def test_mark_read_calls_service(client, mock_email_service):
    resp = client.post("/api/email/mark-read/msg123", cookies={"homie_session": "test-token-123"})
    assert resp.status_code == 200
    mock_email_service.mark_read.assert_called_once_with("msg123")


def test_health_no_auth_required(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
