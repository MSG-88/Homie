"""Tests for chat API endpoint."""
from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from homie_app.tray.dashboard import create_dashboard_app


@pytest.fixture
def mock_inference():
    router = MagicMock()
    router.generate.return_value = "I can help with that email."
    router.active_source = "Local"
    return router


@pytest.fixture
def mock_email():
    svc = MagicMock()
    svc.get_summary.return_value = {"total": 10, "unread": 3, "high_priority": [{"subject": "Deploy alert", "sender": "ci@ops.io"}]}
    svc.search.return_value = []
    svc.read_message.return_value = {"subject": "Test", "body": "Hello"}
    return svc


@pytest.fixture
def client(mock_inference, mock_email):
    app = create_dashboard_app(
        email_service=mock_email,
        inference_router=mock_inference,
        session_token="test-token",
    )
    return TestClient(app)


def test_chat_returns_response(client, mock_inference):
    resp = client.post("/api/chat", json={"message": "What emails need attention?"}, cookies={"homie_session": "test-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "I can help with that email."
    assert data["source"] == "Local"
    mock_inference.generate.assert_called_once()


def test_chat_requires_auth(client):
    resp = client.post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 401


def test_chat_empty_message(client):
    resp = client.post("/api/chat", json={"message": ""}, cookies={"homie_session": "test-token"})
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_chat_history(client):
    client.post("/api/chat", json={"message": "hello"}, cookies={"homie_session": "test-token"})
    resp = client.get("/api/chat/history", cookies={"homie_session": "test-token"})
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert len(messages) == 2  # user + assistant
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_clear_chat_history(client):
    client.post("/api/chat", json={"message": "hello"}, cookies={"homie_session": "test-token"})
    client.delete("/api/chat/history", cookies={"homie_session": "test-token"})
    resp = client.get("/api/chat/history", cookies={"homie_session": "test-token"})
    assert len(resp.json()["messages"]) == 0


def test_chat_no_inference(mock_email):
    app = create_dashboard_app(email_service=mock_email, session_token="tok")
    c = TestClient(app)
    resp = c.post("/api/chat", json={"message": "hi"}, cookies={"homie_session": "tok"})
    assert "No inference" in resp.json()["response"]


def test_chat_page_returns_html(client):
    resp = client.get("/chat", cookies={"homie_session": "test-token"})
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<html" in resp.text
    assert "Chat" in resp.text


def test_email_search(client):
    resp = client.post("/api/email/search", json={"query": "deploy"}, cookies={"homie_session": "test-token"})
    assert resp.status_code == 200
    assert "results" in resp.json()


def test_email_read(client):
    resp = client.get("/api/email/read/msg123", cookies={"homie_session": "test-token"})
    assert resp.status_code == 200
