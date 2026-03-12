"""Tests for SlackProvider — _call helper, connect, list_channels, search, rate limiting."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from homie_core.social.slack_provider import SlackProvider


def _mock_credential(token="xoxb-test-token"):
    cred = MagicMock()
    cred.access_token = token
    return cred


class TestSlackProviderCall:
    """Tests for the _call helper method."""

    @patch("homie_core.social.slack_provider.requests")
    def test_call_posts_with_bearer_token(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": True}
        mock_requests.post.return_value = resp

        provider = SlackProvider(account_id="test-workspace")
        provider._token = "xoxb-test"

        result = provider._call("auth.test")

        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert call_args[0][0] == "https://slack.com/api/auth.test"
        assert call_args[1]["headers"]["Authorization"] == "Bearer xoxb-test"
        assert result == {"ok": True}

    @patch("homie_core.social.slack_provider.requests")
    def test_call_passes_params(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": True, "channels": []}
        mock_requests.post.return_value = resp

        provider = SlackProvider(account_id="test")
        provider._token = "xoxb-test"

        provider._call("conversations.list", types="public_channel", limit=100)

        call_args = mock_requests.post.call_args
        assert call_args[1]["data"]["types"] == "public_channel"
        assert call_args[1]["data"]["limit"] == 100

    @patch("homie_core.social.slack_provider.time.sleep")
    @patch("homie_core.social.slack_provider.requests")
    def test_call_retries_on_rate_limit(self, mock_requests, mock_sleep):
        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "2"}

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}

        mock_requests.post.side_effect = [rate_resp, ok_resp]

        provider = SlackProvider(account_id="test")
        provider._token = "xoxb-test"

        result = provider._call("auth.test")

        assert result == {"ok": True}
        mock_sleep.assert_called_once_with(2)
        assert mock_requests.post.call_count == 2

    def test_call_raises_without_token(self):
        provider = SlackProvider(account_id="test")
        with pytest.raises(RuntimeError, match="Not connected"):
            provider._call("auth.test")

    @patch("homie_core.social.slack_provider.requests", None)
    def test_call_raises_without_requests(self):
        provider = SlackProvider(account_id="test")
        provider._token = "xoxb-test"
        with pytest.raises(ImportError, match="requests"):
            provider._call("auth.test")


class TestSlackProviderConnect:
    @patch("homie_core.social.slack_provider.requests")
    def test_connect_success(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": True, "team": "MyTeam", "user_id": "U123"}
        mock_requests.post.return_value = resp

        provider = SlackProvider(account_id="workspace")
        result = provider.connect(_mock_credential())

        assert result is True
        assert provider._team_name == "MyTeam"
        assert provider._user_id == "U123"

    @patch("homie_core.social.slack_provider.requests")
    def test_connect_failure(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": False, "error": "invalid_auth"}
        mock_requests.post.return_value = resp

        provider = SlackProvider(account_id="workspace")
        result = provider.connect(_mock_credential())

        assert result is False


class TestSlackProviderListChannels:
    @patch("homie_core.social.slack_provider.requests")
    def test_list_channels(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "ok": True,
            "channels": [
                {"id": "C001", "name": "general", "is_im": False, "num_members": 50},
                {"id": "D001", "user": "U999", "is_im": True, "num_members": 0},
            ],
        }
        mock_requests.post.return_value = resp

        provider = SlackProvider(account_id="test")
        provider._token = "xoxb-test"

        channels = provider.list_channels()

        assert len(channels) == 2
        assert channels[0].id == "C001"
        assert channels[0].name == "general"
        assert channels[0].platform == "slack"
        assert channels[0].is_dm is False
        assert channels[1].is_dm is True


class TestSlackProviderSearch:
    @patch("homie_core.social.slack_provider.requests")
    def test_search_messages(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "ok": True,
            "messages": {
                "matches": [
                    {
                        "ts": "1234567890.123",
                        "text": "Hello world",
                        "user": "U123",
                        "channel": {"id": "C001", "is_im": False},
                    },
                ],
            },
        }
        mock_requests.post.return_value = resp

        provider = SlackProvider(account_id="test")
        provider._token = "xoxb-test"
        provider._user_id = "U999"

        results = provider.search_messages("hello", limit=5)

        assert len(results) == 1
        assert results[0].content == "Hello world"
        assert results[0].platform == "slack"
        assert results[0].is_mention is False

    @patch("homie_core.social.slack_provider.requests")
    def test_search_detects_mention(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "ok": True,
            "messages": {
                "matches": [
                    {
                        "ts": "1234567890.123",
                        "text": "Hey <@U999> check this out",
                        "user": "U123",
                        "channel": {"id": "C001", "is_im": False},
                    },
                ],
            },
        }
        mock_requests.post.return_value = resp

        provider = SlackProvider(account_id="test")
        provider._token = "xoxb-test"
        provider._user_id = "U999"

        results = provider.search_messages("check", limit=5)
        assert results[0].is_mention is True


class TestSlackProviderSendMessage:
    @patch("homie_core.social.slack_provider.requests")
    def test_send_message(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": True, "ts": "1234567890.999"}
        mock_requests.post.return_value = resp

        provider = SlackProvider(account_id="test")
        provider._token = "xoxb-test"

        ts = provider.send_message("C001", "Hello!")
        assert ts == "1234567890.999"

        call_data = mock_requests.post.call_args[1]["data"]
        assert call_data["channel"] == "C001"
        assert call_data["text"] == "Hello!"

    @patch("homie_core.social.slack_provider.requests")
    def test_send_message_in_thread(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": True, "ts": "1234567890.999"}
        mock_requests.post.return_value = resp

        provider = SlackProvider(account_id="test")
        provider._token = "xoxb-test"

        provider.send_message("C001", "Reply!", thread_id="1234567890.000")

        call_data = mock_requests.post.call_args[1]["data"]
        assert call_data["thread_ts"] == "1234567890.000"
