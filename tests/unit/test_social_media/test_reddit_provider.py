"""Tests for the Reddit social-media provider."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from homie_core.social_media.reddit_provider import RedditProvider


def _make_credential(token: str = "tok123") -> MagicMock:
    cred = MagicMock()
    cred.access_token = token
    return cred


def _me_response() -> dict:
    return {
        "name": "test_user",
        "total_karma": 4200,
        "created_utc": 1600000000.0,
        "icon_img": "https://example.com/avatar.png",
        "verified": True,
        "subreddit": {
            "title": "Test User",
            "public_description": "Hello, I am a test user.",
        },
    }


def _listing(children: list[dict]) -> dict:
    return {"data": {"children": [{"data": c} for c in children]}}


SAMPLE_POST = {
    "name": "t3_abc123",
    "author": "poster",
    "title": "Cool post",
    "selftext": "Some body text",
    "created_utc": 1700000000.0,
    "permalink": "/r/test/comments/abc123/cool_post/",
    "ups": 42,
    "num_comments": 7,
    "is_self": True,
}


# ------------------------------------------------------------------
# Connection
# ------------------------------------------------------------------


@patch("homie_core.social_media.reddit_provider.requests")
class TestConnect:
    def test_connect_success(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _me_response()
        mock_requests.request.return_value = resp

        provider = RedditProvider()
        assert provider.connect(_make_credential()) is True
        assert provider.is_connected is True
        assert provider._username == "test_user"

    def test_connect_failure(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 401
        resp.raise_for_status.side_effect = Exception("Unauthorized")
        resp.json.side_effect = Exception("Unauthorized")
        mock_requests.request.return_value = resp

        provider = RedditProvider()
        assert provider.connect(_make_credential()) is False
        assert provider.is_connected is False


# ------------------------------------------------------------------
# Feed
# ------------------------------------------------------------------


@patch("homie_core.social_media.reddit_provider.requests")
class TestFeed:
    def test_get_feed(self, mock_requests):
        # Connect response
        connect_resp = MagicMock(status_code=200)
        connect_resp.json.return_value = _me_response()

        # Feed response
        feed_resp = MagicMock(status_code=200)
        feed_resp.json.return_value = _listing([SAMPLE_POST])

        mock_requests.request.side_effect = [connect_resp, feed_resp]

        provider = RedditProvider()
        provider.connect(_make_credential())
        posts = provider.get_feed(limit=5)

        assert len(posts) == 1
        post = posts[0]
        assert post.id == "t3_abc123"
        assert post.author == "poster"
        assert post.content == "Some body text"
        assert post.likes == 42
        assert post.comments == 7
        assert post.platform == "reddit"


# ------------------------------------------------------------------
# Profile
# ------------------------------------------------------------------


@patch("homie_core.social_media.reddit_provider.requests")
class TestProfile:
    def test_get_profile(self, mock_requests):
        me = _me_response()
        resp = MagicMock(status_code=200)
        resp.json.return_value = me
        mock_requests.request.return_value = resp

        provider = RedditProvider()
        provider.connect(_make_credential())
        profile = provider.get_profile()

        assert profile.username == "test_user"
        assert profile.display_name == "Test User"
        assert profile.bio == "Hello, I am a test user."
        assert profile.platform == "reddit"
        assert profile.verified is True


# ------------------------------------------------------------------
# Publish
# ------------------------------------------------------------------


@patch("homie_core.social_media.reddit_provider.requests")
class TestPublish:
    def test_publish(self, mock_requests):
        connect_resp = MagicMock(status_code=200)
        connect_resp.json.return_value = _me_response()

        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"success": True}

        mock_requests.request.side_effect = [connect_resp, submit_resp]

        provider = RedditProvider()
        provider.connect(_make_credential())
        result = provider.publish("Hello Reddit!")

        assert result == {"success": True}
        # Verify the submit call
        submit_call = mock_requests.request.call_args_list[1]
        assert submit_call[0] == ("POST", "https://oauth.reddit.com/api/submit")
        assert submit_call[1]["json"]["kind"] == "self"
        assert submit_call[1]["json"]["sr"] == "u_test_user"
        assert submit_call[1]["json"]["text"] == "Hello Reddit!"


# ------------------------------------------------------------------
# Direct Messages
# ------------------------------------------------------------------


@patch("homie_core.social_media.reddit_provider.requests")
class TestSendMessage:
    def test_send_message(self, mock_requests):
        connect_resp = MagicMock(status_code=200)
        connect_resp.json.return_value = _me_response()

        compose_resp = MagicMock(status_code=200)
        compose_resp.json.return_value = {"success": True}

        mock_requests.request.side_effect = [connect_resp, compose_resp]

        provider = RedditProvider()
        provider.connect(_make_credential())
        result = provider.send_message("other_user", "Hey!")

        assert result == {"success": True}
        compose_call = mock_requests.request.call_args_list[1]
        assert compose_call[0] == ("POST", "https://oauth.reddit.com/api/compose")
        assert compose_call[1]["json"]["to"] == "other_user"
        assert compose_call[1]["json"]["text"] == "Hey!"


# ------------------------------------------------------------------
# Rate-limit retry
# ------------------------------------------------------------------


@patch("homie_core.social_media.reddit_provider.time")
@patch("homie_core.social_media.reddit_provider.requests")
class TestRateLimitRetry:
    def test_rate_limit_retry(self, mock_requests, mock_time):
        provider = RedditProvider()
        provider._token = "tok123"
        provider._connected = True

        rate_resp = MagicMock(status_code=429)
        rate_resp.headers = {"Retry-After": "1"}

        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = _me_response()

        mock_requests.request.side_effect = [rate_resp, ok_resp]

        result = provider._call("GET", "/api/v1/me")

        assert result["name"] == "test_user"
        assert mock_requests.request.call_count == 2
        mock_time.sleep.assert_called_once_with(1.0)
