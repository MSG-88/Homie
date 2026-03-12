"""Tests for SocialService facade — initialize, sync_tick, search, provider iteration."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from homie_core.social import SocialService
from homie_core.social.models import SocialChannel, SocialMessage


def _make_msg(**overrides) -> SocialMessage:
    defaults = dict(
        id="msg1", platform="slack", channel="C001", sender="U123",
        content="Hello!", timestamp=1700000000.0,
    )
    defaults.update(overrides)
    return SocialMessage(**defaults)


def _make_channel(**overrides) -> SocialChannel:
    defaults = dict(id="C001", name="general", platform="slack")
    defaults.update(overrides)
    return SocialChannel(**defaults)


class TestSocialServiceInitialize:
    @patch("homie_core.social.slack_provider.SlackProvider")
    def test_initialize_connects_slack(self, MockSlackProvider):
        vault = MagicMock()
        cred = MagicMock()
        cred.active = True
        cred.account_id = "my-workspace"
        vault.list_credentials.return_value = [cred]

        mock_provider = MagicMock()
        mock_provider.connect.return_value = True
        MockSlackProvider.return_value = mock_provider

        service = SocialService(vault=vault)
        connected = service.initialize()

        assert "slack" in connected
        vault.set_connection_status.assert_called_once_with(
            "slack", connected=True, label="my-workspace",
        )

    @patch("homie_core.social.slack_provider.SlackProvider")
    def test_initialize_skips_inactive(self, MockSlackProvider):
        vault = MagicMock()
        cred = MagicMock()
        cred.active = False
        cred.account_id = "my-workspace"
        vault.list_credentials.return_value = [cred]

        service = SocialService(vault=vault)
        connected = service.initialize()

        assert connected == []
        MockSlackProvider.assert_not_called()

    @patch("homie_core.social.slack_provider.SlackProvider")
    def test_initialize_skips_oauth_client(self, MockSlackProvider):
        vault = MagicMock()
        cred = MagicMock()
        cred.active = True
        cred.account_id = "oauth_client"
        vault.list_credentials.return_value = [cred]

        service = SocialService(vault=vault)
        connected = service.initialize()

        assert connected == []


class TestSocialServiceSyncTick:
    def test_sync_tick_pushes_to_working_memory(self):
        vault = MagicMock()
        working_memory = MagicMock()
        provider = MagicMock()
        provider.get_unread_mentions.return_value = [
            _make_msg(is_mention=True, content="Hey check this!"),
        ]

        service = SocialService(vault=vault, working_memory=working_memory)
        service._providers["slack"] = provider

        result = service.sync_tick()

        assert "slack: 1 mention(s)" in result
        working_memory.push.assert_called_once()

    def test_sync_tick_no_providers(self):
        vault = MagicMock()
        service = SocialService(vault=vault)

        result = service.sync_tick()
        assert result == "No social platforms connected"

    def test_sync_tick_handles_error(self):
        vault = MagicMock()
        provider = MagicMock()
        provider.get_unread_mentions.side_effect = RuntimeError("API down")

        service = SocialService(vault=vault)
        service._providers["slack"] = provider

        result = service.sync_tick()
        assert "error" in result


class TestSocialServiceSearch:
    def test_search_across_platforms(self):
        vault = MagicMock()
        provider = MagicMock()
        provider.search_messages.return_value = [_make_msg()]

        service = SocialService(vault=vault)
        service._providers["slack"] = provider

        results = service.search("hello")
        assert len(results) == 1
        assert results[0]["content"] == "Hello!"

    def test_search_filters_platform(self):
        vault = MagicMock()
        slack = MagicMock()
        slack.search_messages.return_value = [_make_msg()]

        service = SocialService(vault=vault)
        service._providers["slack"] = slack

        # Searching for "discord" should skip slack
        results = service.search("hello", platform="discord")
        assert results == []
        slack.search_messages.assert_not_called()


class TestSocialServiceListChannels:
    def test_list_channels(self):
        vault = MagicMock()
        provider = MagicMock()
        provider.list_channels.return_value = [_make_channel()]

        service = SocialService(vault=vault)
        service._providers["slack"] = provider

        channels = service.list_channels()
        assert len(channels) == 1
        assert channels[0]["name"] == "general"


class TestSocialServiceGetMessages:
    def test_get_messages(self):
        vault = MagicMock()
        provider = MagicMock()
        provider.get_recent_messages.return_value = [_make_msg()]

        service = SocialService(vault=vault)
        service._providers["slack"] = provider

        messages = service.get_messages("C001")
        assert len(messages) == 1
        assert messages[0]["sender"] == "U123"

    def test_get_messages_no_matching_provider(self):
        vault = MagicMock()
        service = SocialService(vault=vault)

        messages = service.get_messages("C001")
        assert messages == []


class TestSocialServiceSendMessage:
    def test_send_message(self):
        vault = MagicMock()
        provider = MagicMock()
        provider.send_message.return_value = "1234567890.999"

        service = SocialService(vault=vault)
        service._providers["slack"] = provider

        result = service.send_message("C001", "Hello!")
        assert result["status"] == "sent"
        assert result["message_id"] == "1234567890.999"

    def test_send_message_no_provider(self):
        vault = MagicMock()
        service = SocialService(vault=vault)

        result = service.send_message("C001", "Hello!")
        assert result["status"] == "error"


class TestSocialServiceGetUnread:
    def test_get_unread(self):
        vault = MagicMock()
        provider = MagicMock()
        provider.get_unread_mentions.return_value = [_make_msg(is_mention=True)]

        service = SocialService(vault=vault)
        service._providers["slack"] = provider

        unread = service.get_unread()
        assert "slack" in unread
        assert len(unread["slack"]) == 1
