import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from homie_core.messaging.telegram_provider import TelegramProvider


class TestTelegramProvider:
    def test_init(self):
        provider = TelegramProvider(api_id=12345, api_hash="abc123")
        assert provider.connected is False

    def test_requires_api_id_and_hash(self):
        with pytest.raises(ValueError):
            TelegramProvider(api_id=0, api_hash="")

    def test_session_path(self):
        import tempfile
        provider = TelegramProvider(api_id=12345, api_hash="abc123", session_dir=tempfile.gettempdir())
        assert tempfile.gettempdir() in str(provider._session_path)
