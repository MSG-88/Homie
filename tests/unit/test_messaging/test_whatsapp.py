from unittest.mock import patch
from homie_core.messaging.whatsapp_provider import WhatsAppProvider


class TestWhatsAppProvider:
    def test_init(self):
        provider = WhatsAppProvider()
        assert provider.connected is False
        assert provider.experimental is True

    @patch("homie_core.messaging.whatsapp_provider.shutil.which")
    def test_node_not_found(self, mock_which):
        mock_which.return_value = None
        provider = WhatsAppProvider()
        assert provider.is_node_available() is False

    @patch("homie_core.messaging.whatsapp_provider.shutil.which")
    def test_node_found(self, mock_which):
        mock_which.return_value = "/usr/bin/node"
        provider = WhatsAppProvider()
        assert provider.is_node_available() is True
