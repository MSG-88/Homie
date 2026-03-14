"""Tests for mDNS discovery service."""
from unittest.mock import patch, MagicMock
from homie_core.network.discovery import HomieDiscovery


def test_discovery_init():
    discovery = HomieDiscovery(device_id="desktop-1", device_name="My PC", port=8765)
    assert discovery.device_id == "desktop-1"
    assert discovery.port == 8765
    assert discovery.is_advertising is False


def test_discovery_start_advertising():
    with patch("homie_core.network.discovery.Zeroconf") as MockZC:
        mock_zc = MagicMock()
        MockZC.return_value = mock_zc
        discovery = HomieDiscovery(device_id="desktop-1", device_name="My PC", port=8765)
        discovery.start_advertising()
        assert discovery.is_advertising is True
        mock_zc.register_service.assert_called_once()


def test_discovery_stop_advertising():
    with patch("homie_core.network.discovery.Zeroconf") as MockZC:
        mock_zc = MagicMock()
        MockZC.return_value = mock_zc
        discovery = HomieDiscovery(device_id="desktop-1", device_name="My PC", port=8765)
        discovery.start_advertising()
        discovery.stop_advertising()
        assert discovery.is_advertising is False
        mock_zc.unregister_service.assert_called_once()
        mock_zc.close.assert_called_once()


def test_discovery_discovered_devices_initially_empty():
    discovery = HomieDiscovery(device_id="desktop-1", device_name="My PC", port=8765)
    assert discovery.discovered_devices == {}
