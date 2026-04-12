from unittest.mock import MagicMock, patch
from homie_core.network.discovery import HomieDiscovery

def test_discovery_accepts_mesh_params():
    d = HomieDiscovery(
        device_id="abc", device_name="test", port=8765,
        role="hub", mesh_id="mesh-1", capability_score=290.0,
    )
    assert d.role == "hub"
    assert d.mesh_id == "mesh-1"
    assert d.capability_score == 290.0

def test_discovery_default_mesh_params():
    d = HomieDiscovery(device_id="abc", device_name="test")
    assert d.role == "standalone"
    assert d.mesh_id == ""
    assert d.capability_score == 0.0

def test_discovered_device_includes_mesh_fields():
    d = HomieDiscovery(device_id="local", device_name="me")
    mock_info = MagicMock()
    mock_info.properties = {
        b"device_id": b"remote-1", b"device_name": b"other-box",
        b"role": b"hub", b"mesh_id": b"mesh-1",
        b"capability_score": b"290.0", b"version": b"1.0.0",
    }
    mock_info.addresses = [b"\xc0\xa8\x01\x0a"]
    mock_info.port = 8765
    mock_state_change = MagicMock()
    mock_state_change_cls = MagicMock()
    mock_state_change_cls.Added = mock_state_change
    mock_zeroconf = MagicMock()
    mock_zeroconf.get_service_info.return_value = mock_info
    with patch("homie_core.network.discovery.ServiceStateChange", mock_state_change_cls):
        d._on_service_state_change(mock_zeroconf, "_homie._tcp.local.", "test", mock_state_change)
    devices = d.discovered_devices
    assert "remote-1" in devices
    assert devices["remote-1"]["role"] == "hub"
    assert devices["remote-1"]["mesh_id"] == "mesh-1"
    assert devices["remote-1"]["capability_score"] == 290.0
