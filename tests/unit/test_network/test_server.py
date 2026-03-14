"""Tests for the WebSocket sync server."""
from homie_core.network.server import SyncServer


def test_server_init(tmp_path):
    server = SyncServer(device_id="desktop-1", device_name="My PC", port=8765, data_dir=tmp_path)
    assert server.device_id == "desktop-1"
    assert server.port == 8765
    assert server.paired_devices == {}


def test_server_generate_pairing_code(tmp_path):
    server = SyncServer(device_id="desktop-1", device_name="My PC", port=8765, data_dir=tmp_path)
    code = server.generate_pairing_code()
    assert len(code) == 6
    assert code.isdigit()


def test_server_verify_pairing_code(tmp_path):
    server = SyncServer(device_id="desktop-1", device_name="My PC", port=8765, data_dir=tmp_path)
    code = server.generate_pairing_code()
    assert server.verify_pairing_code(code) is True
    assert server.verify_pairing_code("000000") is False


def test_server_add_paired_device(tmp_path):
    server = SyncServer(device_id="desktop-1", device_name="My PC", port=8765, data_dir=tmp_path)
    server.add_paired_device("phone-1", "My Phone", "fake-key")
    assert "phone-1" in server.paired_devices
    assert server.paired_devices["phone-1"]["name"] == "My Phone"


def test_server_remove_paired_device(tmp_path):
    server = SyncServer(device_id="desktop-1", device_name="My PC", port=8765, data_dir=tmp_path)
    server.add_paired_device("phone-1", "My Phone", "fake-key")
    server.remove_paired_device("phone-1")
    assert "phone-1" not in server.paired_devices


def test_server_paired_devices_persist(tmp_path):
    server1 = SyncServer(device_id="desktop-1", device_name="My PC", port=8765, data_dir=tmp_path)
    server1.add_paired_device("phone-1", "My Phone", "fake-key")
    # Create new server instance with same data_dir — should load persisted devices
    server2 = SyncServer(device_id="desktop-1", device_name="My PC", port=8765, data_dir=tmp_path)
    assert "phone-1" in server2.paired_devices
