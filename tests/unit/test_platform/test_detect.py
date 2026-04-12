import sys
from homie_core.platform.detect import get_platform_adapter
from homie_core.platform.base import PlatformAdapter


def test_get_platform_adapter_returns_adapter():
    adapter = get_platform_adapter()
    assert isinstance(adapter, PlatformAdapter)


def test_adapter_get_hostname():
    adapter = get_platform_adapter()
    assert isinstance(adapter.get_hostname(), str)
    assert len(adapter.get_hostname()) > 0


def test_adapter_get_system_metrics():
    adapter = get_platform_adapter()
    metrics = adapter.get_system_metrics()
    assert metrics["cpu_percent"] >= 0
    assert metrics["ram_total_gb"] > 0
    assert metrics["disk_free_gb"] > 0


def test_adapter_send_notification_does_not_crash():
    adapter = get_platform_adapter()
    adapter.send_notification("Test", "This is a test")
