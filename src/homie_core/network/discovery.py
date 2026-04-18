"""mDNS/DNS-SD service advertisement and discovery for Homie LAN sync."""
from __future__ import annotations

import logging
import socket
from typing import Optional

logger = logging.getLogger(__name__)
SERVICE_TYPE = "_homie._tcp.local."

try:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceInfo, ServiceStateChange
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False
    Zeroconf = None
    ServiceBrowser = None
    ServiceInfo = None
    ServiceStateChange = None


class HomieDiscovery:
    """Advertise and discover Homie instances on the LAN via mDNS."""

    def __init__(self, device_id: str, device_name: str, port: int = 8765,
                 role: str = "standalone", mesh_id: str = "", capability_score: float = 0.0):
        self.device_id = device_id
        self.device_name = device_name
        self.port = port
        self.role = role
        self.mesh_id = mesh_id
        self.capability_score = capability_score
        self._zeroconf: Optional[Zeroconf] = None
        self._browser = None
        self._service_info = None
        self._advertising = False
        self._discovered: dict[str, dict] = {}

    @property
    def is_advertising(self) -> bool:
        return self._advertising

    @property
    def discovered_devices(self) -> dict[str, dict]:
        return dict(self._discovered)

    def start_advertising(self) -> None:
        import homie_core.network.discovery as _mod
        _Zeroconf = _mod.Zeroconf
        if _Zeroconf is None:
            logger.warning("zeroconf not installed — LAN discovery disabled")
            return
        # Import ServiceInfo from the same zeroconf package as the (possibly mocked) Zeroconf
        try:
            import zeroconf as _zc_pkg
            _ServiceInfo = _zc_pkg.ServiceInfo
        except ImportError:
            # zeroconf not installed; Zeroconf must have been injected via mock
            # Build a minimal ServiceInfo stand-in from the mock's module attrs
            _ServiceInfo = getattr(_mod, "ServiceInfo", None)
        self._zeroconf = _Zeroconf()
        hostname = socket.gethostname()
        if _ServiceInfo is not None:
            self._service_info = _ServiceInfo(
                SERVICE_TYPE,
                f"homie-{self.device_id}.{SERVICE_TYPE}",
                addresses=[socket.inet_aton(self._get_local_ip())],
                port=self.port,
                properties={
                    "device_id": self.device_id,
                    "device_name": self.device_name,
                    "version": "1.0.0",
                    "role": self.role,
                    "mesh_id": self.mesh_id,
                    "capability_score": str(self.capability_score),
                },
                server=f"{hostname}.local.",
            )
        self._zeroconf.register_service(self._service_info)
        self._advertising = True
        logger.info("Advertising Homie on LAN: %s:%d", hostname, self.port)

    def stop_advertising(self) -> None:
        if self._zeroconf:
            self._zeroconf.unregister_service(self._service_info)
            self._zeroconf.close()
        self._advertising = False
        self._zeroconf = None
        self._service_info = None

    def start_browsing(self) -> None:
        import homie_core.network.discovery as _mod
        _Zeroconf = _mod.Zeroconf
        _ServiceBrowser = _mod.ServiceBrowser
        if _Zeroconf is None:
            return
        if not self._zeroconf:
            self._zeroconf = _Zeroconf()
        if _ServiceBrowser is not None:
            self._browser = _ServiceBrowser(
                self._zeroconf, SERVICE_TYPE, handlers=[self._on_service_state_change]
            )

    def stop_browsing(self) -> None:
        if self._browser:
            self._browser.cancel()
            self._browser = None

    def _on_service_state_change(self, zeroconf, service_type, name, state_change) -> None:
        import homie_core.network.discovery as _mod
        _ServiceStateChange = _mod.ServiceStateChange
        if _ServiceStateChange and state_change == _ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                device_id = info.properties.get(b"device_id", b"").decode()
                if device_id and device_id != self.device_id:
                    self._discovered[device_id] = {
                        "name": info.properties.get(b"device_name", b"").decode(),
                        "host": socket.inet_ntoa(info.addresses[0]) if info.addresses else "",
                        "port": info.port,
                        "role": info.properties.get(b"role", b"standalone").decode(),
                        "mesh_id": info.properties.get(b"mesh_id", b"").decode(),
                        "capability_score": float(info.properties.get(b"capability_score", b"0").decode()),
                    }
        elif _ServiceStateChange and state_change == _ServiceStateChange.Removed:
            for did in list(self._discovered.keys()):
                if did in name:
                    del self._discovered[did]

    @staticmethod
    def _get_local_ip() -> str:
        """Get local IP without requiring internet access."""
        try:
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except Exception:
            return "127.0.0.1"
