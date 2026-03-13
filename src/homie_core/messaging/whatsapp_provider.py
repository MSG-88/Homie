from __future__ import annotations
import logging
import shutil

logger = logging.getLogger(__name__)


class WhatsAppProvider:
    experimental = True

    def __init__(self, session_dir: str = "~/.homie/whatsapp"):
        self._session_dir = session_dir
        self.connected = False

    def is_node_available(self) -> bool:
        return shutil.which("node") is not None

    def connect(self) -> None:
        if not self.is_node_available():
            raise RuntimeError("WhatsApp requires Node.js 18+. Install from https://nodejs.org")
        logger.info("WhatsApp bridge not yet fully implemented")

    def disconnect(self) -> None:
        self.connected = False
