"""Desktop Companion — wires tray + FastAPI + email into a single process.

Entry point: python -m homie_app.desktop
"""
from __future__ import annotations

import logging
import secrets
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8721


def _generate_session_token() -> str:
    """Generate a cryptographic session token for localhost auth."""
    return secrets.token_urlsafe(32)


class DesktopCompanion:
    """Main orchestrator for the desktop companion."""

    def __init__(self, config_path: Optional[str] = None):
        self._session_token = _generate_session_token()
        self._port = _DEFAULT_PORT
        self._config_path = config_path
        self._config = None
        self._email_service = None
        self._inference_router = None
        self._vault = None
        self._tray = None
        self._server_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start all components: config, vault, email, server, tray."""
        from homie_core.config import load_config
        from homie_core.vault.secure_vault import SecureVault

        self._config = load_config(self._config_path)
        self._running = True

        # Vault
        storage = Path(self._config.storage.path).expanduser()
        vault_dir = storage / "vault"
        self._vault = SecureVault(storage_dir=vault_dir)
        try:
            self._vault.unlock()
            print("  Vault: unlocked")
        except Exception as e:
            print(f"  Vault: {e}")

        # Email service
        self._init_email()

        # Inference
        self._init_inference()

        # Start FastAPI server
        self._start_server()

        # Start email sync loop
        if self._email_service:
            self._start_sync()

        # Start tray (blocks main thread)
        self._start_tray()

    def _init_email(self) -> None:
        """Initialize email service if Gmail credentials exist."""
        try:
            gmail_creds = self._vault.list_credentials("gmail")
            active = [c for c in gmail_creds if c.active and c.account_id != "oauth_client"]
            if not active:
                print("  Email: no Gmail accounts connected")
                return

            from homie_core.email import EmailService
            from homie_core.vault.schema import create_cache_db

            storage = Path(self._config.storage.path).expanduser()
            cache_conn = create_cache_db(storage / "cache.db")
            self._email_service = EmailService(
                vault=self._vault, cache_conn=cache_conn,
            )
            accounts = self._email_service.initialize()
            print(f"  Email: {len(accounts)} account(s) connected")
        except Exception as e:
            print(f"  Email: {e}")

    def _init_inference(self) -> None:
        """Initialize inference router for chat."""
        try:
            from homie_core.inference.router import InferenceRouter
            from homie_core.model.engine import ModelEngine

            engine = ModelEngine(config=self._config.llm)
            self._inference_router = InferenceRouter(
                config=self._config,
                model_engine=engine,
            )
            source = self._inference_router.active_source
            print(f"  Inference: {source}")
        except Exception as e:
            print(f"  Inference: not available ({e})")

    def _start_server(self) -> None:
        """Start FastAPI on a background thread."""
        from homie_app.tray.dashboard import create_dashboard_app

        app = create_dashboard_app(
            config=self._config,
            email_service=self._email_service,
            session_token=self._session_token,
            inference_router=self._inference_router,
        )

        def run_server():
            import uvicorn
            uvicorn.run(app, host="127.0.0.1", port=self._port, log_level="warning")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        print(f"  Dashboard: http://127.0.0.1:{self._port}/briefing")

    def _start_sync(self) -> None:
        """Run email sync every 5 minutes in background."""
        def sync_loop():
            while self._running:
                try:
                    result = self._email_service.sync_tick()
                    logger.info("Email sync: %s", result)
                    if self._tray:
                        summary = self._email_service.get_summary(days=1)
                        self._tray.update_unread_count(summary.get("unread", 0))
                except Exception as e:
                    logger.error("Sync error: %s", e)
                time.sleep(300)

        self._sync_thread = threading.Thread(target=sync_loop, daemon=True)
        self._sync_thread.start()
        print("  Sync: running every 5 minutes")

    def _start_tray(self) -> None:
        """Start system tray (blocks main thread)."""
        from homie_app.tray.app import TrayApp

        def open_briefing():
            url = f"http://127.0.0.1:{self._port}/briefing"
            webbrowser.open(url)

        def on_quit():
            self._running = False

        self._tray = TrayApp(
            on_open_briefing=open_briefing,
            on_open_dashboard=open_briefing,
            on_quit=on_quit,
        )

        if self._email_service:
            try:
                summary = self._email_service.get_summary(days=1)
                self._tray.update_unread_count(summary.get("unread", 0))
            except Exception:
                pass

        print("\n  Homie Desktop Companion is running.")
        print("  Right-click the tray icon for options.\n")
        self._tray.start()

        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self._tray.stop()


def main():
    """CLI entry point for the desktop companion."""
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    companion = DesktopCompanion(config_path=config_path)
    companion.start()


if __name__ == "__main__":
    main()
