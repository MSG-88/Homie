"""OAuth 2.0 flow for Gmail — local redirect server + manual fallback.

Handles:
1. Building the authorization URL
2. Running a local HTTP server to receive the redirect (port 8547)
3. Manual code entry fallback for headless environments
4. Code-to-token exchange
5. Token refresh
"""
from __future__ import annotations

import html
import http.server
import json
import queue
import threading
import time
import urllib.parse
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
]

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_REDIRECT_PORT = 8547
_REDIRECT_URI = f"http://localhost:{_REDIRECT_PORT}/callback"
_ALT_REDIRECT_PORT = 8548
_ALT_REDIRECT_URI = f"http://localhost:{_ALT_REDIRECT_PORT}/callback"


def build_auth_url(
    client_id: str,
    redirect_uri: str = _REDIRECT_URI,
) -> str:
    """Build the Google OAuth consent screen URL."""
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(GMAIL_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{_GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str = _REDIRECT_URI,
) -> dict[str, Any]:
    """Exchange authorization code for access + refresh tokens."""
    if requests is None:
        raise ImportError("requests library required for OAuth")

    resp = requests.post(_GOOGLE_TOKEN_URL, data={
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _make_callback_handler(result_queue: queue.Queue):
    """Create a callback handler that writes the auth code to a queue."""

    class _CallbackHandler(http.server.BaseHTTPRequestHandler):
        """HTTP handler to capture OAuth redirect callback."""

        def do_GET(self):
            # Ignore non-callback requests (e.g., favicon.ico)
            if not self.path.startswith("/callback"):
                self.send_response(404)
                self.end_headers()
                return

            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            if "code" in params:
                result_queue.put(params["code"][0])
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h2>Authorization successful!</h2>"
                                 b"<p>You can close this tab and return to Homie.</p></body></html>")
            else:
                error = html.escape(params.get("error", ["unknown"])[0])
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"<html><body><h2>Error: {error}</h2></body></html>".encode())

        def log_message(self, format, *args):
            pass  # Suppress server logs

    return _CallbackHandler


class GmailOAuth:
    """Handles Gmail OAuth 2.0 lifecycle."""

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = client_id
        self._client_secret = client_secret

    def get_auth_url(self, use_local_server: bool = True, alt_port: bool = False) -> str:
        """Get the authorization URL."""
        if alt_port:
            redirect = _ALT_REDIRECT_URI
        else:
            redirect = _REDIRECT_URI
        return build_auth_url(self._client_id, redirect)

    def wait_for_redirect(self, timeout: int = 120) -> str | None:
        """Start local server and wait for OAuth redirect. Returns auth code or None."""
        result_queue: queue.Queue[str] = queue.Queue()
        handler_cls = _make_callback_handler(result_queue)
        try:
            server = http.server.HTTPServer(("localhost", _REDIRECT_PORT), handler_cls)
        except OSError:
            return None  # Port unavailable

        server.timeout = timeout

        def _serve():
            while result_queue.empty():
                server.handle_request()

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        thread.join(timeout=timeout + 5)
        server.server_close()
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return None

    def wait_for_redirect_alt(self, timeout: int = 120) -> str | None:
        """Try alternate port for OAuth redirect. Returns auth code or None."""
        result_queue: queue.Queue[str] = queue.Queue()
        handler_cls = _make_callback_handler(result_queue)
        try:
            server = http.server.HTTPServer(("localhost", _ALT_REDIRECT_PORT), handler_cls)
        except OSError:
            return None
        server.timeout = timeout

        def _serve():
            while result_queue.empty():
                server.handle_request()

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        thread.join(timeout=timeout + 5)
        server.server_close()
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return None

    def exchange(self, code: str, use_local_server: bool = True, alt_port: bool = False) -> dict[str, Any]:
        """Exchange auth code for tokens."""
        if alt_port:
            redirect = _ALT_REDIRECT_URI
        else:
            redirect = _REDIRECT_URI
        return exchange_code(code, self._client_id, self._client_secret, redirect)

    def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an expired access token."""
        if requests is None:
            raise ImportError("requests library required for OAuth")

        resp = requests.post(_GOOGLE_TOKEN_URL, data={
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }, timeout=30)
        resp.raise_for_status()
        return resp.json()
