"""Slack OAuth 2.0 (V2) flow — local redirect server + code exchange.

Handles:
1. Building the authorization URL
2. Running a local HTTP server to receive the redirect (port 8549)
3. Code-to-token exchange
"""
from __future__ import annotations

import html
import http.server
import queue
import threading
import urllib.parse
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

SLACK_AUTH_URL = "https://slack.com/oauth/v2/authorize"
SLACK_TOKEN_URL = "https://slack.com/api/oauth.v2.access"
SLACK_SCOPES = [
    "channels:history",
    "channels:read",
    "chat:write",
    "groups:history",
    "groups:read",
    "im:history",
    "im:read",
    "mpim:history",
    "mpim:read",
    "search:read",
    "users:read",
]

_REDIRECT_PORT = 8549
_REDIRECT_URI = f"http://localhost:{_REDIRECT_PORT}/callback"


def build_slack_auth_url(client_id: str, redirect_uri: str = _REDIRECT_URI) -> str:
    """Build the Slack OAuth V2 consent screen URL."""
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": ",".join(SLACK_SCOPES),
    }
    return f"{SLACK_AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_slack_code(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str = _REDIRECT_URI,
) -> dict[str, Any]:
    """Exchange authorization code for access token."""
    if requests is None:
        raise ImportError("requests library required for Slack OAuth")

    resp = requests.post(SLACK_TOKEN_URL, data={
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _make_callback_handler(result_queue: queue.Queue):
    """Create a callback handler that writes the auth code to a queue."""

    class _CallbackHandler(http.server.BaseHTTPRequestHandler):
        """HTTP handler to capture OAuth redirect callback."""

        def do_GET(self):
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
                self.wfile.write(
                    b"<html><body><h2>Slack authorization successful!</h2>"
                    b"<p>You can close this tab and return to Homie.</p></body></html>"
                )
            else:
                error = html.escape(params.get("error", ["unknown"])[0])
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"<html><body><h2>Error: {error}</h2></body></html>".encode())

        def log_message(self, format, *args):
            pass  # Suppress server logs

    return _CallbackHandler


class SlackOAuth:
    """Handles Slack OAuth 2.0 V2 lifecycle."""

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = client_id
        self._client_secret = client_secret

    def get_auth_url(self) -> str:
        """Get the authorization URL."""
        return build_slack_auth_url(self._client_id)

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

    def exchange(self, code: str) -> dict[str, Any]:
        """Exchange auth code for tokens."""
        return exchange_slack_code(code, self._client_id, self._client_secret)
