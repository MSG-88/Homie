"""Tests for CloudBackend using a mock HTTP server."""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import ClassVar

import pytest


# ---------------------------------------------------------------------------
# Mock OpenAI-compatible HTTP server
# ---------------------------------------------------------------------------

_MODELS_RESPONSE = {
    "data": [
        {"id": "gpt-4o"},
        {"id": "gpt-4o-mini"},
        {"id": "text-embedding-3-small"},
    ]
}


class _MockHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible API handler."""

    valid_token: ClassVar[str] = "sk-test-key"

    # Silence request logging during tests
    def log_message(self, format, *args):  # noqa: A002
        pass

    # -- auth helper --------------------------------------------------------
    def _check_auth(self) -> bool:
        auth = self.headers.get("Authorization", "")
        if auth != f"Bearer {self.valid_token}":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = json.dumps({"error": {"message": "Invalid API key", "type": "invalid_request_error"}})
            self.wfile.write(body.encode())
            return False
        return True

    # -- GET /v1/models -----------------------------------------------------
    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            if not self._check_auth():
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_MODELS_RESPONSE).encode())
        else:
            self.send_response(404)
            self.end_headers()

    # -- POST /v1/chat/completions ------------------------------------------
    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        if not self._check_auth():
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        if body.get("stream"):
            self._handle_stream(body)
        else:
            self._handle_generate(body)

    def _handle_generate(self, body: dict):
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from cloud!"},
                    "finish_reason": "stop",
                }
            ],
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def _handle_stream(self, body: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        chunks = ["Hello", " from", " cloud!"]
        for token in chunks:
            chunk = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": token}, "finish_reason": None}
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_server():
    """Start a mock HTTP server for the duration of this test module."""
    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture()
def backend():
    from homie_core.model.cloud_backend import CloudBackend
    return CloudBackend()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

API_KEY = "sk-test-key"


def test_load_connects_and_discovers_models(mock_server, backend):
    backend.load("gpt-4o", api_key=API_KEY, base_url=mock_server)
    assert backend._connected is True
    assert backend._model == "gpt-4o"


def test_load_connection_failure(backend):
    with pytest.raises(ConnectionError):
        backend.load("gpt-4o", api_key=API_KEY, base_url="http://127.0.0.1:1")


def test_generate(mock_server, backend):
    backend.load("gpt-4o", api_key=API_KEY, base_url=mock_server)
    result = backend.generate("Say hello")
    assert result == "Hello from cloud!"


def test_generate_auth_failure(mock_server, backend):
    backend.load("gpt-4o", api_key="bad-key", base_url=mock_server)
    with pytest.raises(RuntimeError, match="Invalid API key"):
        backend.generate("Say hello")


def test_stream(mock_server, backend):
    backend.load("gpt-4o", api_key=API_KEY, base_url=mock_server)
    chunks = list(backend.stream("Say hello"))
    assert chunks == ["Hello", " from", " cloud!"]


def test_unload(mock_server, backend):
    backend.load("gpt-4o", api_key=API_KEY, base_url=mock_server)
    backend.unload()
    assert backend._connected is False
    assert backend._model == ""
    assert backend._api_key == ""


def test_discover_models(mock_server, backend):
    models = backend.discover_models(api_key=API_KEY, base_url=mock_server)
    assert models == ["gpt-4o", "gpt-4o-mini", "text-embedding-3-small"]


def test_discover_models_failure(backend):
    models = backend.discover_models(api_key=API_KEY, base_url="http://127.0.0.1:1")
    assert models == []
