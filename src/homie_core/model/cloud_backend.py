"""Backend for OpenAI-compatible cloud APIs (OpenAI, Azure, Groq, etc.)."""

from __future__ import annotations

import json
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class CloudBackend:
    """Backend that calls an OpenAI-compatible REST API over HTTPS."""

    def __init__(self):
        self._base_url: str = ""
        self._model: str = ""
        self._api_key: str = ""
        self._connected: bool = False

    # -- lifecycle -----------------------------------------------------------

    def load(
        self,
        model_id: str,
        api_key: str,
        base_url: str = "https://api.openai.com",
    ) -> None:
        """Connect to the API and verify reachability via the models endpoint."""
        self._base_url = base_url.rstrip("/")
        self._model = model_id
        self._api_key = api_key

        # Verify the endpoint is reachable (auth errors are fine — the
        # server answered, so the connection works; auth is checked at
        # generation time).
        try:
            req = Request(f"{self._base_url}/v1/models", method="GET")
            req.add_header("Authorization", f"Bearer {self._api_key}")
            with urlopen(req, timeout=10):
                pass
        except HTTPError:
            # Server responded (e.g. 401) — connection is alive
            pass
        except (URLError, OSError) as exc:
            self._connected = False
            raise ConnectionError(f"Cannot reach API at {self._base_url}: {exc}") from exc

        self._connected = True

    def unload(self) -> None:
        """Reset all state."""
        self._base_url = ""
        self._model = ""
        self._api_key = ""
        self._connected = False

    # -- inference -----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        payload: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self._base_url}/v1/chat/completions", data=data, method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            with urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"].get("content", "") or ""
        except HTTPError as exc:
            # Try to extract a human-readable error message from the body
            try:
                body = json.loads(exc.read())
                msg = body.get("error", {}).get("message", str(exc))
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"API error ({exc.code}): {msg}") from exc

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        payload: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self._base_url}/v1/chat/completions", data=data, method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            with urlopen(req, timeout=300) as resp:
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except HTTPError as exc:
            try:
                body = json.loads(exc.read())
                msg = body.get("error", {}).get("message", str(exc))
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"API error ({exc.code}): {msg}") from exc

    # -- discovery -----------------------------------------------------------

    def discover_models(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com",
    ) -> list[str]:
        """Query /v1/models and return a list of model IDs. Returns [] on failure."""
        url = f"{base_url.rstrip('/')}/v1/models"
        req = Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []
