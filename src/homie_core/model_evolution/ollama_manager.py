"""OllamaManager — wrapper around Ollama CLI commands."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_OLLAMA_CMD = "ollama"
_TIMEOUT = 300  # 5 minutes for pull/push


class OllamaManager:
    """Manages Ollama model operations via CLI."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key

    def _run(self, args: list[str], timeout: int = _TIMEOUT, env_extra: Optional[dict] = None) -> subprocess.CompletedProcess:
        """Run an ollama command."""
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            [_OLLAMA_CMD] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

    def pull(self, model: str) -> bool:
        """Pull a model from the registry."""
        try:
            result = self._run(["pull", model])
            if result.returncode == 0:
                logger.info("Pulled model: %s", model)
                return True
            logger.warning("Failed to pull %s: %s", model, result.stderr)
            return False
        except Exception as exc:
            logger.error("Pull failed: %s", exc)
            return False

    def create(self, name: str, modelfile: Path | str) -> bool:
        """Create a model from a Modelfile."""
        try:
            result = self._run(["create", name, "-f", str(modelfile)])
            if result.returncode == 0:
                logger.info("Created model: %s", name)
                return True
            logger.warning("Failed to create %s: %s", name, result.stderr)
            return False
        except Exception as exc:
            logger.error("Create failed: %s", exc)
            return False

    def push(self, name: str) -> bool:
        """Push a model to the registry."""
        env_extra = {}
        if self._api_key:
            env_extra["OLLAMA_API_KEY"] = self._api_key
        try:
            result = self._run(["push", name], env_extra=env_extra)
            if result.returncode == 0:
                logger.info("Pushed model: %s", name)
                return True
            logger.warning("Failed to push %s: %s", name, result.stderr)
            return False
        except Exception as exc:
            logger.error("Push failed: %s", exc)
            return False

    def list_models(self) -> list[str]:
        """List installed models."""
        try:
            result = self._run(["list"], timeout=30)
            if result.returncode != 0:
                return []
            lines = result.stdout.strip().split("\n")
            # Skip header line
            return [line.split()[0] for line in lines[1:] if line.strip()]
        except Exception:
            return []

    def show(self, name: str) -> str:
        """Show model info."""
        try:
            result = self._run(["show", name], timeout=30)
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    def remove(self, name: str) -> bool:
        """Remove a model."""
        try:
            result = self._run(["rm", name], timeout=60)
            return result.returncode == 0
        except Exception:
            return False
