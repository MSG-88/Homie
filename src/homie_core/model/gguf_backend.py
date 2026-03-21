from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


# Default location for the bundled llama-server binary
_DEFAULT_SERVER_DIR = Path.home() / ".homie" / "llama-server"


def _find_llama_server() -> str:
    """Locate the llama-server executable."""
    exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"

    # Check bundled location first (known-good version)
    candidate = _DEFAULT_SERVER_DIR / exe
    if candidate.exists():
        return str(candidate)

    # Check PATH
    for d in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(d) / exe
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"llama-server not found. Install it to {_DEFAULT_SERVER_DIR} or add it to PATH.\n"
        f"Download from: https://github.com/ggml-org/llama.cpp/releases"
    )


def _find_cuda_libs() -> list[str]:
    """Find directories containing CUDA runtime libraries."""
    cuda_dirs = []
    search_paths = [
        # LM Studio bundled CUDA libs
        Path.home() / ".lmstudio" / "extensions" / "backends" / "vendor" / "win-llama-cuda12-vendor-v2",
        Path.home() / ".lmstudio" / "extensions" / "backends" / "vendor" / "win-llama-cuda-vendor-v2",
        # Standard CUDA toolkit locations
        Path(os.environ.get("CUDA_PATH", "")) / "bin" if os.environ.get("CUDA_PATH") else None,
    ]
    for p in search_paths:
        if p and p.exists():
            cuda_dirs.append(str(p))
    return cuda_dirs


def _find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class GGUFBackend:
    """Backend that runs llama-server as a subprocess and communicates via HTTP."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._base_url: str = ""
        self._model_path: str = ""

    @staticmethod
    def _compute_safe_gpu_layers(model_path: str | Path) -> int:
        """Calculate GPU layers based on model file size and available VRAM."""
        try:
            from homie_core.hardware.detector import HardwareDetector
            from homie_core.hardware.profiles import compute_gpu_layers
            hw = HardwareDetector()
            vram = hw.best_gpu_vram_gb
            if vram <= 0:
                return 0  # no GPU
            model_size_gb = Path(model_path).stat().st_size / (1024**3)
            layers = compute_gpu_layers(model_size_gb, vram)
            return layers
        except Exception:
            return -1  # fallback to full offload attempt

    def load(self, model_path: str | Path, n_ctx: int = 16384, n_gpu_layers: int = -1, **kwargs) -> None:
        server_bin = _find_llama_server()
        port = _find_free_port()
        self._base_url = f"http://127.0.0.1:{port}"
        self._model_path = str(model_path)

        # Smart GPU layer calculation when set to full offload (-1)
        if n_gpu_layers == -1:
            n_gpu_layers = self._compute_safe_gpu_layers(model_path)

        cmd = [
            server_bin,
            "--model", str(model_path),
            "--ctx-size", str(n_ctx),
            "--n-gpu-layers", str(n_gpu_layers),
            "--host", "127.0.0.1",
            "--port", str(port),
        ]

        # GPU-specific optimisations: flash-attention and quantised KV cache
        if n_gpu_layers != 0:
            cmd += [
                "--flash-attn", "on",
                "--cache-type-k", "q8_0",
                "--cache-type-v", "q8_0",
            ]

        # Build environment with CUDA library paths
        env = os.environ.copy()
        cuda_dirs = _find_cuda_libs()
        server_dir = str(Path(server_bin).parent)
        extra_paths = [server_dir] + cuda_dirs
        env["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + env.get("PATH", "")

        # Start the server process
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=creation_flags,
        )

        # Register cleanup
        atexit.register(self._kill_server)

        # Wait for the server to become ready (large models can take a while)
        self._wait_for_server(timeout=300)
        self._warm_up()

    def _wait_for_server(self, timeout: int = 300) -> None:
        """Poll the health endpoint until the server is ready."""
        deadline = time.monotonic() + timeout
        last_error = None
        printed_loading = False
        while time.monotonic() < deadline:
            # Check if process died
            if self._process and self._process.poll() is not None:
                stderr = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
                raise RuntimeError(
                    f"llama-server exited with code {self._process.returncode}.\n"
                    f"stderr: {stderr[-2000:]}"
                )
            try:
                req = Request(f"{self._base_url}/health", method="GET")
                with urlopen(req, timeout=2) as resp:
                    data = json.loads(resp.read())
                    status = data.get("status", "")
                    if status == "ok":
                        if printed_loading:
                            print()  # newline after dots
                        return
                    # "loading model" — keep waiting
                    if not printed_loading:
                        print("  Loading model into memory", end="", flush=True)
                        printed_loading = True
                    print(".", end="", flush=True)
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
                last_error = e
            time.sleep(2)
        if printed_loading:
            print()
        raise TimeoutError(
            f"llama-server did not become ready within {timeout}s. Last error: {last_error}"
        )

    def _warm_up(self) -> None:
        """Pre-populate CUDA kernels with a minimal inference call."""
        try:
            self.generate("Hi", max_tokens=1, temperature=0.0)
        except Exception:
            pass

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> str:
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        req = Request(f"{self._base_url}/v1/chat/completions", data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        with urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            msg = result["choices"][0]["message"]
            content = msg.get("content", "") or ""
            return content

    def stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> Iterator[str]:
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        req = Request(f"{self._base_url}/v1/chat/completions", data=data, method="POST")
        req.add_header("Content-Type", "application/json")

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
                    # Only yield actual content, skip reasoning_content from thinking models
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def _kill_server(self) -> None:
        """Terminate the llama-server subprocess."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

    def unload(self) -> None:
        self._kill_server()
        self._process = None
        self._base_url = ""
        self._model_path = ""
