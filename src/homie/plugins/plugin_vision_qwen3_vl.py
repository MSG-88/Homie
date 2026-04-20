"""Homie Vision-Language Plugin â€” Qwen3-VL-2B-Instruct

Provides local multimodal (image + text) inference using the Qwen3-VL-2B-Instruct
model via llama.cpp's GGUF backend. Users can ask questions about images, get
descriptions, extract text, or perform visual reasoning entirely offline.

Requires a GGUF-quantised build of Qwen3-VL-2B-Instruct placed in the
configured model directory (default: ~/.homie/models/).
"""

from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from homie.config import HomieConfig, cfg_get

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path.home() / ".homie" / "models"
DEFAULT_MODEL_FILENAME = "qwen3-vl-2b-instruct.gguf"
DEFAULT_MMPROJ_FILENAME = "qwen3-vl-2b-instruct-mmproj.gguf"
DEFAULT_CTX_SIZE = 2048
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 512
DEFAULT_GPU_LAYERS = 0

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


@dataclass
class VisionResult:
    """Container for a vision-language inference result."""

    text: str
    model: str
    image_path: str
    prompt: str
    tokens_predicted: Optional[int] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class VisionPluginConfig:
    """Resolved configuration for the vision plugin."""

    model_path: Path
    mmproj_path: Path
    llama_cli_path: str
    ctx_size: int
    temperature: float
    max_tokens: int
    gpu_layers: int
    extra_args: List[str] = field(default_factory=list)


class QwenVisionPlugin:
    """Local vision-language plugin powered by Qwen3-VL-2B-Instruct (GGUF).

    Integrates with Homie's configuration system and uses llama.cpp for
    inference.  All processing happens on-device â€” no network calls.

    Configuration keys (under ``plugins.vision_qwen3_vl`` in homie.config.yaml)::

        plugins:
          vision_qwen3_vl:
            enabled: true
            model_dir: ~/.homie/models
            model_filename: qwen3-vl-2b-instruct.gguf
            mmproj_filename: qwen3-vl-2b-instruct-mmproj.gguf
            llama_cli: llama-llava-cli          # or absolute path
            ctx_size: 2048
            temperature: 0.3
            max_tokens: 512
            gpu_layers: 0                       # set >0 if GPU offload available
            extra_args: []                      # additional CLI flags
    """

    name: str = "vision_qwen3_vl"
    version: str = "0.1.0"

    def __init__(self) -> None:
        self._active: bool = False
        self._plugin_cfg: Optional[VisionPluginConfig] = None
        self._homie_cfg: Optional[HomieConfig] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self, cfg: HomieConfig) -> None:
        """Activate the plugin â€” resolve and validate configuration."""
        self._homie_cfg = cfg
        self._plugin_cfg = self._resolve_config(cfg)
        self._validate()
        self._active = True
        logger.info(
            "QwenVisionPlugin activated  model=%s  mmproj=%s  gpu_layers=%d",
            self._plugin_cfg.model_path,
            self._plugin_cfg.mmproj_path,
            self._plugin_cfg.gpu_layers,
        )

    def deactivate(self) -> None:
        """Deactivate the plugin and release references."""
        self._active = False
        self._plugin_cfg = None
        self._homie_cfg = None
        logger.info("QwenVisionPlugin deactivated")

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe_image(self, image_path: str | Path, prompt: Optional[str] = None) -> VisionResult:
        """Run vision-language inference on a local image.

        Parameters
        ----------
        image_path:
            Absolute or relative path to a local image file.
        prompt:
            Free-form user prompt.  Defaults to a generic description request.

        Returns
        -------
        VisionResult
            The model output together with metadata.
        """
        if not self._active or self._plugin_cfg is None:
            return VisionResult(
                text="",
                model="",
                image_path=str(image_path),
                prompt=prompt or "",
                error="Plugin is not activated. Call activate() first.",
            )

        image_path = Path(image_path).resolve()
        if not image_path.is_file():
            return VisionResult(
                text="",
                model=str(self._plugin_cfg.model_path),
                image_path=str(image_path),
                prompt=prompt or "",
                error=f"Image not found: {image_path}",
            )

        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return VisionResult(
                text="",
                model=str(self._plugin_cfg.model_path),
                image_path=str(image_path),
                prompt=prompt or "",
                error=f"Unsupported image format '{image_path.suffix}'. Supported: {SUPPORTED_IMAGE_EXTENSIONS}",
            )

        user_prompt = prompt or "Describe this image in detail."
        return self._run_inference(image_path, user_prompt)

    def answer_question(self, image_path: str | Path, question: str) -> VisionResult:
        """Answer a question about a local image."""
        return self.describe_image(image_path, prompt=question)

    def extract_text(self, image_path: str | Path) -> VisionResult:
        """Attempt OCR-like text extraction from an image."""
        return self.describe_image(
            image_path,
            prompt="Extract and list all readable text from this image verbatim.",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config(cfg: HomieConfig) -> VisionPluginConfig:
        """Build a typed config from the Homie config tree."""
        section = "vision_qwen3_vl"
        model_dir = Path(
            cfg_get(cfg, "plugins", section, "model_dir", default=str(DEFAULT_MODEL_DIR))
        ).expanduser()
        model_filename = cfg_get(cfg, "plugins", section, "model_filename", default=DEFAULT_MODEL_FILENAME)
        mmproj_filename = cfg_get(cfg, "plugins", section, "mmproj_filename", default=DEFAULT_MMPROJ_FILENAME)

        return VisionPluginConfig(
            model_path=model_dir / model_filename,
            mmproj_path=model_dir / mmproj_filename,
            llama_cli_path=cfg_get(cfg, "plugins", section, "llama_cli", default="llama-llava-cli"),
            ctx_size=int(cfg_get(cfg, "plugins", section, "ctx_size", default=DEFAULT_CTX_SIZE)),
            temperature=float(cfg_get(cfg, "plugins", section, "temperature", default=DEFAULT_TEMPERATURE)),
            max_tokens=int(cfg_get(cfg, "plugins", section, "max_tokens", default=DEFAULT_MAX_TOKENS)),
            gpu_layers=int(cfg_get(cfg, "plugins", section, "gpu_layers", default=DEFAULT_GPU_LAYERS)),
            extra_args=cfg_get(cfg, "plugins", section, "extra_args", default=[]) or [],
        )

    def _validate(self) -> None:
        """Check that required model files exist."""
        assert self._plugin_cfg is not None
        missing: List[str] = []
        if not self._plugin_cfg.model_path.is_file():
            missing.append(f"Model GGUF not found: {self._plugin_cfg.model_path}")
        if not self._plugin_cfg.mmproj_path.is_file():
            missing.append(f"Multimodal projector GGUF not found: {self._plugin_cfg.mmproj_path}")
        if missing:
            raise FileNotFoundError(
                "QwenVisionPlugin cannot activate â€” missing files:\n  "
                + "\n  ".join(missing)
                + "\nDownload GGUF quantisations from https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct"
            )

    def _build_command(self, image_path: Path, prompt: str) -> List[str]:
        """Assemble the llama.cpp CLI invocation."""
        assert self._plugin_cfg is not None
        cmd: List[str] = [
            self._plugin_cfg.llama_cli_path,
            "--model", str(self._plugin_cfg.model_path),
            "--mmproj", str(self._plugin_cfg.mmproj_path),
            "--image", str(image_path),
            "--prompt", prompt,
            "--ctx-size", str(self._plugin_cfg.ctx_size),
            "--temp", str(self._plugin_cfg.temperature),
            "--n-predict", str(self._plugin_cfg.max_tokens),
            "--n-gpu-layers", str(self._plugin_cfg.gpu_layers),
        ]
        if self._plugin_cfg.extra_args:
            cmd.extend(self._plugin_cfg.extra_args)
        return cmd

    def _run_inference(self, image_path: Path, prompt: str) -> VisionResult:
        """Execute llama.cpp and capture output."""
        assert self._plugin_cfg is not None
        cmd = self._build_command(image_path, prompt)
        logger.debug("Running vision inference: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            return VisionResult(
                text="",
                model=str(self._plugin_cfg.model_path),
                image_path=str(image_path),
                prompt=prompt,
                error=(
                    f"llama.cpp CLI not found at '{self._plugin_cfg.llama_cli_path}'. "
                    "Install llama.cpp and ensure the binary is on PATH."
                ),
            )
        except subprocess.TimeoutExpired:
            return VisionResult(
                text="",
                model=str(self._plugin_cfg.model_path),
                image_path=str(image_path),
                prompt=prompt,
                error="Inference timed out after 120 seconds.",
            )

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "").strip()[:500]
            return VisionResult(
                text="",
                model=str(self._plugin_cfg.model_path),
                image_path=str(image_path),
                prompt=prompt,
                error=f"llama.cpp exited with code {result.returncode}: {stderr_snippet}",
            )

        output_text = (result.stdout or "").strip()
        return VisionResult(
            text=output_text,
            model=str(self._plugin_cfg.model_path),
            image_path=str(image_path),
            prompt=prompt,
        )


# ------------------------------------------------------------------
# Module-level convenience for Homie's plugin loader
# ------------------------------------------------------------------

_instance: Optional[QwenVisionPlugin] = None


def register() -> QwenVisionPlugin:
    """Return a singleton plugin instance for Homie's plugin registry."""
    global _instance
    if _instance is None:
        _instance = QwenVisionPlugin()
    return _instance


__all__ = [
    "QwenVisionPlugin",
    "VisionResult",
    "VisionPluginConfig",
    "register",
]
