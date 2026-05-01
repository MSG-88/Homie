"""Homie plugin for local vision-language understanding using Qwen3-VL-2B-Instruct.

Provides image analysis, visual question answering, and image-to-text
capabilities running entirely on-device via the Qwen3-VL-2B-Instruct
multimodal model. Supports single images, multi-image comparison,
and conversational visual reasoning without any network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.3


@dataclass
class VisionResult:
    """Container for a vision inference result."""

    query: str
    response: str
    image_paths: List[str]
    tokens_used: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class PluginConfig:
    """Configuration for the visual understanding plugin."""

    model_id: str = MODEL_ID
    device: str = "cpu"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginConfig:
        """Build config from a dictionary, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


def _validate_image_path(path: str | Path) -> Path:
    """Validate that an image path exists and has a supported extension."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{p.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return p


class Qwen3VLVisualUnderstandingPlugin:
    """Local vision-language plugin powered by Qwen3-VL-2B-Instruct.

    Provides on-device image understanding for Homie, including:
    - Visual question answering ("What is in this image?")
    - Image description and captioning
    - Multi-image comparison
    - Structured extraction from screenshots or documents
    - Multi-turn visual conversations

    All inference runs locally; no network calls are made after model load.
    """

    name: str = "qwen3_vl_visual_understanding"
    version: str = "1.0.0"

    def __init__(self, config: Optional[PluginConfig] = None) -> None:
        self._config = config or PluginConfig()
        self._model: Any = None
        self._processor: Any = None
        self._active = False
        self._conversation_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Load the model and processor into memory."""
        if self._active:
            logger.info("%s already active, skipping reload", self.name)
            return

        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies. Install with: "
                "pip install transformers torch accelerate qwen-vl-utils Pillow"
            ) from exc

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._config.torch_dtype, torch.float16)

        logger.info(
            "Loading %s on device=%s dtype=%s",
            self._config.model_id,
            self._config.device,
            self._config.torch_dtype,
        )

        self._processor = AutoProcessor.from_pretrained(
            self._config.model_id,
            trust_remote_code=self._config.trust_remote_code,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._config.model_id,
            torch_dtype=torch_dtype,
            device_map=self._config.device if self._config.device != "cpu" else None,
            low_cpu_mem_usage=self._config.low_cpu_mem_usage,
            trust_remote_code=self._config.trust_remote_code,
        )
        if self._config.device == "cpu":
            self._model = self._model.to("cpu")

        self._model.eval()
        self._active = True
        logger.info("%s activated successfully", self.name)

    def deactivate(self) -> None:
        """Release model resources and clear conversation history."""
        if not self._active:
            return
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._conversation_history.clear()
        self._active = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("%s deactivated", self.name)

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                f"{self.name} is not active. Call activate() before inference."
            )

    def _build_messages(
        self,
        query: str,
        image_paths: Sequence[str | Path],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build the chat messages list for the processor."""
        messages: List[Dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        user_content: List[Dict[str, Any]] = []
        for img in image_paths:
            validated = _validate_image_path(img)
            user_content.append({"type": "image", "image": f"file://{validated}"})

        user_content.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _generate(self, messages: List[Dict[str, Any]]) -> tuple[str, int]:
        """Run generation and return (text, token_count)."""
        import torch
        from qwen_vl_utils import process_vision_info

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
                do_sample=self._config.temperature > 0,
            )

        # Trim input tokens from output
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return text, generated_ids.shape[1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        image_paths: Sequence[str | Path],
        system_prompt: Optional[str] = None,
    ) -> VisionResult:
        """Ask a question about one or more images.

        Args:
            query: Natural-language question or instruction.
            image_paths: One or more local image file paths.
            system_prompt: Optional system-level instruction.

        Returns:
            VisionResult with the model's response.
        """
        self._ensure_active()
        str_paths = [str(p) for p in image_paths]

        try:
            messages = self._build_messages(query, image_paths, system_prompt)
            text, tokens = self._generate(messages)
            return VisionResult(
                query=query,
                response=text,
                image_paths=str_paths,
                tokens_used=tokens,
            )
        except Exception as exc:
            logger.exception("Vision inference failed")
            return VisionResult(
                query=query,
                response="",
                image_paths=str_paths,
                error=str(exc),
            )

    def describe(self, image_path: str | Path) -> VisionResult:
        """Generate a detailed description of a single image."""
        return self.ask(
            "Describe this image in detail, including objects, colors, layout, and any text visible.",
            [image_path],
        )

    def compare(self, image_paths: Sequence[str | Path]) -> VisionResult:
        """Compare two or more images and describe differences."""
        if len(image_paths) < 2:
            return VisionResult(
                query="compare",
                response="",
                image_paths=[str(p) for p in image_paths],
                error="At least two images are required for comparison.",
            )
        return self.ask(
            "Compare these images. Describe the key similarities and differences.",
            image_paths,
        )

    def extract_text(self, image_path: str | Path) -> VisionResult:
        """Extract any visible text from an image (OCR-like)."""
        return self.ask(
            "Extract and return all visible text from this image, preserving layout where possible.",
            [image_path],
            system_prompt="You are a precise text extraction assistant. Return only the text found in the image.",
        )

    def chat(
        self,
        query: str,
        image_paths: Optional[Sequence[str | Path]] = None,
    ) -> VisionResult:
        """Multi-turn visual conversation with history tracking.

        Images only need to be provided on the first turn or when
        introducing new images into the conversation.
        """
        self._ensure_active()
        paths = list(image_paths or [])
        str_paths = [str(p) for p in paths]

        try:
            user_content: List[Dict[str, Any]] = []
            for img in paths:
                validated = _validate_image_path(img)
                user_content.append({"type": "image", "image": f"file://{validated}"})
            user_content.append({"type": "text", "text": query})

            self._conversation_history.append({"role": "user", "content": user_content})

            text, tokens = self._generate(self._conversation_history)

            self._conversation_history.append(
                {"role": "assistant", "content": [{"type": "text", "text": text}]}
            )

            return VisionResult(
                query=query,
                response=text,
                image_paths=str_paths,
                tokens_used=tokens,
            )
        except Exception as exc:
            logger.exception("Visual chat inference failed")
            return VisionResult(
                query=query,
                response="",
                image_paths=str_paths,
                error=str(exc),
            )

    def clear_history(self) -> None:
        """Reset the multi-turn conversation history."""
        self._conversation_history.clear()


# ------------------------------------------------------------------
# Module-level convenience for Homie plugin registration
# ------------------------------------------------------------------

_plugin_instance: Optional[Qwen3VLVisualUnderstandingPlugin] = None


def register(config: Optional[Dict[str, Any]] = None) -> Qwen3VLVisualUnderstandingPlugin:
    """Register and return the plugin singleton.

    Called by Homie's plugin loader at startup.  The model is **not**
    loaded until ``activate()`` is explicitly called, keeping startup fast.

    Args:
        config: Optional dict of overrides (device, max_new_tokens, etc.).

    Returns:
        The plugin instance (not yet activated).
    """
    global _plugin_instance
    if _plugin_instance is None:
        plugin_cfg = PluginConfig.from_dict(config) if config else PluginConfig()
        _plugin_instance = Qwen3VLVisualUnderstandingPlugin(config=plugin_cfg)
        logger.info(
            "Registered %s (model=%s, activate to load)",
            _plugin_instance.name,
            plugin_cfg.model_id,
        )
    return _plugin_instance
