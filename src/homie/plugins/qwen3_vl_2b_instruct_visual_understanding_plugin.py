"""Homie Visual Understanding Plugin â€” Qwen3-VL-2B-Instruct

Provides local vision-language capabilities using the Qwen3-VL-2B-Instruct model.
Supports image captioning, visual question answering, and image-based reasoning
entirely on-device without network calls.

Requires: transformers, torch, Pillow
Optional: flash-attn (for faster inference on supported GPUs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_DEVICE = "auto"


@dataclass
class VLConfig:
    """Configuration for the visual understanding plugin."""

    model_id: str = DEFAULT_MODEL_ID
    device: str = DEFAULT_DEVICE
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    torch_dtype: str = "auto"
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = True
    local_files_only: bool = False
    cache_dir: Optional[str] = None


class VisualUnderstandingPlugin:
    """Local vision-language plugin powered by Qwen3-VL-2B-Instruct.

    Provides image captioning, visual QA, and multi-turn visual conversation
    running entirely on the local device.
    """

    name: str = "visual_understanding"
    version: str = "1.0.0"

    def __init__(self, config: Optional[VLConfig] = None) -> None:
        self.config = config or VLConfig()
        self._model = None
        self._processor = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Load the model and processor into memory."""
        if self._active:
            logger.info("Visual understanding plugin already active.")
            return

        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies. Install with: "
                "pip install transformers torch Pillow"
            ) from exc

        logger.info("Loading model %s ...", self.config.model_id)

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, "auto")

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": self.config.device,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.cache_dir:
            load_kwargs["cache_dir"] = self.config.cache_dir
        if self.config.local_files_only:
            load_kwargs["local_files_only"] = True

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_id, **load_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir or None,
        )
        self._active = True
        logger.info("Visual understanding plugin activated.")

    def deactivate(self) -> None:
        """Unload model from memory."""
        if not self._active:
            return
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._active = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        logger.info("Visual understanding plugin deactivated.")

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError("Plugin not active. Call activate() first.")

    def describe_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Generate a description or answer a question about a local image.

        Args:
            image_path: Absolute or relative path to a local image file.
            prompt: Optional question or instruction. Defaults to captioning.

        Returns:
            Model-generated text response.
        """
        self._ensure_active()
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        from PIL import Image

        image = Image.open(path).convert("RGB")
        user_prompt = prompt or "Describe this image in detail."

        return self._run_inference(images=[image], prompt=user_prompt)

    def visual_qa(self, image_path: str, question: str) -> str:
        """Answer a specific question about an image.

        Args:
            image_path: Path to the image file.
            question: Natural language question about the image.

        Returns:
            Model-generated answer.
        """
        return self.describe_image(image_path, prompt=question)

    def compare_images(self, image_paths: List[str], prompt: Optional[str] = None) -> str:
        """Compare multiple images and describe differences or similarities.

        Args:
            image_paths: List of paths to image files (2+).
            prompt: Optional comparison instruction.

        Returns:
            Model-generated comparison text.
        """
        self._ensure_active()
        from PIL import Image

        images = []
        for p in image_paths:
            path = Path(p).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            images.append(Image.open(path).convert("RGB"))

        user_prompt = prompt or "Compare these images and describe the key differences."
        return self._run_inference(images=images, prompt=user_prompt)

    def _run_inference(self, images: List[Any], prompt: str) -> str:
        """Run model inference on one or more images with a text prompt."""
        import torch

        # Build message in Qwen VL chat format
        content: List[Dict[str, Any]] = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Apply chat template via processor
        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text_input],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        # Decode only the new tokens (skip the input)
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        result = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return result.strip()


def register(homie_config: Optional[Dict[str, Any]] = None) -> VisualUnderstandingPlugin:
    """Factory function to create and configure the plugin from Homie config.

    Args:
        homie_config: Optional dict with plugin-specific settings under
                      'plugins.visual_understanding'.

    Returns:
        Configured (but not yet activated) plugin instance.
    """
    plugin_cfg = {}
    if homie_config:
        plugin_cfg = (
            homie_config.get("plugins", {}).get("visual_understanding", {})
        )

    config = VLConfig(
        model_id=plugin_cfg.get("model_id", DEFAULT_MODEL_ID),
        device=plugin_cfg.get("device", DEFAULT_DEVICE),
        max_new_tokens=plugin_cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
        torch_dtype=plugin_cfg.get("torch_dtype", "auto"),
        low_cpu_mem_usage=plugin_cfg.get("low_cpu_mem_usage", True),
        local_files_only=plugin_cfg.get("local_files_only", False),
        cache_dir=plugin_cfg.get("cache_dir"),
    )
    return VisualUnderstandingPlugin(config=config)
