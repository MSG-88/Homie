"""Homie Vision Plugin â€” Qwen3-VL-2B-Instruct

Provides local multimodal vision-language capabilities using the
Qwen3-VL-2B-Instruct model.  Accepts an image (file path or PIL Image)
and a text prompt, then returns the model's text response.

Designed for edge / desktop deployment: runs entirely on-device with
no network calls.  Supports CPU and CUDA; automatically selects the
best available device.

Typical use-cases:
  - Describe what is on screen or in a photo
  - Extract text from documents or screenshots (lightweight OCR)
  - Answer visual questions ("What colour is the car?")
  - Summarise diagrams or charts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.3


@dataclass
class VisionResult:
    """Structured result from a vision query."""

    text: str
    model: str = MODEL_ID
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class QwenVisionPlugin:
    """Local multimodal vision plugin powered by Qwen3-VL-2B-Instruct.

    Parameters
    ----------
    device : str | None
        ``"cpu"``, ``"cuda"``, or *None* for auto-detect.
    dtype : str
        Torch dtype string â€” ``"float16"``, ``"bfloat16"``, or ``"float32"``.
    max_new_tokens : int
        Maximum tokens the model may generate per query.
    temperature : float
        Sampling temperature.  Lower â†’ more deterministic.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        dtype: str = "float16",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._model: Any = None
        self._processor: Any = None
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Load model and processor into memory."""
        if self._active:
            logger.debug("QwenVisionPlugin already active")
            return

        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "QwenVisionPlugin requires 'torch' and 'transformers>=4.45'. "
                "Install them with:  pip install torch transformers accelerate"
            ) from exc

        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, self.dtype, torch.float16)

        logger.info(
            "Loading %s on %s (%s) â€¦", MODEL_ID, resolved_device, self.dtype,
        )

        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=resolved_device,
        )
        self._model.eval()
        self._active = True
        logger.info("QwenVisionPlugin activated (%s)", resolved_device)

    def deactivate(self) -> None:
        """Unload model and free memory."""
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

        logger.info("QwenVisionPlugin deactivated")

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def query(
        self,
        image: Union[str, Path, "Image.Image"],  # noqa: F821
        prompt: str = "Describe this image.",
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> VisionResult:
        """Run a vision-language query on a single image.

        Parameters
        ----------
        image
            A file path (str / Path) or a ``PIL.Image.Image`` instance.
        prompt
            The text question or instruction.
        max_new_tokens
            Override instance default for this call.
        temperature
            Override instance default for this call.

        Returns
        -------
        VisionResult
            Structured response with generated text and metadata.
        """
        if not self._active:
            raise RuntimeError("Plugin is not active. Call activate() first.")

        from PIL import Image  # type: ignore[import-untyped]
        import torch

        pil_image = self._load_image(image)
        max_tok = max_new_tokens or self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text_input],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tok,
                temperature=temp,
                do_sample=temp > 0,
            )

        # Trim input tokens from output
        input_len = inputs["input_ids"].shape[-1]
        output_ids = generated_ids[:, input_len:]
        response_text: str = self._processor.batch_decode(
            output_ids, skip_special_tokens=True,
        )[0].strip()

        return VisionResult(
            text=response_text,
            tokens_used=int(output_ids.shape[-1]),
            metadata={"prompt": prompt, "max_new_tokens": max_tok, "temperature": temp},
        )

    def describe(self, image: Union[str, Path, "Image.Image"]) -> str:  # noqa: F821
        """Convenience shortcut â€” return a plain-text description."""
        return self.query(image, prompt="Describe this image in detail.").text

    def ocr(self, image: Union[str, Path, "Image.Image"]) -> str:  # noqa: F821
        """Convenience shortcut â€” extract visible text from an image."""
        return self.query(
            image,
            prompt="Extract all visible text from this image. Return only the text, preserving layout where possible.",
        ).text

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(source: Union[str, Path, "Image.Image"]) -> "Image.Image":  # noqa: F821
        from PIL import Image  # type: ignore[import-untyped]

        if isinstance(source, Image.Image):
            return source.convert("RGB")
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")


# ------------------------------------------------------------------
# Module-level register / helpers
# ------------------------------------------------------------------

_default_instance: Optional[QwenVisionPlugin] = None


def register(*, device: Optional[str] = None, dtype: str = "float16") -> QwenVisionPlugin:
    """Create, activate, and return a module-level plugin instance."""
    global _default_instance
    if _default_instance is not None and _default_instance.is_active:
        return _default_instance
    _default_instance = QwenVisionPlugin(device=device, dtype=dtype)
    _default_instance.activate()
    return _default_instance


def get_instance() -> Optional[QwenVisionPlugin]:
    """Return the current module-level instance (may be *None*)."""
    return _default_instance


__all__ = [
    "QwenVisionPlugin",
    "VisionResult",
    "register",
    "get_instance",
]
