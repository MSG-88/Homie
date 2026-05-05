"""Homie Visual Understanding Plugin â€” Qwen3-VL-2B-Instruct

Provides local vision-language capabilities using the Qwen3-VL-2B-Instruct model
via the Hugging Face Transformers library.  Supports image captioning, visual
question-answering, and structured image analysis entirely on-device.

The model is loaded lazily on first use and cached for subsequent calls.  All
inference runs locally â€” no network calls are made after the initial (opt-in)
model download.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_DEVICE = "cpu"


@dataclass
class VisualResult:
    """Encapsulates a single vision-language inference result."""

    query: str
    answer: str
    image_path: str
    model_id: str = MODEL_ID
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualUnderstandingPlugin:
    """Homie plugin for local visual understanding via Qwen3-VL-2B-Instruct.

    Lifecycle
    ---------
    * ``activate()``  â€” validates dependencies and prepares config (lazy load).
    * ``deactivate()`` â€” releases the model from memory.

    Public API
    ----------
    * ``caption(image_path)``        â€” generate a natural-language caption.
    * ``ask(image_path, question)``  â€” visual question-answering.
    * ``analyze(image_path, prompt)``â€” freeform vision-language prompt.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = DEFAULT_DEVICE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self._model: Any = None
        self._processor: Any = None
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Validate that required libraries are importable.

        The actual model is loaded lazily on first inference call to keep
        activation lightweight.
        """
        try:
            import torch  # noqa: F401
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for VisualUnderstandingPlugin. "
                "Install with: pip install transformers torch qwen-vl-utils Pillow"
            ) from exc
        self._active = True
        logger.info("VisualUnderstandingPlugin activated (model loads on first use).")

    def deactivate(self) -> None:
        """Release model and processor from memory."""
        self._model = None
        self._processor = None
        self._active = False
        logger.info("VisualUnderstandingPlugin deactivated.")

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the model and processor if not already in memory."""
        if not self._active:
            raise RuntimeError("Plugin is not activated. Call activate() first.")
        if self._model is not None:
            return

        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info("Loading model %s on %s â€¦", self.model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device if self.device != "cpu" else None,
            cache_dir=self.cache_dir,
        )
        if self.device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()
        logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _infer(self, image_path: str, user_text: str) -> str:
        """Run a single vision-language inference turn."""
        import torch
        from PIL import Image

        self._ensure_loaded()
        assert self._processor is not None and self._model is not None

        img = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=[img],
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        # Trim input tokens from output
        trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def caption(self, image_path: str) -> VisualResult:
        """Generate a descriptive caption for the image at *image_path*."""
        self._validate_image(image_path)
        query = "Describe this image in detail."
        answer = self._infer(image_path, query)
        return VisualResult(query=query, answer=answer, image_path=image_path)

    def ask(self, image_path: str, question: str) -> VisualResult:
        """Answer a natural-language *question* about the given image."""
        self._validate_image(image_path)
        answer = self._infer(image_path, question)
        return VisualResult(query=question, answer=answer, image_path=image_path)

    def analyze(self, image_path: str, prompt: str) -> VisualResult:
        """Run an arbitrary vision-language *prompt* against the image."""
        self._validate_image(image_path)
        answer = self._infer(image_path, prompt)
        return VisualResult(query=prompt, answer=answer, image_path=image_path)

    def batch_caption(self, image_paths: Sequence[str]) -> List[VisualResult]:
        """Caption multiple images sequentially."""
        return [self.caption(p) for p in image_paths]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image_path: str) -> None:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}:
            raise ValueError(f"Unsupported image format: {p.suffix}")


# ------------------------------------------------------------------
# Module-level convenience (matches Homie plugin registration pattern)
# ------------------------------------------------------------------

_default_instance: Optional[VisualUnderstandingPlugin] = None


def register(config: Optional[Dict[str, Any]] = None) -> VisualUnderstandingPlugin:
    """Create, configure, and activate the plugin.

    Parameters
    ----------
    config : dict, optional
        Keys: ``model_id``, ``device``, ``max_new_tokens``, ``cache_dir``.
    """
    global _default_instance
    cfg = config or {}
    plugin = VisualUnderstandingPlugin(
        model_id=cfg.get("model_id", MODEL_ID),
        device=cfg.get("device", DEFAULT_DEVICE),
        max_new_tokens=int(cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
        cache_dir=cfg.get("cache_dir"),
    )
    plugin.activate()
    _default_instance = plugin
    return plugin
