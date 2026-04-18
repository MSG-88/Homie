"""HuggingFace Local Backend — serve models via transformers directly.

Replaces Ollama entirely. Loads models from HF Hub or local path,
runs inference on local GPU via transformers + torch.

Supports:
- Loading from HF Hub (downloads on first use)
- Loading from local merged/fine-tuned models
- Streaming token generation
- Automatic GPU/CPU detection
- Model quantization (4-bit via bitsandbytes)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class HFLocalBackend:
    """Inference backend using HuggingFace transformers locally."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_id: str = ""
        self._loaded: bool = False
        self._device: str = "cpu"
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(
        self,
        model_id: str,
        quantize_4bit: bool = True,
        max_memory: Optional[dict] = None,
        trust_remote_code: bool = True,
    ) -> None:
        """Load a model from HF Hub or local path.

        Args:
            model_id: HF model ID (e.g., 'Muthu88/Homie') or local path
            quantize_4bit: Use 4-bit quantization (saves VRAM)
            max_memory: GPU memory allocation map
            trust_remote_code: Allow custom model code
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model: %s (4bit=%s)", model_id, quantize_4bit)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": "auto",
        }

        if quantize_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self._device = "cuda"
        elif torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
            self._device = "cuda"
        else:
            load_kwargs["torch_dtype"] = torch.float32
            load_kwargs["device_map"] = "cpu"
            self._device = "cpu"

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self._model.eval()
        self._model_id = model_id
        self._loaded = True

        param_count = sum(p.numel() for p in self._model.parameters())
        logger.info("Model loaded: %s (%.1fM params, device=%s)",
                     model_id, param_count / 1e6, self._device)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        timeout: int = 120,
    ) -> str:
        """Generate a response."""
        import torch

        if not self._loaded:
            raise RuntimeError("No model loaded")

        with self._lock:
            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Apply stop sequences
            if stop:
                for s in stop:
                    if s in response:
                        response = response[:response.index(s)]

            return response.strip()

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """Stream tokens one at a time."""
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread

        if not self._loaded:
            raise RuntimeError("No model loaded")

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True,
        )

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=self._model.generate, kwargs=generate_kwargs)
        thread.start()

        stop_set = set(stop) if stop else set()
        accumulated = ""
        for token in streamer:
            accumulated += token
            # Check stop sequences
            should_stop = False
            for s in stop_set:
                if s in accumulated:
                    # Yield up to the stop sequence
                    idx = accumulated.index(s)
                    remaining = accumulated[:idx]
                    if remaining:
                        yield remaining
                    should_stop = True
                    break
            if should_stop:
                break
            yield token

        thread.join(timeout=5)

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        self._model_id = ""

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
