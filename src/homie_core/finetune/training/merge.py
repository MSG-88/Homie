"""LoRA merge and GGUF quantization pipeline for Ollama deployment."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


class ModelMerger:
    """Merge LoRA adapters, quantize to GGUF, and import into Ollama."""

    def __init__(
        self,
        base_model: str,
        registry_name: str = "PyMasters/Homie",
    ):
        self.base_model = base_model
        self.registry_name = registry_name

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_lora(
        self,
        base_model_path: str,
        adapter_path: str,
        output_path: str,
    ) -> bool:
        """Merge LoRA adapter into base model weights via peft."""
        raise NotImplementedError("Requires model files")

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------

    def quantize(
        self,
        model_path: str,
        output_path: str,
        quant_type: str = "Q4_K_M",
    ) -> bool:
        """Quantize to GGUF via the ``llama-quantize`` binary."""
        result = subprocess.run(
            ["llama-quantize", model_path, output_path, quant_type],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return result.returncode == 0

    # ------------------------------------------------------------------
    # Ollama helpers
    # ------------------------------------------------------------------

    def build_modelfile(self, gguf_path: str, system_prompt: str = "") -> str:
        """Generate Ollama Modelfile content."""
        lines = [f"FROM {gguf_path}"]
        if system_prompt:
            lines.append(f'SYSTEM """{system_prompt}"""')
        return "\n".join(lines)

    def import_to_ollama(self, modelfile_path: str) -> bool:
        """Import as staging candidate: ``ollama create registry:candidate``."""
        result = subprocess.run(
            [
                "ollama",
                "create",
                f"{self.registry_name}:candidate",
                "-f",
                modelfile_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode == 0

    def promote_candidate(self) -> bool:
        """Atomic swap: ``ollama cp registry:candidate registry:latest``."""
        result = subprocess.run(
            [
                "ollama",
                "cp",
                f"{self.registry_name}:candidate",
                f"{self.registry_name}:latest",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode == 0
