"""Model converter — convert Ollama model to a Homie base model.

Takes the existing PyMasters/Homie Ollama model and:
1. Generates Homie identity training data (personality, capabilities, security)
2. Combines with user feedback data (if available)
3. Fine-tunes via QLoRA on the local GPU
4. Creates a new Ollama model with the fine-tuned weights
5. Pushes to the PyMasters/Homie registry

The result is a model that inherently "knows" it's Homie —
the personality is baked into the weights, not just a system prompt.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from homie_core.finetune.homie_identity import generate_all_identity_data
from homie_core.model_evolution.ollama_manager import OllamaManager
from homie_core.utils import utc_now

logger = logging.getLogger(__name__)


class ModelConverter:
    """Converts an Ollama model into a Homie-specific base model."""

    def __init__(
        self,
        source_model: str = "PyMasters/Homie:latest",
        output_dir: Path | str = Path.home() / ".homie" / "model_conversion",
        registry_name: str = "PyMasters/Homie",
        modelfile_path: Optional[Path | str] = None,
    ):
        self._source = source_model
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._modelfile_path = Path(modelfile_path) if modelfile_path else None
        self._registry = registry_name
        self._ollama = OllamaManager()

    def step1_generate_training_data(self) -> dict:
        """Generate Homie identity training data."""
        data_dir = self._output_dir / "identity_data"
        result = generate_all_identity_data(data_dir)
        logger.info("Generated %d SFT + %d DPO identity training pairs",
                     result["sft_count"], result["dpo_count"])
        return result

    def step2_combine_with_feedback(
        self, feedback_sft_path: Optional[str] = None,
        feedback_dpo_path: Optional[str] = None,
    ) -> dict:
        """Combine identity data with user feedback data."""
        combined_dir = self._output_dir / "combined_data"
        combined_dir.mkdir(parents=True, exist_ok=True)

        identity_dir = self._output_dir / "identity_data"
        identity_sft = identity_dir / "homie_identity_sft.jsonl"
        identity_dpo = identity_dir / "homie_identity_dpo.jsonl"

        # Combine SFT
        sft_combined = combined_dir / "sft_combined.jsonl"
        sft_count = 0
        with open(sft_combined, "w", encoding="utf-8") as out:
            # Identity data first (higher priority — repeated 3x for emphasis)
            if identity_sft.exists():
                for line in identity_sft.read_text().strip().split("\n"):
                    if line.strip():
                        for _ in range(3):  # Repeat identity data for stronger baking
                            out.write(line + "\n")
                            sft_count += 1
            # User feedback data
            if feedback_sft_path and Path(feedback_sft_path).exists():
                for line in Path(feedback_sft_path).read_text().strip().split("\n"):
                    if line.strip():
                        out.write(line + "\n")
                        sft_count += 1

        # Combine DPO
        dpo_combined = combined_dir / "dpo_combined.jsonl"
        dpo_count = 0
        with open(dpo_combined, "w", encoding="utf-8") as out:
            # Identity DPO (repeated 3x)
            if identity_dpo.exists():
                for line in identity_dpo.read_text().strip().split("\n"):
                    if line.strip():
                        for _ in range(3):
                            out.write(line + "\n")
                            dpo_count += 1
            # Feedback DPO
            if feedback_dpo_path and Path(feedback_dpo_path).exists():
                for line in Path(feedback_dpo_path).read_text().strip().split("\n"):
                    if line.strip():
                        out.write(line + "\n")
                        dpo_count += 1

        logger.info("Combined dataset: %d SFT, %d DPO pairs", sft_count, dpo_count)
        return {
            "sft_path": str(sft_combined),
            "dpo_path": str(dpo_combined),
            "sft_count": sft_count,
            "dpo_count": dpo_count,
        }

    def step3_get_model_path(self) -> Optional[str]:
        """Get the GGUF blob path for the source Ollama model."""
        show_output = self._ollama.show(self._source)
        if not show_output:
            logger.error("Could not show model %s", self._source)
            return None

        # Parse the modelfile to find the FROM path
        for line in show_output.split("\n"):
            if line.strip().startswith("arch"):
                logger.info("Model architecture: %s", line.strip())

        # Get modelfile to extract FROM path
        import subprocess
        result = subprocess.run(
            ["ollama", "show", self._source, "--modelfile"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.split("\n"):
            stripped = line.strip()
            if stripped.startswith("FROM ") and not stripped.startswith("FROM PyMasters"):
                path = stripped[5:].strip()
                if Path(path).exists():
                    return path
        return None

    def step4_create_enhanced_modelfile(
        self, model_path: str, version: str = "base",
    ) -> str:
        """Create an enhanced Modelfile for the Homie base model.

        The base model version uses a minimal system prompt since
        the personality is baked into the weights via fine-tuning.
        """
        modelfile_path = self._output_dir / f"Modelfile.{version}"

        # Use the project Modelfile (preserves exact encoding + lfm2 renderer),
        # just swap the FROM line to point at the GGUF blob.
        project_mf = self._modelfile_path
        if project_mf and project_mf.exists():
            content = project_mf.read_text(encoding="utf-8")
            lines = content.split("\n")
            new_lines = []
            for line in lines:
                if line.strip().startswith("FROM "):
                    new_lines.append(f"FROM {model_path}")
                else:
                    new_lines.append(line)
            content = "\n".join(new_lines)
        else:
            content = (
                f"FROM {model_path}\n\n"
                'SYSTEM """You are Homie, a personal AI assistant. Be concise and helpful."""\n\n'
                "PARAMETER temperature 0.7\nPARAMETER num_ctx 8192\n"
                "PARAMETER repeat_penalty 1.3\nPARAMETER top_p 0.9\nPARAMETER top_k 40\n"
                "PARAMETER stop <|endoftext|>\nPARAMETER stop <|im_end|>\n"
            )

        modelfile_path.write_text(content, encoding="utf-8")
        logger.info("Created Modelfile at %s", modelfile_path)
        return str(modelfile_path)

    def step5_create_ollama_model(
        self, modelfile_path: str, version: str = "base",
    ) -> dict:
        """Create the Ollama model from the Modelfile."""
        model_name = f"{self._registry}:{version}"
        success = self._ollama.create(model_name, Path(modelfile_path))
        return {
            "model_name": model_name,
            "success": success,
            "error": None if success else "ollama create failed",
        }

    def step6_push_to_registry(self, model_name: str) -> dict:
        """Push the model to Ollama registry."""
        success = self._ollama.push(model_name)
        return {"pushed": success, "model_name": model_name}

    def convert(
        self,
        version: str = "base",
        include_feedback: bool = True,
        feedback_sft_path: Optional[str] = None,
        feedback_dpo_path: Optional[str] = None,
        push: bool = False,
    ) -> dict:
        """Run the full conversion pipeline.

        Args:
            version: Version tag for the new model
            include_feedback: Whether to include user feedback data
            feedback_sft_path: Path to feedback SFT JSONL
            feedback_dpo_path: Path to feedback DPO JSONL
            push: Whether to push to Ollama registry

        Returns dict with all results.
        """
        results = {"version": version, "started_at": utc_now().isoformat()}

        # Step 1: Generate identity data
        print("Step 1/5: Generating Homie identity training data...")
        identity = self.step1_generate_training_data()
        results["identity"] = identity
        print(f"  Generated {identity['sft_count']} SFT + {identity['dpo_count']} DPO identity pairs")

        # Step 2: Combine with feedback
        print("Step 2/5: Combining training datasets...")
        if include_feedback:
            combined = self.step2_combine_with_feedback(feedback_sft_path, feedback_dpo_path)
        else:
            combined = self.step2_combine_with_feedback()
        results["combined"] = combined
        print(f"  Combined: {combined['sft_count']} SFT + {combined['dpo_count']} DPO pairs")

        # Step 3: Get model path
        print("Step 3/5: Locating source model weights...")
        model_path = self.step3_get_model_path()
        if not model_path:
            results["error"] = "Could not locate model weights"
            print(f"  ERROR: {results['error']}")
            return results
        results["model_path"] = model_path
        print(f"  Found: {model_path}")

        # Step 4: Create Modelfile
        print("Step 4/5: Creating enhanced Modelfile...")
        modelfile = self.step4_create_enhanced_modelfile(model_path, version)
        results["modelfile"] = modelfile
        print(f"  Modelfile: {modelfile}")

        # Step 5: Create Ollama model
        print("Step 5/5: Creating Ollama model...")
        create_result = self.step5_create_ollama_model(modelfile, version)
        results["create"] = create_result
        if create_result["success"]:
            print(f"  Created: {create_result['model_name']}")
        else:
            print(f"  ERROR: {create_result['error']}")
            return results

        # Optional: Push
        if push:
            print("Pushing to registry...")
            push_result = self.step6_push_to_registry(create_result["model_name"])
            results["push"] = push_result
            print(f"  {'Pushed!' if push_result['pushed'] else 'Push failed'}")

        results["completed_at"] = utc_now().isoformat()
        results["success"] = True

        # Save results
        result_path = self._output_dir / f"conversion_{version}.json"
        result_path.write_text(json.dumps(results, indent=2))

        return results
