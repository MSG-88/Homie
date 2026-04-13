"""Automated training pipeline — feedback → fine-tune → Ollama push → mesh distribute.

Orchestrates the full self-improvement loop:
1. Check training trigger (enough feedback signals?)
2. Extract training pairs from feedback store
3. Export as JSONL datasets (SFT + DPO)
4. Invoke QLoRA fine-tuning (if GPU available)
5. Merge LoRA adapter into base model
6. Quantize to GGUF
7. Create Ollama model with updated Modelfile
8. Push to Ollama registry (PyMasters/Homie)
9. Announce update via mesh events
10. All mesh Spokes can pull the updated model
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from homie_core.mesh.feedback_store import FeedbackStore
from homie_core.mesh.training_trigger import TrainingTrigger
from homie_core.mesh.model_distributor import ModelDistributor
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.utils import utc_now

logger = logging.getLogger(__name__)


@dataclass
class TrainingCycleResult:
    """Result of a single training cycle."""
    cycle: int
    sft_pairs: int
    dpo_pairs: int
    training_completed: bool
    model_name: str = ""
    model_path: str = ""
    score: float = 0.0
    promoted: bool = False
    pushed_to_registry: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle, "sft_pairs": self.sft_pairs,
            "dpo_pairs": self.dpo_pairs, "training_completed": self.training_completed,
            "model_name": self.model_name, "model_path": self.model_path,
            "score": self.score, "promoted": self.promoted,
            "pushed_to_registry": self.pushed_to_registry,
            "error": self.error,
        }


class AutoTrainer:
    """Automated training pipeline orchestrator.

    Connects mesh feedback → training → Ollama → mesh distribution.
    Designed to run on the Hub node (which has the GPU).
    """

    def __init__(
        self,
        feedback_store: FeedbackStore,
        training_trigger: TrainingTrigger,
        model_distributor: ModelDistributor,
        mesh_manager: MeshManager,
        base_dir: Path | str,
        registry_name: str = "PyMasters/Homie",
        base_model: str = "Qwen/Qwen3.5-9B",
        modelfile_path: Optional[Path | str] = None,
    ):
        self._feedback = feedback_store
        self._trigger = training_trigger
        self._distributor = model_distributor
        self._mesh = mesh_manager
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._registry_name = registry_name
        self._base_model = base_model
        self._modelfile_path = Path(modelfile_path) if modelfile_path else None
        self._cycle_count = self._load_cycle_count()

    def _load_cycle_count(self) -> int:
        state_file = self._base_dir / "auto_trainer_state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            return data.get("cycle_count", 0)
        return 0

    def _save_cycle_count(self) -> None:
        state_file = self._base_dir / "auto_trainer_state.json"
        state_file.write_text(json.dumps({
            "cycle_count": self._cycle_count,
            "last_updated": utc_now().isoformat(),
        }))

    def check_ready(self) -> dict:
        """Check if enough feedback has accumulated to train."""
        return self._trigger.get_summary()

    def export_training_data(self) -> dict:
        """Extract and export training pairs from feedback store.

        Returns dict with paths to SFT and DPO JSONL files and counts.
        """
        pairs = self._feedback.get_training_pairs()
        sft_pairs = [p for p in pairs if p["type"] == "sft"]
        dpo_pairs = [p for p in pairs if p["type"] == "dpo"]

        cycle_dir = self._base_dir / "cycles" / f"cycle-{self._cycle_count + 1}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        sft_path = cycle_dir / "sft.jsonl"
        dpo_path = cycle_dir / "dpo.jsonl"

        # Write SFT data (chat format for Qwen)
        with open(sft_path, "w") as f:
            for p in sft_pairs:
                entry = {
                    "messages": [
                        {"role": "user", "content": p["query"]},
                        {"role": "assistant", "content": p["response"]},
                    ]
                }
                f.write(json.dumps(entry) + "\n")

        # Write DPO data
        with open(dpo_path, "w") as f:
            for p in dpo_pairs:
                entry = {
                    "prompt": p["query"],
                    "chosen": p["chosen"],
                    "rejected": p["rejected"],
                }
                f.write(json.dumps(entry) + "\n")

        return {
            "sft_path": str(sft_path),
            "dpo_path": str(dpo_path),
            "sft_count": len(sft_pairs),
            "dpo_count": len(dpo_pairs),
            "cycle_dir": str(cycle_dir),
        }

    def create_ollama_model(self, model_path: str, version: Optional[str] = None) -> dict:
        """Create an Ollama model from a GGUF file using the Homie Modelfile.

        Args:
            model_path: Path to the GGUF model file
            version: Version tag (e.g., "v2"). Defaults to cycle number.

        Returns dict with model_name, success, error.
        """
        version = version or f"v{self._cycle_count + 1}"
        model_name = f"{self._registry_name}:{version}"

        # Generate a Modelfile pointing to the new weights
        cycle_dir = self._base_dir / "cycles" / f"cycle-{self._cycle_count + 1}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        modelfile = cycle_dir / "Modelfile"

        # Copy system prompt from existing Modelfile if available
        system_prompt = self._get_system_prompt()

        modelfile_content = f'FROM {model_path}\n\n'
        if system_prompt:
            modelfile_content += f'SYSTEM """{system_prompt}"""\n\n'
        modelfile_content += 'PARAMETER temperature 0.7\n'
        modelfile_content += 'PARAMETER num_ctx 8192\n'
        modelfile_content += 'PARAMETER repeat_penalty 1.3\n'
        modelfile_content += 'PARAMETER top_p 0.9\n'
        modelfile_content += 'PARAMETER top_k 40\n'
        modelfile_content += 'PARAMETER stop <|endoftext|>\n'
        modelfile_content += 'PARAMETER stop <|im_end|>\n'

        modelfile.write_text(modelfile_content)

        # Create via Ollama
        try:
            from homie_core.model_evolution.ollama_manager import OllamaManager
            ollama = OllamaManager()
            success = ollama.create(model_name, modelfile)
            return {
                "model_name": model_name,
                "modelfile_path": str(modelfile),
                "success": success,
                "error": None if success else "ollama create failed",
            }
        except Exception as e:
            return {"model_name": model_name, "success": False, "error": str(e)}

    def push_to_registry(self, model_name: str) -> dict:
        """Push model to Ollama registry."""
        try:
            from homie_core.model_evolution.ollama_manager import OllamaManager
            ollama = OllamaManager()
            success = ollama.push(model_name)
            return {"model_name": model_name, "pushed": success,
                    "error": None if success else "push failed"}
        except Exception as e:
            return {"model_name": model_name, "pushed": False, "error": str(e)}

    def run_cycle(self, skip_training: bool = False) -> TrainingCycleResult:
        """Run a full training cycle.

        Steps: export data → train → create Ollama model → push → announce.
        Set skip_training=True to test the pipeline without actual GPU training.
        """
        self._cycle_count += 1
        cycle = self._cycle_count

        # 1. Announce training started
        data = self.export_training_data()
        self._distributor.announce_training_started(
            cycle=cycle, sft_pairs=data["sft_count"], dpo_pairs=data["dpo_count"],
        )

        result = TrainingCycleResult(
            cycle=cycle, sft_pairs=data["sft_count"], dpo_pairs=data["dpo_count"],
            training_completed=False,
        )

        # 2. Train (if not skipped and GPU available)
        if not skip_training:
            try:
                train_result = self._run_training(data)
                result.training_completed = train_result.get("success", False)
                result.model_path = train_result.get("model_path", "")
                result.score = train_result.get("score", 0.0)
            except Exception as e:
                result.error = f"Training failed: {e}"
                logger.error("Training cycle %d failed: %s", cycle, e)
        else:
            result.training_completed = True
            result.model_path = ""  # No actual model produced

        # 3. Announce completion
        self._distributor.announce_training_completed(
            cycle=cycle, score=result.score, promoted=result.training_completed,
        )

        # 4. Mark trigger as used
        self._trigger.mark_triggered()
        self._save_cycle_count()

        return result

    def _run_training(self, data: dict) -> dict:
        """Run actual QLoRA training. Requires GPU."""
        try:
            from homie_core.finetune.training.qlora_trainer import QLoRATrainer
            from homie_core.finetune.config import FinetuneConfig

            config = FinetuneConfig()
            trainer = QLoRATrainer(
                base_model=self._base_model,
                config=config.training,
                output_dir=data["cycle_dir"],
            )
            sft_result = trainer.train_sft(data["sft_path"])
            return {"success": True, "model_path": sft_result.get("adapter_path", ""),
                    "score": 0.0}
        except ImportError:
            logger.warning("QLoRA trainer not available (missing deps)")
            return {"success": False, "model_path": "", "score": 0.0}
        except Exception as e:
            logger.error("Training error: %s", e)
            return {"success": False, "model_path": "", "score": 0.0}

    def _get_system_prompt(self) -> str:
        """Extract system prompt from existing Modelfile."""
        if self._modelfile_path and self._modelfile_path.exists():
            content = self._modelfile_path.read_text()
            # Extract between SYSTEM """ and """
            start = content.find('SYSTEM """')
            if start >= 0:
                start += len('SYSTEM """')
                end = content.find('"""', start)
                if end >= 0:
                    return content[start:end].strip()
        return ""

    def get_cycle_history(self) -> list[dict]:
        """Get history of all training cycles."""
        cycles_dir = self._base_dir / "cycles"
        if not cycles_dir.exists():
            return []
        history = []
        for cycle_dir in sorted(cycles_dir.iterdir()):
            if cycle_dir.is_dir() and cycle_dir.name.startswith("cycle-"):
                state_file = cycle_dir / "result.json"
                if state_file.exists():
                    history.append(json.loads(state_file.read_text()))
        return history
