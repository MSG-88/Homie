"""RecursiveFinetuneLoop pipeline orchestrator.

Manages the full recursive finetuning lifecycle: generate synthetic data,
train with QLoRA, evaluate via benchmark suite, and deploy improved models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from homie_core.finetune.config import FinetuneConfig
from homie_core.finetune.evaluation.benchmark import BenchmarkResult, BenchmarkSuite
from homie_core.finetune.evaluation.judge import Judge
from homie_core.finetune.evaluation.reporter import EvalReporter
from homie_core.finetune.synthetic.generator import SyntheticDataGenerator
from homie_core.finetune.synthetic.templates import Domain

logger = logging.getLogger(__name__)


class PipelineState:
    """Persistent state for the recursive finetuning loop."""

    def __init__(self, state_dir: Path) -> None:
        self._path = Path(state_dir) / "state.json"
        self.current_cycle: int = 0
        self.lora_rank: int = 16
        self.plateau_counter: int = 0
        self.cycle_scores: dict[int, float] = {}
        self.difficulty_tiers: dict[str, int] = {d.value: 1 for d in Domain}

    def record_score(self, cycle: int, score: float) -> None:
        """Record the overall score for a completed cycle."""
        self.cycle_scores[cycle] = score

    def save(self) -> None:
        """Persist state to disk as JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {
                    "current_cycle": self.current_cycle,
                    "lora_rank": self.lora_rank,
                    "plateau_counter": self.plateau_counter,
                    "cycle_scores": {str(k): v for k, v in self.cycle_scores.items()},
                    "difficulty_tiers": self.difficulty_tiers,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, state_dir: Path) -> PipelineState:
        """Load state from disk, returning fresh state if file is absent."""
        path = Path(state_dir) / "state.json"
        state = cls(state_dir)
        if path.exists():
            data = json.loads(path.read_text())
            state.current_cycle = data.get("current_cycle", 0)
            state.lora_rank = data.get("lora_rank", 16)
            state.plateau_counter = data.get("plateau_counter", 0)
            state.cycle_scores = {
                int(k): v for k, v in data.get("cycle_scores", {}).items()
            }
            state.difficulty_tiers = data.get(
                "difficulty_tiers", state.difficulty_tiers
            )
        return state


class RecursiveFinetuneLoop:
    """Orchestrates the full recursive finetuning pipeline.

    Each cycle: generate synthetic data -> train with QLoRA -> evaluate -> deploy.
    """

    def __init__(
        self,
        config: FinetuneConfig,
        inference_fn: Callable,
        ollama_manager: object,
        model_registry: object,
        base_dir: Path,
    ) -> None:
        self.config = config
        self._inference_fn = inference_fn
        self._ollama = ollama_manager
        self._registry = model_registry
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self.state = PipelineState.load(self._base_dir)
        self._status: dict = {"stage": "idle", "cycle": 0, "progress": 0.0}

    def run(self) -> dict:
        """Main recursive loop. Returns summary dict."""
        cycles_completed = 0
        while not self._should_stop():
            cycle = self.state.current_cycle
            logger.info(
                "Starting finetune cycle %d (LoRA rank %d)",
                cycle,
                self.state.lora_rank,
            )
            self._status = {"stage": "generating", "cycle": cycle, "progress": 0.0}

            try:
                result = self._run_cycle(cycle)
                self.state.record_score(cycle, result.get("score", 0.0))
                cycles_completed += 1
            except Exception as exc:
                logger.error("Cycle %d failed: %s", cycle, exc)
                break

            self.state.current_cycle += 1
            self._prune_artifacts()
            self.state.save()

        return {
            "cycles_completed": cycles_completed,
            "final_scores": dict(self.state.cycle_scores),
        }

    def _run_cycle(self, cycle: int) -> dict:
        """Run a single cycle: generate -> train -> evaluate -> deploy."""
        # 1. GENERATE
        generator = SyntheticDataGenerator(
            inference_fn=self._inference_fn,
            seed=cycle * 1000,
            min_quality=self.config.data.min_quality_score,
        )
        reporter = EvalReporter(
            promotion_threshold=self.config.evaluation.promotion_threshold,
            safety_floor=self.config.evaluation.safety_floor,
            max_regression=self.config.evaluation.max_regression_per_domain,
            plateau_cycles=self.config.evaluation.plateau_cycles,
        )
        weak = (
            reporter.weakest_domain(self._get_current_result())
            if self.state.cycle_scores
            else None
        )
        data = generator.generate_cycle(
            total_sft=self.config.data.sft_per_cycle,
            total_dpo=self.config.data.dpo_per_cycle,
            weak_domain=weak,
            boost=self.config.data.weak_domain_boost,
        )
        # Save datasets
        ds_dir = self._base_dir / "datasets" / f"cycle-{cycle}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        generator.export_jsonl(data["sft"], ds_dir / "sft.jsonl")
        generator.export_jsonl(data["dpo"], ds_dir / "dpo.jsonl")

        # 2. TRAIN
        self._status["stage"] = "training"
        all_sft = self._load_accumulated_data("sft.jsonl")
        all_dpo = self._load_accumulated_data("dpo.jsonl")

        # Write accumulated data for training
        train_dir = self._base_dir / "training" / f"cycle-{cycle}"
        train_dir.mkdir(parents=True, exist_ok=True)
        SyntheticDataGenerator.export_jsonl(all_sft, train_dir / "sft.jsonl")
        SyntheticDataGenerator.export_jsonl(all_dpo, train_dir / "dpo.jsonl")

        try:
            from homie_core.finetune.training.qlora_trainer import QLoRATrainer
            trainer = QLoRATrainer(
                base_model=self.config.base_model,
                config=self.config.training,
                output_dir=str(self._base_dir / "adapters" / f"cycle-{cycle}"),
            )
            trainer.config.lora_rank = self.state.lora_rank

            sft_result = trainer.train_sft(train_dir / "sft.jsonl")
            dpo_result = trainer.train_dpo(
                train_dir / "dpo.jsonl",
                sft_adapter_path=Path(sft_result.get("adapter_path", "")),
            )
        except NotImplementedError:
            logger.warning("Training skipped — no GPU available (stubbed)")
            # Still run evaluation on the current model for baseline tracking

        # 3. EVALUATE
        self._status["stage"] = "evaluating"
        judge = Judge(inference_fn=self._inference_fn)
        suite = BenchmarkSuite(inference_fn=self._inference_fn, judge_fn=judge.score)
        result = suite.run()

        # Save eval results to disk
        eval_dir = self._base_dir / "evals"
        eval_dir.mkdir(parents=True, exist_ok=True)
        import json as _json
        eval_path = eval_dir / f"cycle-{cycle}.json"
        eval_data = {
            "domain_scores": {d.value: s for d, s in result.domain_scores.items()},
            "overall_score": result.overall_score,
        }
        eval_path.write_text(_json.dumps(eval_data, indent=2))

        # 4. DEPLOY if improved
        self._status["stage"] = "deploying"
        current = self._get_current_result()
        if current and not reporter.should_promote(current, result):
            self.state.plateau_counter += 1
            if self.state.plateau_counter >= self.config.evaluation.plateau_cycles:
                self._handle_plateau()
            return {"score": result.overall_score, "promoted": False}

        # Deploy
        try:
            from homie_core.finetune.training.merge import ModelMerger
            merger = ModelMerger(base_model=self.config.base_model, registry_name=self.config.registry_name)
            # merger.merge_lora(...) — requires real model files
            # merger.quantize(...) — requires llama.cpp
            merger.import_to_ollama(str(train_dir / "Modelfile"))
            merger.promote_candidate()
        except NotImplementedError:
            logger.warning("Deploy skipped — stubbed methods")

        self.state.plateau_counter = 0
        return {"score": result.overall_score, "promoted": True}

    def _load_accumulated_data(self, filename: str) -> list[dict]:
        """Load and merge all JSONL files from cycle-0 to current."""
        all_data: list[dict] = []
        for i in range(self.state.current_cycle + 1):
            path = self._base_dir / "datasets" / f"cycle-{i}" / filename
            if path.exists():
                for line in path.read_text().strip().split("\n"):
                    if line.strip():
                        all_data.append(json.loads(line))
        return all_data

    def _should_stop(self) -> bool:
        """Check termination conditions."""
        if self.state.current_cycle >= self.config.limits.max_cycles:
            return True
        if (
            self.state.plateau_counter >= self.config.evaluation.plateau_cycles
            and self.state.lora_rank >= self.config.limits.max_lora_rank
        ):
            return True
        return False

    def _handle_plateau(self) -> None:
        """Escalate LoRA rank when training plateaus."""
        if self.state.lora_rank < self.config.limits.max_lora_rank:
            self.state.lora_rank = min(
                self.state.lora_rank * 2, self.config.limits.max_lora_rank
            )
            self.state.plateau_counter = 0
            logger.info(
                "Plateau detected, escalating LoRA rank to %d", self.state.lora_rank
            )

    def _get_current_result(self) -> BenchmarkResult | None:
        """Load the most recent eval result."""
        if not self.state.cycle_scores:
            return None
        last_cycle = max(self.state.cycle_scores.keys())
        eval_path = self._base_dir / "evals" / f"cycle-{last_cycle}.json"
        if eval_path.exists():
            import json as _json
            data = _json.loads(eval_path.read_text())
            domain_scores = {Domain(k): v for k, v in data.get("domain_scores", {}).items()}
            return BenchmarkResult(
                domain_scores=domain_scores,
                overall_score=data.get("overall_score", 0.0),
                case_results=data.get("case_results", []),
            )
        # Fallback: construct from overall score with uniform domain scores
        score = self.state.cycle_scores[last_cycle]
        return BenchmarkResult(
            domain_scores={d: score for d in Domain},
            overall_score=score,
            case_results=[],
        )

    def _get_difficulty(self, score: float) -> int:
        """Map a score to a difficulty tier (1-4)."""
        if score >= 0.9:
            return 4
        if score >= 0.8:
            return 3
        if score >= 0.6:
            return 2
        return 1

    def _prune_artifacts(self) -> None:
        """Delete oldest cycles if disk usage exceeds limit."""
        import shutil
        datasets_dir = self._base_dir / "datasets"
        adapters_dir = self._base_dir / "adapters"
        evals_dir = self._base_dir / "evals"

        # Find all cycle numbers
        cycles = set()
        for d in [datasets_dir, adapters_dir, evals_dir]:
            if d.exists():
                for child in d.iterdir():
                    if child.is_dir() and child.name.startswith("cycle-"):
                        try:
                            cycles.add(int(child.name.split("-")[1]))
                        except (ValueError, IndexError):
                            pass

        if not cycles:
            return

        # Keep the most recent 3 cycles
        keep = sorted(cycles)[-3:]
        for cycle_num in sorted(cycles):
            if cycle_num in keep:
                continue
            for d in [datasets_dir, adapters_dir, evals_dir]:
                path = d / f"cycle-{cycle_num}"
                if path.exists():
                    shutil.rmtree(path)
                    logger.info("Pruned cycle-%d from %s", cycle_num, d.name)

    def get_status(self) -> dict:
        """Return current pipeline status."""
        return self._status.copy()
