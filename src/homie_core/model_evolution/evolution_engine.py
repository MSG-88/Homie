"""Evolution engine — coordinates the full model evolution pipeline."""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from .milestone_tracker import MilestoneTracker
from .model_registry import ModelRegistry
from .modelfile_builder import ModelfileBuilder
from .ollama_manager import OllamaManager
from .validator import ModelValidator

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """Coordinates model evolution: build -> validate -> push."""

    def __init__(
        self,
        storage,
        ollama_manager: OllamaManager,
        preference_engine,
        knowledge_query,
        customization_manager,
        profiler,
        inference_fn: Callable[[str], str],
        base_model: str = "lfm2",
        registry_name: str = "PyMasters/Homie",
        user_name: str = "Master",
        modelfile_dir: str | Path = "",
        benchmark_threshold: float = 0.7,
        min_facts: int = 50,
        min_prefs: int = 10,
        min_customs: int = 3,
    ) -> None:
        self._storage = storage
        self._ollama = ollama_manager
        self._pref = preference_engine
        self._kg = knowledge_query
        self._customs = customization_manager
        self._profiler = profiler
        self._infer = inference_fn
        self._base_model = base_model
        self._registry_name = registry_name
        self._user_name = user_name
        self._modelfile_dir = Path(modelfile_dir) if modelfile_dir else Path.home() / ".homie" / "model_evolution"

        self._registry = ModelRegistry(storage=storage)
        self._milestone = MilestoneTracker(min_facts=min_facts, min_prefs=min_prefs, min_customs=min_customs)
        self._validator = ModelValidator(inference_fn=inference_fn, benchmark_threshold=benchmark_threshold)
        self._last_hash: Optional[str] = None

    def should_evolve(self) -> bool:
        """Check if model should be rebuilt."""
        return self._milestone.should_rebuild()

    def trigger_evolution(self) -> None:
        """Manually trigger model evolution."""
        self._milestone.trigger_manual()

    def record_new_fact(self) -> None:
        self._milestone.record_new_fact()

    def record_preference_change(self) -> None:
        self._milestone.record_preference_change()

    def record_new_customization(self) -> None:
        self._milestone.record_new_customization()

    def build_modelfile(self) -> ModelfileBuilder:
        """Build the Modelfile from current learned state."""
        builder = ModelfileBuilder(base_model=self._base_model, user_name=self._user_name)

        # Preferences layer
        try:
            profile = self._pref.get_active_profile()
            verb = "concise" if profile.verbosity < 0.4 else "detailed" if profile.verbosity > 0.7 else ""
            form = "casual" if profile.formality < 0.4 else "formal" if profile.formality > 0.7 else ""
            depth = "expert" if profile.technical_depth > 0.7 else "simple" if profile.technical_depth < 0.3 else ""
            builder.set_preferences(verbosity=verb, formality=form, depth=depth, format_pref=profile.format_preference)
        except Exception:
            pass

        # Customizations layer
        try:
            customs = self._customs.list_customizations()
            active = [c["request_text"] for c in customs if c.get("status") == "active"]
            if active:
                builder.set_customizations(active)
        except Exception:
            pass

        # Parameters from profiler
        try:
            profile = self._profiler.get_profile("general")
            if profile and profile.sample_count > 5:
                builder.set_parameters(temperature=round(profile.temperature, 2))
        except Exception:
            pass

        return builder

    def evolve(self) -> dict[str, Any]:
        """Run the evolution pipeline. Returns status dict."""
        if not self.should_evolve():
            return {"status": "no_changes"}

        # Build Modelfile
        builder = self.build_modelfile()
        new_hash = builder.content_hash()

        # Check if content actually changed
        if new_hash == self._last_hash:
            self._milestone.reset()
            return {"status": "no_changes", "reason": "modelfile unchanged"}

        # Write Modelfile
        self._modelfile_dir.mkdir(parents=True, exist_ok=True)
        modelfile_path = self._modelfile_dir / "Modelfile"
        builder.write(modelfile_path)

        # Create model via Ollama
        if not self._ollama.create(self._registry_name, modelfile_path):
            return {"status": "create_failed"}

        # Register version
        version = self._registry.register(
            self._base_model, self._registry_name, new_hash,
            changelog=f"Milestone rebuild (hash: {new_hash})",
        )

        # Run benchmark
        benchmark = self._validator.run_benchmark()
        if not benchmark.passed:
            return {
                "status": "benchmark_failed",
                "version": version.version_id,
                "scores": benchmark.scores,
            }

        # Promote (skip shadow test for now — can be wired later)
        self._registry.promote(version.version_id)
        self._last_hash = new_hash
        self._milestone.reset()

        logger.info("Model evolved to %s (hash: %s)", version.version_id, new_hash)
        return {
            "status": "promoted",
            "version": version.version_id,
            "benchmark_scores": benchmark.scores,
        }

    def evolve_finetune(self, config: "FinetuneConfig | None" = None, base_dir: str | Path | None = None) -> dict:
        """Run recursive finetuning pipeline (long-running, call from scheduler)."""
        from homie_core.finetune.pipeline import RecursiveFinetuneLoop
        from homie_core.finetune.config import FinetuneConfig

        cfg = config or FinetuneConfig()
        if base_dir is None:
            base_dir = Path.home() / ".homie" / "finetune"
        loop = RecursiveFinetuneLoop(
            config=cfg,
            inference_fn=self._infer,
            ollama_manager=self._ollama,
            model_registry=self._registry,
            base_dir=Path(base_dir),
        )
        return loop.run()
