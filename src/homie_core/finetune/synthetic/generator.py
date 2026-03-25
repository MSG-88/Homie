"""Synthetic data generator with teacher pipeline.

Orchestrates template selection, context randomization, teacher inference,
and quality filtering to produce SFT and DPO training examples.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

from homie_core.finetune.synthetic.context_randomizer import ContextRandomizer
from homie_core.finetune.synthetic.quality_filter import QualityFilter
from homie_core.finetune.synthetic.templates import (
    Domain,
    ScenarioTemplate,
    get_templates_for_domain,
)

logger = logging.getLogger(__name__)

# Base domain weights for allocation
_BASE_WEIGHTS: dict[Domain, float] = {
    Domain.INTENT: 0.25,
    Domain.CONTEXT: 0.20,
    Domain.CONVERSATIONAL: 0.20,
    Domain.ORCHESTRATION: 0.15,
    Domain.SELF_AWARENESS: 0.10,
    Domain.SAFETY: 0.10,
}

# Exponential backoff delays for teacher retries
_BACKOFF_DELAYS = (60, 300, 900)  # 1min, 5min, 15min

# Minimum rate-limit interval between cloud API calls (seconds)
_RATE_LIMIT_INTERVAL = 6.0


class SyntheticDataGenerator:
    """Generate synthetic SFT and DPO training data via a teacher model.

    Parameters
    ----------
    inference_fn:
        Called as ``inference_fn(prompt=..., max_tokens=..., temperature=...)``.
    seed:
        Seed for deterministic context randomization.
    min_quality:
        Minimum quality score (1-5) required to keep a generated example.
    rate_limit:
        Seconds to sleep between inference calls. Set to 0 to disable.
    """

    def __init__(
        self,
        inference_fn: Callable[..., str],
        seed: int = 42,
        min_quality: int = 4,
        rate_limit: float = _RATE_LIMIT_INTERVAL,
    ) -> None:
        self._inference_fn = inference_fn
        self._ctx = ContextRandomizer(seed)
        self._quality = QualityFilter(inference_fn, min_quality)
        self._min_quality = min_quality
        self._rate_limit = rate_limit
        self._last_call: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttled_inference(self, **kwargs: object) -> str:
        """Call inference with rate limiting."""
        if self._rate_limit > 0:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._rate_limit:
                time.sleep(self._rate_limit - elapsed)
        result = self._inference_fn(**kwargs)
        self._last_call = time.monotonic()
        return result

    def _call_teacher(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Call teacher with retry + exponential backoff.

        Tries up to 3 retries. On persistent failure raises RuntimeError.
        """
        last_err: Exception | None = None
        for attempt, delay in enumerate(_BACKOFF_DELAYS):
            try:
                return self._throttled_inference(
                    prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning(
                    "Teacher call failed (attempt %d/%d): %s",
                    attempt + 1, len(_BACKOFF_DELAYS), exc,
                )
                if attempt < len(_BACKOFF_DELAYS) - 1:
                    time.sleep(delay)

        raise RuntimeError(f"Teacher inference failed after retries: {last_err}")

    def _pick_template(
        self, templates: list[ScenarioTemplate],
    ) -> ScenarioTemplate:
        """Pick a random template from the list using the context randomizer's RNG."""
        idx = self._ctx._rng.randint(0, len(templates) - 1)
        return templates[idx]

    def _render_user_prompt(self, template: ScenarioTemplate, detail: str) -> str:
        """Render the user prompt, substituting {detail} if present."""
        try:
            return template.user_prompt_template.format(detail=detail)
        except (KeyError, IndexError):
            return template.user_prompt_template

    # ------------------------------------------------------------------
    # SFT generation
    # ------------------------------------------------------------------

    def generate_sft(
        self, count: int, domain: Domain, max_difficulty: int = 4,
    ) -> list[dict]:
        """Generate SFT examples for a domain.

        Each example: ``{"system": str, "user": str, "assistant": str}``
        """
        templates = get_templates_for_domain(domain, max_difficulty)
        if not templates:
            logger.warning("No templates found for domain=%s max_difficulty=%d", domain, max_difficulty)
            return []

        examples: list[dict] = []
        retries_left = count * 3  # safety cap to avoid infinite loops

        while len(examples) < count and retries_left > 0:
            retries_left -= 1
            template = self._pick_template(templates)
            ctx = self._ctx.generate()
            system_prompt = ctx.to_system_prompt()
            detail = ctx.to_detail()
            user_msg = self._render_user_prompt(template, detail)

            teacher_prompt = (
                f"You are a teacher model generating an ideal assistant response.\n\n"
                f"System context: {template.system_context}\n"
                f"Good behavior: {template.good_behavior}\n"
                f"User message: {user_msg}\n\n"
                f"Write the assistant's ideal response. Be concise and helpful."
            )

            try:
                response = self._call_teacher(teacher_prompt)
            except RuntimeError:
                logger.warning("Skipping example after teacher failure for template=%s", template.name)
                continue

            if self._quality.passes(system_prompt, user_msg, response):
                examples.append({
                    "system": system_prompt,
                    "user": user_msg,
                    "assistant": response,
                })

        return examples

    # ------------------------------------------------------------------
    # DPO generation
    # ------------------------------------------------------------------

    def generate_dpo(
        self, count: int, domain: Domain, max_difficulty: int = 4,
    ) -> list[dict]:
        """Generate DPO pairs for a domain.

        Each pair: ``{"system": str, "user": str, "chosen": str, "rejected": str}``
        """
        templates = get_templates_for_domain(domain, max_difficulty)
        if not templates:
            logger.warning("No templates found for domain=%s max_difficulty=%d", domain, max_difficulty)
            return []

        pairs: list[dict] = []
        retries_left = count * 3

        while len(pairs) < count and retries_left > 0:
            retries_left -= 1
            template = self._pick_template(templates)
            ctx = self._ctx.generate()
            system_prompt = ctx.to_system_prompt()
            detail = ctx.to_detail()
            user_msg = self._render_user_prompt(template, detail)

            dpo_prompt = (
                f"You are a teacher model generating a chosen/rejected pair.\n\n"
                f"System context: {template.system_context}\n"
                f"Good behavior: {template.good_behavior}\n"
                f"Bad behavior: {template.bad_behavior}\n"
                f"User message: {user_msg}\n\n"
                f"Return a JSON object with \"chosen\" (ideal response) and "
                f"\"rejected\" (poor response) keys."
            )

            try:
                raw = self._call_teacher(dpo_prompt, max_tokens=1024)
                parsed = json.loads(raw)
                chosen = parsed["chosen"]
                rejected = parsed["rejected"]
            except (RuntimeError, json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Skipping DPO pair for template=%s", template.name)
                continue

            # Quality-check the chosen response only
            if self._quality.passes(system_prompt, user_msg, chosen):
                pairs.append({
                    "system": system_prompt,
                    "user": user_msg,
                    "chosen": chosen,
                    "rejected": rejected,
                })

        return pairs

    # ------------------------------------------------------------------
    # Domain allocation
    # ------------------------------------------------------------------

    def compute_allocation(
        self,
        total_sft: int,
        total_dpo: int,
        weak_domain: Domain | None = None,
        boost: float = 0.4,
    ) -> dict[Domain, dict]:
        """Compute per-domain allocation of SFT and DPO counts.

        Base weights: INTENT=25%, CONTEXT=20%, CONVERSATIONAL=20%,
        ORCHESTRATION=15%, SELF_AWARENESS=10%, SAFETY=10%.

        If *weak_domain* is given, boost it to *boost* fraction and
        proportionally reduce the other domains.
        """
        weights = dict(_BASE_WEIGHTS)

        if weak_domain is not None:
            remaining = 1.0 - boost
            old_weak = weights[weak_domain]
            scale = remaining / (1.0 - old_weak)
            for d in weights:
                if d == weak_domain:
                    weights[d] = boost
                else:
                    weights[d] *= scale

        # Convert weights to integer counts, ensuring totals match exactly
        alloc: dict[Domain, dict] = {}
        domains = list(Domain)

        for kind, total in [("sft", total_sft), ("dpo", total_dpo)]:
            raw = {d: weights[d] * total for d in domains}
            floored = {d: int(raw[d]) for d in domains}
            remainder = total - sum(floored.values())
            # Distribute remainder by fractional part descending
            fracs = sorted(domains, key=lambda d: raw[d] - floored[d], reverse=True)
            for i, d in enumerate(fracs):
                if i < remainder:
                    floored[d] += 1

            for d in domains:
                alloc.setdefault(d, {})[kind] = floored[d]

        return alloc

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------

    def generate_cycle(
        self,
        total_sft: int = 2000,
        total_dpo: int = 1000,
        weak_domain: Domain | None = None,
        boost: float = 0.4,
        max_difficulty_per_domain: dict[Domain, int] | None = None,
    ) -> dict:
        """Generate a full cycle of training data.

        Returns ``{"sft": list[dict], "dpo": list[dict], "stats": dict}``.
        """
        alloc = self.compute_allocation(total_sft, total_dpo, weak_domain, boost)
        max_diff = max_difficulty_per_domain or {}

        all_sft: list[dict] = []
        all_dpo: list[dict] = []
        stats: dict[str, dict] = {}

        for domain in Domain:
            target_sft = alloc[domain]["sft"]
            target_dpo = alloc[domain]["dpo"]
            difficulty = max_diff.get(domain, 4)

            # SFT with retry if below 50% target
            sft_examples: list[dict] = []
            for attempt in range(3):
                sft_examples = self.generate_sft(target_sft, domain, difficulty)
                if len(sft_examples) >= target_sft * 0.5:
                    break
                logger.warning(
                    "Domain %s SFT below 50%% target (got %d/%d), retry %d/3",
                    domain.value, len(sft_examples), target_sft, attempt + 1,
                )

            # DPO with retry if below 50% target
            dpo_pairs: list[dict] = []
            for attempt in range(3):
                dpo_pairs = self.generate_dpo(target_dpo, domain, difficulty)
                if len(dpo_pairs) >= target_dpo * 0.5:
                    break
                logger.warning(
                    "Domain %s DPO below 50%% target (got %d/%d), retry %d/3",
                    domain.value, len(dpo_pairs), target_dpo, attempt + 1,
                )

            all_sft.extend(sft_examples)
            all_dpo.extend(dpo_pairs)
            stats[domain.value] = {
                "sft_target": target_sft,
                "sft_generated": len(sft_examples),
                "dpo_target": target_dpo,
                "dpo_generated": len(dpo_pairs),
            }

        return {"sft": all_sft, "dpo": all_dpo, "stats": stats}

    # ------------------------------------------------------------------
    # Export / format utilities
    # ------------------------------------------------------------------

    @staticmethod
    def export_jsonl(examples: list[dict], path: Path) -> None:
        """Write list of dicts as JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    @staticmethod
    def format_adapter(example: dict, from_format: str = "alpaca") -> dict:
        """Convert between training data formats.

        Currently supports:
        - ``"alpaca"`` → ChatML: ``{instruction, input, output}`` →
          ``{system, user, assistant}``
        """
        if from_format == "alpaca":
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            user = f"{instruction}\n{inp}" if inp else instruction
            return {
                "system": "",
                "user": user,
                "assistant": example.get("output", ""),
            }
        raise ValueError(f"Unsupported format: {from_format}")
