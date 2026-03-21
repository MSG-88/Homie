"""Recovery strategies for inference failures."""

import logging
from ..engine import RecoveryResult, RecoveryTier

logger = logging.getLogger(__name__)


def retry_with_reduced_tokens(module, status, error, model_engine=None, **ctx) -> RecoveryResult:
    """T1: Retry inference with shorter max_tokens."""
    if model_engine is None:
        return RecoveryResult(success=False, action="no model engine", tier=RecoveryTier.RETRY)
    try:
        model_engine.generate("ping", max_tokens=1, timeout=10)
        return RecoveryResult(success=True, action="retry with reduced max_tokens succeeded", tier=RecoveryTier.RETRY)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"retry failed: {exc}", tier=RecoveryTier.RETRY)


def fallback_reduce_context(module, status, error, config=None, model_engine=None, **ctx) -> RecoveryResult:
    """T2: Reduce context_length and retry."""
    if config is None or model_engine is None:
        return RecoveryResult(success=False, action="missing config or engine", tier=RecoveryTier.FALLBACK)
    try:
        original = config.llm.context_length
        reduced = max(original // 2, 2048)
        config.llm.context_length = reduced
        model_engine.generate("ping", max_tokens=1, timeout=15)
        logger.info("Reduced context_length from %d to %d", original, reduced)
        return RecoveryResult(
            success=True,
            action=f"reduced context_length {original} → {reduced}",
            tier=RecoveryTier.FALLBACK,
        )
    except Exception as exc:
        return RecoveryResult(success=False, action=f"fallback failed: {exc}", tier=RecoveryTier.FALLBACK)


def switch_to_smaller_model(module, status, error, model_engine=None, model_registry=None, **ctx) -> RecoveryResult:
    """T3: Switch to a smaller/alternative model."""
    if model_engine is None or model_registry is None:
        return RecoveryResult(success=False, action="missing engine or registry", tier=RecoveryTier.REBUILD)
    try:
        models = model_registry.list_models()
        if not models:
            return RecoveryResult(success=False, action="no alternative models available", tier=RecoveryTier.REBUILD)
        # Pick first available alternative
        alt = models[0]
        model_engine.unload()
        model_engine.load(alt)
        logger.info("Switched to alternative model: %s", alt.name)
        return RecoveryResult(success=True, action=f"switched to {alt.name}", tier=RecoveryTier.REBUILD)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"model switch failed: {exc}", tier=RecoveryTier.REBUILD)


def degrade_to_cached(module, status, error, **ctx) -> RecoveryResult:
    """T4: Degrade to cached/static responses."""
    logger.warning("Inference fully degraded — serving cached responses only")
    return RecoveryResult(
        success=True,
        action="degraded to cached response mode",
        tier=RecoveryTier.DEGRADE,
        details={"mode": "cached_only"},
    )
