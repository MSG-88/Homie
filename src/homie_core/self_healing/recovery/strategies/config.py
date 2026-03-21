"""Recovery strategies for configuration failures."""

import logging
from pathlib import Path

import yaml

from ..engine import RecoveryResult, RecoveryTier

logger = logging.getLogger(__name__)


def reparse_config(module, status, error, config_path=None, **ctx) -> RecoveryResult:
    """T1: Re-read and validate the config file."""
    if not config_path or not Path(config_path).exists():
        return RecoveryResult(success=False, action="config file not found", tier=RecoveryTier.RETRY)
    try:
        with open(config_path) as f:
            yaml.safe_load(f)
        return RecoveryResult(success=True, action="config re-parsed successfully", tier=RecoveryTier.RETRY)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"reparse failed: {exc}", tier=RecoveryTier.RETRY)


def use_last_known_good(module, status, error, config_cache=None, **ctx) -> RecoveryResult:
    """T2: Fall back to cached known-good config."""
    if config_cache is None:
        return RecoveryResult(success=False, action="no cached config", tier=RecoveryTier.FALLBACK)
    logger.info("Using last known good configuration")
    return RecoveryResult(success=True, action="reverted to cached config", tier=RecoveryTier.FALLBACK)


def reset_to_defaults(module, status, error, config=None, **ctx) -> RecoveryResult:
    """T4: Reset config to factory defaults."""
    logger.warning("Config reset to defaults")
    return RecoveryResult(
        success=True,
        action="config reset to defaults",
        tier=RecoveryTier.DEGRADE,
        details={"mode": "defaults"},
    )
