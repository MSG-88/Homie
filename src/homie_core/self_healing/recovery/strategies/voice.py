"""Recovery strategies for voice pipeline failures."""

import logging
from ..engine import RecoveryResult, RecoveryTier

logger = logging.getLogger(__name__)


def restart_voice_engine(module, status, error, voice_manager=None, **ctx) -> RecoveryResult:
    """T1: Stop and restart the voice manager."""
    if voice_manager is None:
        return RecoveryResult(success=False, action="no voice manager", tier=RecoveryTier.RETRY)
    try:
        voice_manager.stop()
        voice_manager.start()
        return RecoveryResult(success=True, action="voice engine restarted", tier=RecoveryTier.RETRY)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"restart failed: {exc}", tier=RecoveryTier.RETRY)


def switch_tts_engine(module, status, error, voice_manager=None, **ctx) -> RecoveryResult:
    """T2: Switch to a different TTS engine."""
    if voice_manager is None:
        return RecoveryResult(success=False, action="no voice manager", tier=RecoveryTier.FALLBACK)
    try:
        # Voice manager handles engine fallback internally
        voice_manager.stop()
        voice_manager.start()
        logger.info("Switched TTS engine via restart")
        return RecoveryResult(success=True, action="TTS engine switched", tier=RecoveryTier.FALLBACK)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"TTS switch failed: {exc}", tier=RecoveryTier.FALLBACK)


def degrade_to_text_only(module, status, error, voice_manager=None, **ctx) -> RecoveryResult:
    """T4: Disable voice entirely, text-only mode."""
    if voice_manager:
        try:
            voice_manager.stop()
        except Exception:
            pass
    logger.warning("Voice pipeline degraded — text-only mode active")
    return RecoveryResult(
        success=True,
        action="degraded to text-only mode",
        tier=RecoveryTier.DEGRADE,
        details={"mode": "text_only"},
    )
