"""Health probe for the voice pipeline."""

from .base import BaseProbe, HealthStatus, ProbeResult


class VoiceProbe(BaseProbe):
    """Checks STT, TTS, and VAD engine availability."""

    name = "voice"
    interval = 30.0

    def __init__(self, voice_manager=None) -> None:
        self._vm = voice_manager

    def check(self) -> ProbeResult:
        if self._vm is None:
            return ProbeResult(
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                error_count=0,
                last_error="Voice manager not initialized",
            )

        try:
            engines = self._vm.available_engines
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=str(exc),
            )

        stt_ok = engines.get("stt", False)
        tts_ok = engines.get("tts", False)
        vad_ok = engines.get("vad", False)

        metadata = {"stt": stt_ok, "tts": tts_ok, "vad": vad_ok}

        if not stt_ok and not tts_ok:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error="STT and TTS both unavailable",
                metadata=metadata,
            )

        if not stt_ok or not tts_ok or not vad_ok:
            missing = [k for k, v in engines.items() if not v]
            return ProbeResult(
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error_count=len(missing),
                last_error=f"Unavailable: {', '.join(missing)}",
                metadata=metadata,
            )

        return ProbeResult(
            status=HealthStatus.HEALTHY,
            latency_ms=0,
            error_count=0,
            metadata=metadata,
        )
