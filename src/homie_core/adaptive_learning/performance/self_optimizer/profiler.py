"""Optimization profiler — persists learned parameters per (query_type, hardware)."""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


def generate_hardware_fingerprint(gpu_name: str = "", vram_mb: int = 0, ram_gb: float = 0.0) -> str:
    """Generate a stable fingerprint for the current hardware."""
    raw = f"{gpu_name}:{vram_mb}:{ram_gb:.1f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class OptimizationProfile:
    """Learned optimization parameters for a query type on specific hardware."""

    query_type: str
    hardware_fingerprint: str
    temperature: float = 0.7
    max_tokens: int = 1024
    context_budget: int = 5000
    pipeline_tier: str = "moderate"
    avg_response_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    sample_count: int = 0

    def update(self, response_tokens: float, latency_ms: float, learning_rate: float = 0.1) -> None:
        """Update profile with new observation via EMA."""
        if self.sample_count == 0:
            self.avg_response_tokens = response_tokens
            self.avg_latency_ms = latency_ms
        else:
            self.avg_response_tokens = learning_rate * response_tokens + (1 - learning_rate) * self.avg_response_tokens
            self.avg_latency_ms = learning_rate * latency_ms + (1 - learning_rate) * self.avg_latency_ms
        self.sample_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "hardware_fingerprint": self.hardware_fingerprint,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_budget": self.context_budget,
            "pipeline_tier": self.pipeline_tier,
            "avg_response_tokens": self.avg_response_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class OptimizationProfiler:
    """Manages optimization profiles — save, load, update."""

    def __init__(self, storage, hardware_fingerprint: str) -> None:
        self._storage = storage
        self._hw_fp = hardware_fingerprint
        self._cache: dict[str, OptimizationProfile] = {}

    def get_profile(self, query_type: str) -> Optional[OptimizationProfile]:
        """Get profile for a query type on current hardware."""
        if query_type in self._cache:
            return self._cache[query_type]
        data = self._storage.get_optimization_profile(query_type, self._hw_fp)
        if data:
            profile = OptimizationProfile.from_dict(data)
            self._cache[query_type] = profile
            return profile
        return None

    def save_profile(self, profile: OptimizationProfile) -> None:
        """Save a profile to storage."""
        self._cache[profile.query_type] = profile
        self._storage.save_optimization_profile(profile.query_type, self._hw_fp, profile.to_dict())

    def record_observation(
        self,
        query_type: str,
        temperature_used: float,
        max_tokens_used: int,
        response_tokens: float,
        latency_ms: float,
    ) -> None:
        """Record an inference observation and update the profile."""
        profile = self.get_profile(query_type)
        if profile is None:
            profile = OptimizationProfile(
                query_type=query_type,
                hardware_fingerprint=self._hw_fp,
                temperature=temperature_used,
                max_tokens=max_tokens_used,
            )
        profile.update(response_tokens, latency_ms)
        self.save_profile(profile)
