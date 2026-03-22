import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.performance.self_optimizer.profiler import (
    OptimizationProfiler,
    OptimizationProfile,
    generate_hardware_fingerprint,
)


class TestOptimizationProfile:
    def test_defaults(self):
        p = OptimizationProfile(query_type="coding", hardware_fingerprint="abc123")
        assert p.temperature == 0.7
        assert p.max_tokens == 1024
        assert p.sample_count == 0

    def test_to_dict_from_dict(self):
        p = OptimizationProfile(query_type="chat", hardware_fingerprint="xyz", temperature=0.3, max_tokens=256)
        d = p.to_dict()
        p2 = OptimizationProfile.from_dict(d)
        assert p2.temperature == 0.3
        assert p2.max_tokens == 256

    def test_update_with_ema(self):
        p = OptimizationProfile(query_type="coding", hardware_fingerprint="abc", avg_response_tokens=100, sample_count=10)
        p.update(response_tokens=200, latency_ms=500, learning_rate=0.2)
        assert p.avg_response_tokens > 100  # moved toward 200
        assert p.avg_latency_ms > 0
        assert p.sample_count == 11


class TestHardwareFingerprint:
    def test_generates_string(self):
        fp = generate_hardware_fingerprint(gpu_name="RTX 4090", vram_mb=24576, ram_gb=32.0)
        assert isinstance(fp, str)
        assert len(fp) > 0

    def test_same_hardware_same_fingerprint(self):
        fp1 = generate_hardware_fingerprint(gpu_name="RTX 4090", vram_mb=24576, ram_gb=32.0)
        fp2 = generate_hardware_fingerprint(gpu_name="RTX 4090", vram_mb=24576, ram_gb=32.0)
        assert fp1 == fp2

    def test_different_hardware_different_fingerprint(self):
        fp1 = generate_hardware_fingerprint(gpu_name="RTX 4090", vram_mb=24576, ram_gb=32.0)
        fp2 = generate_hardware_fingerprint(gpu_name="RTX 3060", vram_mb=12288, ram_gb=16.0)
        assert fp1 != fp2


class TestOptimizationProfiler:
    def test_get_profile_returns_none_for_unknown(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        profiler = OptimizationProfiler(storage=storage, hardware_fingerprint="abc")
        assert profiler.get_profile("unknown_type") is None

    def test_save_and_get_profile(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        profiler = OptimizationProfiler(storage=storage, hardware_fingerprint="abc")
        profile = OptimizationProfile(query_type="coding", hardware_fingerprint="abc", temperature=0.4)
        profiler.save_profile(profile)
        storage.save_optimization_profile.assert_called_once()

    def test_record_observation_creates_profile(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        profiler = OptimizationProfiler(storage=storage, hardware_fingerprint="abc")
        profiler.record_observation("coding", temperature_used=0.5, max_tokens_used=512, response_tokens=200, latency_ms=300)
        storage.save_optimization_profile.assert_called()
