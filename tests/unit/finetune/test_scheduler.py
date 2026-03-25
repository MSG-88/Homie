"""Tests for the FinetuneScheduler."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from homie_core.finetune.config import ScheduleConfig
from homie_core.finetune.scheduler import FinetuneScheduler


class TestFinetuneScheduler:
    """FinetuneScheduler business-hours, idle detection, and scheduling tests."""

    def _make_scheduler(self, **overrides) -> FinetuneScheduler:
        return FinetuneScheduler(ScheduleConfig(**overrides))

    # ── is_business_hours ────────────────────────────────────────────

    def test_is_business_hours_weekday_daytime(self):
        sched = self._make_scheduler()
        # Monday 10am
        dt = datetime(2026, 3, 23, 10, 0)  # Monday
        assert sched.is_business_hours(dt) is True

    def test_is_not_business_hours_evening(self):
        sched = self._make_scheduler()
        # Monday 10pm
        dt = datetime(2026, 3, 23, 22, 0)  # Monday
        assert sched.is_business_hours(dt) is False

    def test_is_not_business_hours_weekend(self):
        sched = self._make_scheduler()
        # Saturday 10am
        dt = datetime(2026, 3, 28, 10, 0)  # Saturday
        assert sched.is_business_hours(dt) is False

    # ── is_system_idle ───────────────────────────────────────────────

    @patch("homie_core.finetune.scheduler._get_idle_minutes", return_value=45)
    @patch("homie_core.finetune.scheduler._get_gpu_vram_usage", return_value=10.0)
    @patch("homie_core.finetune.scheduler._get_cpu_usage", return_value=5.0)
    def test_is_system_idle(self, _cpu, _gpu, _idle):
        sched = self._make_scheduler()
        assert sched.is_system_idle() is True

    @patch("homie_core.finetune.scheduler._get_idle_minutes", return_value=45)
    @patch("homie_core.finetune.scheduler._get_gpu_vram_usage", return_value=10.0)
    @patch("homie_core.finetune.scheduler._get_cpu_usage", return_value=80.0)
    def test_not_idle_when_cpu_high(self, _cpu, _gpu, _idle):
        sched = self._make_scheduler()
        assert sched.is_system_idle() is False

    # ── can_start ────────────────────────────────────────────────────

    @patch("homie_core.finetune.scheduler._get_idle_minutes", return_value=0)
    @patch("homie_core.finetune.scheduler._get_gpu_vram_usage", return_value=50.0)
    @patch("homie_core.finetune.scheduler._get_cpu_usage", return_value=50.0)
    def test_can_start_outside_business_hours(self, _cpu, _gpu, _idle):
        sched = self._make_scheduler()
        # Saturday 10am — not business hours, so can_start even if not idle
        with patch.object(sched, "is_business_hours", return_value=False):
            assert sched.can_start() is True

    @patch("homie_core.finetune.scheduler._get_idle_minutes", return_value=45)
    @patch("homie_core.finetune.scheduler._get_gpu_vram_usage", return_value=10.0)
    @patch("homie_core.finetune.scheduler._get_cpu_usage", return_value=5.0)
    def test_can_start_during_business_hours_if_idle(self, _cpu, _gpu, _idle):
        sched = self._make_scheduler()
        with patch.object(sched, "is_business_hours", return_value=True):
            assert sched.can_start() is True

    @patch("homie_core.finetune.scheduler._get_idle_minutes", return_value=0)
    @patch("homie_core.finetune.scheduler._get_gpu_vram_usage", return_value=50.0)
    @patch("homie_core.finetune.scheduler._get_cpu_usage", return_value=50.0)
    def test_cannot_start_during_business_hours_if_not_idle(self, _cpu, _gpu, _idle):
        sched = self._make_scheduler()
        with patch.object(sched, "is_business_hours", return_value=True):
            assert sched.can_start() is False

    # ── should_interrupt ─────────────────────────────────────────────

    @patch("homie_core.finetune.scheduler._get_idle_minutes", return_value=2)
    @patch("homie_core.finetune.scheduler._get_gpu_vram_usage", return_value=10.0)
    @patch("homie_core.finetune.scheduler._get_cpu_usage", return_value=70.0)
    def test_should_interrupt_on_user_activity(self, _cpu, _gpu, _idle):
        sched = self._make_scheduler()
        assert sched.should_interrupt() is True

    @patch("homie_core.finetune.scheduler._get_idle_minutes", return_value=30)
    @patch("homie_core.finetune.scheduler._get_gpu_vram_usage", return_value=10.0)
    @patch("homie_core.finetune.scheduler._get_cpu_usage", return_value=20.0)
    def test_should_not_interrupt_when_idle_and_low_cpu(self, _cpu, _gpu, _idle):
        sched = self._make_scheduler()
        assert sched.should_interrupt() is False

    @patch("homie_core.finetune.scheduler._get_idle_minutes", return_value=30)
    @patch("homie_core.finetune.scheduler._get_gpu_vram_usage", return_value=90.0)
    @patch("homie_core.finetune.scheduler._get_cpu_usage", return_value=20.0)
    def test_should_interrupt_on_high_gpu_vram(self, _cpu, _gpu, _idle):
        sched = self._make_scheduler()
        assert sched.should_interrupt() is True
