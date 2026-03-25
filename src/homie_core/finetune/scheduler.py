"""Finetune scheduler with idle detection and business hours awareness.

Decides when finetuning cycles are allowed to run based on system load,
user activity, and time-of-day constraints.
"""

from __future__ import annotations

import logging
import platform
from datetime import datetime

from homie_core.finetune.config import ScheduleConfig

logger = logging.getLogger(__name__)


def _get_cpu_usage() -> float:
    """Return 5-min avg CPU usage percent. Uses psutil if available."""
    try:
        import psutil

        return psutil.cpu_percent(interval=1)
    except ImportError:
        return 0.0


def _get_gpu_vram_usage() -> float:
    """Return GPU VRAM usage percent."""
    try:
        import torch

        if torch.cuda.is_available():
            used = (
                torch.cuda.memory_allocated()
                / torch.cuda.get_device_properties(0).total_mem
                * 100
            )
            return used
    except (ImportError, RuntimeError):
        pass
    return 0.0


def _get_idle_minutes() -> int:
    """Get minutes since last user input. Windows: GetLastInputInfo. Linux: /proc."""
    if platform.system() == "Windows":
        try:
            import ctypes

            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.c_uint),
                    ("dwTime", ctypes.c_uint),
                ]

            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
            ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
            millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
            return millis // 60000
        except Exception:
            return 0
    return 0  # Can't detect on other platforms, assume not idle


class FinetuneScheduler:
    """Schedule finetuning cycles based on business hours and system idle state."""

    def __init__(self, config: ScheduleConfig) -> None:
        self._config = config

    def is_business_hours(self, dt: datetime | None = None) -> bool:
        """Return True if *dt* falls within configured business hours on a business day."""
        dt = dt or datetime.now()
        if dt.weekday() not in self._config.business_days:
            return False
        return self._config.business_hours_start <= dt.hour < self._config.business_hours_end

    def is_system_idle(self) -> bool:
        """Return True if CPU, GPU, and user idle time all indicate low activity."""
        cpu = _get_cpu_usage()
        gpu = _get_gpu_vram_usage()
        idle = _get_idle_minutes()
        return cpu < 15.0 and gpu < 30.0 and idle >= self._config.min_idle_minutes

    def can_start(self) -> bool:
        """True if non-business hours OR system idle."""
        return not self.is_business_hours() or self.is_system_idle()

    def should_interrupt(self) -> bool:
        """True if CPU > 60%, user recently active, or GPU VRAM > 80%."""
        cpu = _get_cpu_usage()
        idle = _get_idle_minutes()
        gpu = _get_gpu_vram_usage()
        return cpu > 60.0 or idle < 5 or gpu > 80.0
