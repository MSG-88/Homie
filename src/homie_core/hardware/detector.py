from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field

import psutil


@dataclass
class GPUInfo:
    name: str = ""
    vram_mb: int = 0
    cuda_available: bool = False


@dataclass
class HardwareInfo:
    os_name: str = ""
    os_version: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0.0
    gpus: list[GPUInfo] = field(default_factory=list)
    has_microphone: bool = False

    @property
    def best_gpu_vram_gb(self) -> float:
        if not self.gpus:
            return 0.0
        return max(g.vram_mb for g in self.gpus) / 1024.0


def detect_hardware() -> HardwareInfo:
    info = HardwareInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        cpu_cores=psutil.cpu_count(logical=True) or 1,
        ram_gb=round(psutil.virtual_memory().total / (1024**3), 1),
    )
    info.gpus = _detect_gpus()
    info.has_microphone = _detect_microphone()
    return info


def _detect_gpus() -> list[GPUInfo]:
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if "," in line:
                    name, vram = line.split(",", 1)
                    gpus.append(GPUInfo(name=name.strip(), vram_mb=int(vram.strip()), cuda_available=True))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus


def _detect_microphone() -> bool:
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return any(d.get("max_input_channels", 0) > 0 for d in devices)
    except Exception:
        pass

    # Windows fallback: query PnP audio endpoints via PowerShell
    import sys
    if sys.platform == "win32":
        try:
            import subprocess
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-PnpDevice -Class AudioEndpoint -Status OK"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                if "microphone" in line.lower():
                    return True
        except Exception:
            pass

    return False
