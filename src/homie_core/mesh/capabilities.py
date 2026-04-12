"""Hardware capability detection for mesh node scoring."""
from __future__ import annotations
import shutil, sys
from dataclasses import dataclass
from typing import Optional
import psutil

@dataclass
class NodeCapabilities:
    gpu: Optional[dict]
    cpu_cores: int
    ram_gb: float
    disk_free_gb: float
    os: str
    has_mic: bool
    has_display: bool
    has_model_loaded: bool
    model_name: Optional[str]

    def capability_score(self) -> float:
        score = 0.0
        if self.gpu:
            score += self.gpu.get("vram_gb", 0) * 10
        score += self.ram_gb * 2
        score += self.cpu_cores
        if self.has_model_loaded:
            score += 50
        return score

    def to_dict(self) -> dict:
        return {
            "gpu": self.gpu, "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb, "disk_free_gb": self.disk_free_gb,
            "os": self.os, "has_mic": self.has_mic, "has_display": self.has_display,
            "has_model_loaded": self.has_model_loaded, "model_name": self.model_name,
        }

def _detect_gpu() -> Optional[dict]:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split("\n")[0].split(", ")
            if len(parts) == 2:
                return {"name": parts[0].strip(), "vram_gb": round(float(parts[1].strip()) / 1024, 1)}
    except (FileNotFoundError, Exception):
        pass
    return None

def _detect_mic() -> bool:
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return any(d.get("max_input_channels", 0) > 0 for d in devices)
    except Exception:
        return False

def _detect_display() -> bool:
    if sys.platform == "win32":
        return True
    try:
        import os
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    except Exception:
        return False

def _detect_os() -> str:
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "macos"
    return "linux"

def detect_capabilities() -> NodeCapabilities:
    disk = shutil.disk_usage("/") if sys.platform != "win32" else shutil.disk_usage("C:\\")
    return NodeCapabilities(
        gpu=_detect_gpu(),
        cpu_cores=psutil.cpu_count(logical=True) or 1,
        ram_gb=round(psutil.virtual_memory().total / (1024 ** 3), 1),
        disk_free_gb=round(disk.free / (1024 ** 3), 1),
        os=_detect_os(),
        has_mic=_detect_mic(),
        has_display=_detect_display(),
        has_model_loaded=False,
        model_name=None,
    )
