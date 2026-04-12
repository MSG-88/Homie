from __future__ import annotations

import shutil
import socket
import subprocess
from typing import Optional

import psutil

from homie_core.platform.base import PlatformAdapter


class MacOSAdapter(PlatformAdapter):
    def get_hostname(self) -> str:
        return socket.gethostname()

    def get_active_window(self) -> Optional[str]:
        try:
            script = (
                'tell application "System Events" to get name of first process '
                'whose frontmost is true'
            )
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                return result.stdout.strip() or None
            return None
        except Exception:
            return None

    def get_system_metrics(self) -> dict:
        ram = psutil.virtual_memory()
        disk = shutil.disk_usage("/")
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_total_gb": round(ram.total / (1024 ** 3), 2),
            "ram_used_gb": round(ram.used / (1024 ** 3), 2),
            "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        }

    def send_notification(self, title: str, body: str) -> None:
        try:
            script = f'display notification "{body}" with title "{title}"'
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

    def get_gpu_info(self) -> Optional[dict]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) < 5:
                return None
            return {
                "name": parts[0],
                "memory_total_mb": parts[1],
                "memory_used_mb": parts[2],
                "temperature_c": parts[3],
                "utilization_percent": parts[4],
            }
        except Exception:
            return None
