#!/usr/bin/env python3
"""Unified cross-platform build script for Homie AI.

Usage:
    python installer/build.py --target deb
    python installer/build.py --target rpm
    python installer/build.py --target appimage
    python installer/build.py --target dmg
    python installer/build.py --target msi
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALLER = ROOT / "installer"


def get_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def build_target(target: str, version: str) -> None:
    targets = {
        "msi": INSTALLER / "build_msi.py",
        "deb": INSTALLER / "linux" / "build_deb.py",
        "rpm": INSTALLER / "linux" / "build_rpm.py",
        "appimage": INSTALLER / "linux" / "build_appimage.py",
        "dmg": INSTALLER / "macos" / "build_dmg.py",
        "docker": INSTALLER / "docker" / "build_docker.py",
    }
    script = targets.get(target)
    if not script:
        print(f"ERROR: Unknown target '{target}'")
        sys.exit(1)
    run([sys.executable, str(script)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Homie AI packages")
    parser.add_argument(
        "--target", required=True,
        choices=["deb", "rpm", "appimage", "dmg", "msi", "docker", "all"],
    )
    parser.add_argument("--skip-freeze", action="store_true")
    args = parser.parse_args()

    version = get_version()
    print(f"Building Homie AI v{version} — target: {args.target}")

    if args.target == "all":
        os_name = platform.system()
        if os_name == "Linux":
            for t in ["deb", "rpm", "appimage"]:
                build_target(t, version)
        elif os_name == "Darwin":
            build_target("dmg", version)
        elif os_name == "Windows":
            build_target("msi", version)
    else:
        build_target(args.target, version)


if __name__ == "__main__":
    main()
