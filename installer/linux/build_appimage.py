#!/usr/bin/env python3
"""Build Homie AI AppImage (universal Linux package)."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INSTALLER = ROOT / "installer"
LINUX = INSTALLER / "linux"
DIST = ROOT / "dist"


def get_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT, env=env)


def main() -> None:
    version = get_version()
    print(f"Building Homie AI AppImage v{version}")

    frozen_dir = DIST / "homie"
    if not frozen_dir.exists():
        print("ERROR: dist/homie/ not found. Run PyInstaller freeze first.")
        sys.exit(1)

    if not shutil.which("appimagetool"):
        print("ERROR: appimagetool not found.")
        sys.exit(1)

    appdir = DIST / "HomieAI.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)

    usr_bin = appdir / "usr" / "bin"
    usr_lib = appdir / "usr" / "lib"
    usr_bin.mkdir(parents=True)
    usr_lib.mkdir(parents=True)

    for item in frozen_dir.iterdir():
        if item.is_file() and item.name in ("homie", "homie-daemon"):
            dest = usr_bin / item.name
            shutil.copy2(item, dest)
            dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
        elif item.is_file():
            shutil.copy2(item, usr_lib / item.name)
        else:
            shutil.copytree(item, usr_lib / item.name)

    apprun = appdir / "AppRun"
    shutil.copy2(LINUX / "AppRun", apprun)
    apprun.chmod(apprun.stat().st_mode | stat.S_IEXEC)

    shutil.copy2(LINUX / "homie.desktop", appdir / "homie.desktop")

    # Minimal placeholder icon
    icon_path = appdir / "homie.png"
    icon_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    appimage_name = f"HomieAI-{version}-x86_64.AppImage"
    appimage_path = DIST / appimage_name
    env = os.environ.copy()
    env["ARCH"] = "x86_64"
    run(["appimagetool", str(appdir), str(appimage_path)], env=env)

    if not appimage_path.exists():
        print(f"ERROR: AppImage not created at {appimage_path}")
        sys.exit(1)

    size_mb = appimage_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 50}")
    print(f"  SUCCESS: {appimage_name} ({size_mb:.1f} MB)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
