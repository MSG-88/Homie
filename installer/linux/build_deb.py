#!/usr/bin/env python3
"""Build Homie AI .deb package for Ubuntu/Debian."""
from __future__ import annotations

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


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def freeze() -> Path:
    print("\n=== Stage 1: PyInstaller freeze ===")
    if not shutil.which("pyinstaller"):
        print("ERROR: pyinstaller not found. pip install pyinstaller")
        sys.exit(1)
    spec = INSTALLER / "homie.spec"
    run(["pyinstaller", str(spec), "--noconfirm", "--distpath", str(DIST)])
    output = DIST / "homie"
    for binary in ["homie", "homie-daemon"]:
        if not (output / binary).exists():
            print(f"ERROR: {binary} not found in {output}")
            sys.exit(1)
    return output


def build_deb(frozen_dir: Path, version: str) -> Path:
    print("\n=== Stage 2: Build .deb ===")
    pkg_name = "homie-ai"
    arch = "amd64"
    deb_root = DIST / f"{pkg_name}_{version}_{arch}"
    if deb_root.exists():
        shutil.rmtree(deb_root)

    bin_dir = deb_root / "usr" / "local" / "bin"
    share_dir = deb_root / "usr" / "local" / "share" / "homie"
    apps_dir = deb_root / "usr" / "share" / "applications"
    debian_dir = deb_root / "DEBIAN"
    for d in [bin_dir, share_dir, apps_dir, debian_dir]:
        d.mkdir(parents=True)

    # Copy frozen output — only binaries to bin, everything else to share
    for item in frozen_dir.iterdir():
        if item.is_file() and item.name in ("homie", "homie-daemon"):
            dest = bin_dir / item.name
            shutil.copy2(item, dest)
            dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
        elif item.is_file():
            shutil.copy2(item, share_dir / item.name)
        else:
            shutil.copytree(item, share_dir / item.name)

    shutil.copy2(LINUX / "homie.desktop", apps_dir / "homie.desktop")
    shutil.copy2(LINUX / "homie.service", share_dir / "homie.service")

    (debian_dir / "control").write_text(
        f"Package: {pkg_name}\n"
        f"Version: {version}\n"
        f"Section: utils\n"
        f"Priority: optional\n"
        f"Architecture: {arch}\n"
        f"Maintainer: MSG <muthu.g.subramanian@outlook.com>\n"
        f"Description: Homie AI - Local-first personal AI assistant\n"
        f" Fully local, privacy-first personal AI assistant with\n"
        f" background intelligence, voice interaction, and email integration.\n"
    )

    for script_name in ["postinst", "prerm"]:
        src = LINUX / script_name
        if src.exists():
            dest = debian_dir / script_name
            shutil.copy2(src, dest)
            dest.chmod(dest.stat().st_mode | stat.S_IEXEC)

    deb_path = DIST / f"{pkg_name}_{version}_{arch}.deb"
    run(["dpkg-deb", "--build", str(deb_root), str(deb_path)])

    if not deb_path.exists():
        print(f"ERROR: .deb not created at {deb_path}")
        sys.exit(1)
    print(f"  .deb created: {deb_path}")
    return deb_path


def main() -> None:
    version = get_version()
    print(f"Building Homie AI .deb v{version}")
    skip_freeze = "--skip-freeze" in sys.argv
    if skip_freeze:
        frozen_dir = DIST / "homie"
        if not frozen_dir.exists():
            print("ERROR: dist/homie/ not found. Run without --skip-freeze first.")
            sys.exit(1)
    else:
        frozen_dir = freeze()
    deb_path = build_deb(frozen_dir, version)
    size_mb = deb_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 50}")
    print(f"  SUCCESS: {deb_path.name} ({size_mb:.1f} MB)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
