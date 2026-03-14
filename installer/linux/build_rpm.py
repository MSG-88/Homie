#!/usr/bin/env python3
"""Build Homie AI .rpm package for Fedora/RHEL."""
from __future__ import annotations

import shutil
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


def main() -> None:
    version = get_version()
    print(f"Building Homie AI .rpm v{version}")

    frozen_dir = DIST / "homie"
    if not frozen_dir.exists():
        print("ERROR: dist/homie/ not found. Run PyInstaller freeze first.")
        sys.exit(1)

    if not shutil.which("rpmbuild"):
        print("ERROR: rpmbuild not found. Install rpm-build package.")
        sys.exit(1)

    rpmbuild_root = DIST / "rpmbuild"
    for d in ["BUILD", "RPMS", "SOURCES", "SPECS", "SRPMS"]:
        (rpmbuild_root / d).mkdir(parents=True, exist_ok=True)

    sources = rpmbuild_root / "SOURCES"
    if (sources / "homie").exists():
        shutil.rmtree(sources / "homie")
    shutil.copytree(frozen_dir, sources / "homie")
    shutil.copy2(LINUX / "homie.desktop", sources / "homie.desktop")
    shutil.copy2(LINUX / "homie.service", sources / "homie.service")

    spec_content = (LINUX / "homie.spec.rpmbuild").read_text().replace("@@VERSION@@", version)
    spec_dest = rpmbuild_root / "SPECS" / "homie-ai.spec"
    spec_dest.write_text(spec_content)

    run([
        "rpmbuild", "-bb",
        f"--define=_topdir {rpmbuild_root}",
        f"--define=_sourcedir {sources}",
        str(spec_dest),
    ])

    rpms = list((rpmbuild_root / "RPMS").rglob("*.rpm"))
    if rpms:
        final = DIST / rpms[0].name
        shutil.copy2(rpms[0], final)
        size_mb = final.stat().st_size / (1024 * 1024)
        print(f"\n{'=' * 50}")
        print(f"  SUCCESS: {final.name} ({size_mb:.1f} MB)")
        print(f"{'=' * 50}")
    else:
        print("ERROR: No .rpm found after build")
        sys.exit(1)


if __name__ == "__main__":
    main()
