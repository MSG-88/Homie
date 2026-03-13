#!/usr/bin/env python3
"""Build HomieSetup MSI installer.

Usage:
    python installer/build_msi.py
    python installer/build_msi.py --skip-freeze   (reuse existing dist/)
    python installer/build_msi.py --sign cert.pfx  (sign the MSI)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALLER = ROOT / "installer"
DIST = ROOT / "dist"
SPEC = INSTALLER / "homie.spec"


def get_version() -> str:
    """Extract version from pyproject.toml."""
    pyproject = ROOT / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command, raising on failure."""
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def check_tool(name: str) -> bool:
    """Check if a tool is on PATH."""
    return shutil.which(name) is not None


def stage_freeze() -> Path:
    """Stage 1: PyInstaller freeze."""
    print("\n=== Stage 1: PyInstaller freeze ===")
    if not check_tool("pyinstaller"):
        print("ERROR: pyinstaller not found. Install with: pip install pyinstaller")
        sys.exit(1)

    run(["pyinstaller", str(SPEC), "--noconfirm", "--distpath", str(DIST)])

    output = DIST / "homie"
    if not (output / "homie.exe").exists():
        print("ERROR: homie.exe not found in dist/homie/")
        sys.exit(1)
    if not (output / "homie-daemon.exe").exists():
        print("ERROR: homie-daemon.exe not found in dist/homie/")
        sys.exit(1)

    print(f"  Frozen to {output}")
    return output


def stage_msi(frozen_dir: Path, version: str) -> Path:
    """Stage 2: WiX build."""
    print("\n=== Stage 2: WiX MSI build ===")
    for tool in ("heat", "candle", "light"):
        if not check_tool(tool):
            print(f"ERROR: {tool}.exe not found. Install WiX Toolset v3.14 and add to PATH.")
            sys.exit(1)

    build_dir = DIST / "msi_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Harvest files from frozen directory
    components_wxs = build_dir / "components.wxs"
    run([
        "heat", "dir", str(frozen_dir),
        "-cg", "HarvestedFiles",
        "-dr", "INSTALLFOLDER",
        "-srd",
        "-ag",
        "-sfrag",
        "-var", "var.SourceDir",
        "-o", str(components_wxs),
    ])

    # Compile
    product_obj = build_dir / "product.wixobj"
    components_obj = build_dir / "components.wixobj"

    run([
        "candle",
        str(INSTALLER / "product.wxs"),
        f"-dVersion={version}",
        f"-dSourceDir={frozen_dir}",
        "-o", str(product_obj),
        "-ext", "WixUIExtension",
        "-ext", "WixUtilExtension",
    ])

    run([
        "candle",
        str(components_wxs),
        f"-dSourceDir={frozen_dir}",
        "-o", str(components_obj),
    ])

    # Link
    msi_name = f"HomieSetup-{version}.msi"
    msi_path = DIST / msi_name
    run([
        "light",
        str(product_obj),
        str(components_obj),
        "-o", str(msi_path),
        "-ext", "WixUIExtension",
        "-ext", "WixUtilExtension",
        "-cultures:en-US",
        "-b", str(frozen_dir),
        "-b", str(INSTALLER),
    ])

    if not msi_path.exists():
        print(f"ERROR: MSI not created at {msi_path}")
        sys.exit(1)

    print(f"  MSI created: {msi_path}")
    return msi_path


def stage_sign(msi_path: Path, cert_path: str) -> None:
    """Optional: sign the MSI."""
    print("\n=== Stage 3: Code signing ===")
    if not check_tool("signtool"):
        print("WARNING: signtool not found, skipping signing")
        return
    run([
        "signtool", "sign",
        "/f", cert_path,
        "/tr", "http://timestamp.digicert.com",
        "/td", "sha256",
        "/fd", "sha256",
        str(msi_path),
    ])
    print(f"  Signed: {msi_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Homie AI MSI installer")
    parser.add_argument("--skip-freeze", action="store_true",
                        help="Skip PyInstaller, reuse existing dist/homie/")
    parser.add_argument("--sign", metavar="CERT_PFX",
                        help="Sign MSI with certificate")
    args = parser.parse_args()

    version = get_version()
    print(f"Building Homie AI MSI v{version}")

    if args.skip_freeze:
        frozen_dir = DIST / "homie"
        if not frozen_dir.exists():
            print("ERROR: dist/homie/ not found. Run without --skip-freeze first.")
            sys.exit(1)
    else:
        frozen_dir = stage_freeze()

    msi_path = stage_msi(frozen_dir, version)

    if args.sign:
        stage_sign(msi_path, args.sign)

    size_mb = msi_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*50}")
    print(f"  SUCCESS: {msi_path.name} ({size_mb:.1f} MB)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
