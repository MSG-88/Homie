"""Build Docker images for Homie AI."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def get_version() -> str:
    pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
    for line in pyproject.read_text().splitlines():
        if line.strip().startswith("version"):
            return line.split("=")[1].strip().strip('"')
    return "0.0.0"


def build(target: str = "full", tag: str = "") -> None:
    version = get_version()
    if not tag:
        tag = f"homie-ai:{version}-{target}"

    root = Path(__file__).parent.parent.parent
    cmd = [
        "docker", "build",
        "--target", target,
        "--tag", tag,
        "--tag", f"homie-ai:latest-{target}",
        str(root),
    ]
    print(f"Building: {tag}")
    subprocess.run(cmd, check=True)
    print(f"Built: {tag}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build Homie Docker images")
    parser.add_argument("--target", choices=["spoke", "full"], default="full")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    build(target=args.target, tag=args.tag)


if __name__ == "__main__":
    main()
