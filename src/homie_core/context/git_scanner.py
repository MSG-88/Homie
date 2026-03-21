from __future__ import annotations
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GitRepoStatus:
    path: str
    branch: str
    uncommitted_count: int
    last_commit_msg: str
    last_commit_time: str


def scan_git_repos(root_dirs: list[str | Path], max_depth: int = 3) -> list[GitRepoStatus]:
    """Find git repos under root_dirs, up to max_depth levels deep."""
    repos = []
    for root in root_dirs:
        root = Path(root)
        if not root.exists():
            continue
        for git_dir in _find_git_dirs(root, max_depth):
            status = _get_repo_status(git_dir.parent)
            if status:
                repos.append(status)
    return repos


def _find_git_dirs(root: Path, max_depth: int) -> list[Path]:
    results = []

    def walk(path: Path, depth: int):
        if depth > max_depth:
            return
        try:
            for child in path.iterdir():
                if child.is_dir():
                    if child.name == ".git":
                        results.append(child)
                    elif not child.name.startswith(".") and child.name not in (
                        "node_modules", "__pycache__", ".venv", "venv"
                    ):
                        walk(child, depth + 1)
        except PermissionError:
            pass

    walk(root, 0)
    return results


def _get_repo_status(path: Path) -> Optional[GitRepoStatus]:
    def run(cmd):
        try:
            r = subprocess.run(cmd, cwd=path, capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if not branch:
        return None
    status_lines = run(["git", "status", "--porcelain"])
    uncommitted = len([l for l in status_lines.splitlines() if l.strip()])
    return GitRepoStatus(
        path=str(path),
        branch=branch,
        uncommitted_count=uncommitted,
        last_commit_msg=run(["git", "log", "-1", "--pretty=%s"]),
        last_commit_time=run(["git", "log", "-1", "--pretty=%cr"]),
    )
