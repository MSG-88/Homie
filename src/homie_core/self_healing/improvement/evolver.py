"""Architecture evolver — creates, removes, and restructures modules."""

import logging
from pathlib import Path
from typing import Optional

from .rollback import RollbackManager

logger = logging.getLogger(__name__)


class ArchitectureEvolver:
    """Evolves module structure — create, remove, split, merge."""

    def __init__(
        self,
        rollback_manager: RollbackManager,
        project_root: Path | str,
        locked_paths: Optional[list[str]] = None,
    ) -> None:
        self._rollback = rollback_manager
        self._root = Path(project_root)
        self._locked = locked_paths or []

    def _is_locked(self, file_path: Path) -> bool:
        try:
            rel = file_path.relative_to(self._root)
        except ValueError:
            rel = file_path
        rel_str = str(rel).replace("\\", "/")
        for lock in self._locked:
            if lock.endswith("/"):
                if rel_str.startswith(lock) or rel_str.startswith(lock.rstrip("/")):
                    return True
            elif rel_str == lock:
                return True
        return False

    def create_module(
        self,
        file_path: Path | str,
        content: str,
        reason: str = "",
    ) -> str:
        """Create a new module file. Returns version_id."""
        file_path = Path(file_path)
        if self._is_locked(file_path):
            raise PermissionError(f"Path is core-locked: {file_path}")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Snapshot the creation (empty file marker for rollback)
        version_id = self._rollback.snapshot([], reason=f"create: {reason}")

        file_path.write_text(content)
        logger.info("Created module %s (version: %s): %s", file_path, version_id, reason)
        return version_id

    def remove_module(
        self,
        file_path: Path | str,
        reason: str = "",
    ) -> str:
        """Remove a module file. Returns version_id for rollback."""
        file_path = Path(file_path)
        if self._is_locked(file_path):
            raise PermissionError(f"Path is core-locked: {file_path}")

        version_id = self._rollback.snapshot(file_path, reason=f"remove: {reason}")
        file_path.unlink()
        logger.info("Removed module %s (version: %s): %s", file_path, version_id, reason)
        return version_id

    def split_module(
        self,
        source: Path | str,
        targets: dict[Path | str, str],
        reason: str = "",
    ) -> str:
        """Split a module into multiple files. Removes the original."""
        source = Path(source)
        if self._is_locked(source):
            raise PermissionError(f"Source is core-locked: {source}")

        # Snapshot source before splitting
        version_id = self._rollback.snapshot(source, reason=f"split: {reason}")

        # Write target files
        for target_path, content in targets.items():
            target_path = Path(target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content)

        # Remove original
        source.unlink()
        logger.info("Split %s into %d files (version: %s): %s", source, len(targets), version_id, reason)
        return version_id
