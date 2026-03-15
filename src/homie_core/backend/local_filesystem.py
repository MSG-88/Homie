from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Optional

from homie_core.backend.protocol import (
    EditResult,
    ExecutionResult,
    FileContent,
    FileInfo,
    GrepMatch,
)


class LocalFilesystemBackend:
    """Backend that operates on the local filesystem rooted at *root_dir*.

    All public path arguments are resolved relative to *root_dir*.  Any
    attempt to escape the root via ``..`` traversal or absolute paths that
    land outside the root raises :class:`ValueError`.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir).resolve()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, path: str | Path) -> Path:
        """Resolve *path* within the root, raising ValueError if it escapes.

        The special value ``"/"`` (or ``Path("/")``) is treated as the backend
        root regardless of platform.  Any other path that resolves outside the
        root raises :class:`ValueError`.
        """
        # Normalise to a string first so we can check the sentinel value.
        path_str = str(path).replace("\\", "/")

        # Treat bare "/" as "the backend root" (platform-independent sentinel).
        if path_str in ("/", ""):
            return self._root

        p = Path(path)

        if p.is_absolute():
            # Absolute paths are always suspect — check if they live inside
            # root after resolving, and reject if not.
            candidate = p.resolve()
        else:
            candidate = (self._root / p).resolve()

        # Security check: must be inside root (or equal to root).
        try:
            candidate.relative_to(self._root)
        except ValueError:
            raise ValueError(
                f"Path {path!r} escapes root {self._root}"
            ) from None

        return candidate

    # ------------------------------------------------------------------
    # ls
    # ------------------------------------------------------------------

    def ls(self, path: str = "/") -> list[FileInfo]:
        """List directory entries under *path*."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"No such directory: {path!r}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {path!r}")

        entries: list[FileInfo] = []
        for child in resolved.iterdir():
            try:
                stat = child.stat()
                size: Optional[int] = stat.st_size if not child.is_dir() else None
                modified: Optional[float] = stat.st_mtime
            except OSError:
                size = None
                modified = None

            entries.append(
                FileInfo(
                    path=str(child),
                    name=child.name,
                    is_dir=child.is_dir(),
                    size=size,
                    modified=modified,
                )
            )
        return entries

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        """Read file lines with optional pagination.

        Uses :py:meth:`str.splitlines` so a trailing newline does **not**
        produce a spurious empty final line.
        """
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"No such file: {path!r}")

        raw = resolved.read_text(encoding="utf-8", errors="replace")
        lines = raw.splitlines()
        total = len(lines)

        sliced = lines[offset : offset + limit]
        truncated = (offset + limit) < total

        return FileContent(
            content="\n".join(sliced),
            total_lines=total,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(self, path: str, content: str) -> None:
        """Write *content* to *path*, creating parent directories as needed.

        On Unix the file is opened with ``O_NOFOLLOW`` to prevent writing
        through symlinks.  On Windows an explicit symlink check is performed
        before writing.
        """
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)

        if platform.system() == "Windows":
            # Explicit symlink check on Windows
            if resolved.exists() and resolved.is_symlink():
                raise ValueError(f"Refusing to write through symlink: {path!r}")
            resolved.write_text(content, encoding="utf-8")
        else:
            # Unix: open with O_NOFOLLOW | O_WRONLY | O_CREAT | O_TRUNC
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
            try:
                fd = os.open(resolved, flags, 0o666)
            except OSError as exc:
                # ELOOP is raised on Linux when the path is a symlink and
                # O_NOFOLLOW is set.
                import errno
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise ValueError(
                        f"Refusing to write through symlink: {path!r}"
                    ) from exc
                raise
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(content)
            except Exception:
                # fd is closed by fdopen on exception but re-raise for caller
                raise

    # ------------------------------------------------------------------
    # edit
    # ------------------------------------------------------------------

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace *old* with *new* in the file at *path*.

        If *replace_all* is False the replacement is only performed when
        *old* appears exactly once; otherwise an error is returned with the
        occurrence count.
        """
        resolved = self._resolve(path)
        if not resolved.exists():
            return EditResult(success=False, occurrences=0, error=f"File not found: {path!r}")

        content = resolved.read_text(encoding="utf-8")
        count = content.count(old)

        if count == 0:
            return EditResult(success=False, occurrences=0, error=f"String not found in {path!r}")

        if not replace_all and count > 1:
            return EditResult(
                success=False,
                occurrences=count,
                error=(
                    f"String appears {count} times in {path!r}; "
                    "use replace_all=True to replace all occurrences"
                ),
            )

        new_content = content.replace(old, new)
        resolved.write_text(new_content, encoding="utf-8")
        return EditResult(success=True, occurrences=count)

    # ------------------------------------------------------------------
    # glob
    # ------------------------------------------------------------------

    def glob(self, pattern: str) -> list[str]:
        """Return paths matching *pattern* relative to the root."""
        matches = self._root.glob(pattern)
        return [str(p) for p in matches]

    # ------------------------------------------------------------------
    # grep
    # ------------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        path: str = "/",
        include: Optional[str] = None,
    ) -> list[GrepMatch]:
        """Search for *pattern* (regex) in files under *path*.

        Binary files and unreadable files are silently skipped.
        """
        search_root = self._resolve(path)
        compiled = re.compile(pattern)

        # Collect files to search
        if search_root.is_file():
            candidates = [search_root]
        else:
            glob_pattern = f"**/{include}" if include else "**/*"
            candidates = [p for p in search_root.glob(glob_pattern) if p.is_file()]

        results: list[GrepMatch] = []
        for file_path in candidates:
            try:
                raw_bytes = file_path.read_bytes()
                # Heuristic: files containing null bytes are binary — skip them.
                if b"\x00" in raw_bytes:
                    continue
                text = raw_bytes.decode("utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                # Skip binary or unreadable files
                continue

            for lineno, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    results.append(
                        GrepMatch(
                            path=str(file_path),
                            line_number=lineno,
                            line=line,
                        )
                    )

        return results

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Run *command* in a subprocess, capturing stdout/stderr."""
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self._root),
            )
            return ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                stdout=exc.stdout or "" if isinstance(exc.stdout, str) else (exc.stdout.decode(errors="replace") if exc.stdout else ""),
                stderr=exc.stderr or "" if isinstance(exc.stderr, str) else (exc.stderr.decode(errors="replace") if exc.stderr else ""),
                exit_code=-1,
                timed_out=True,
            )
