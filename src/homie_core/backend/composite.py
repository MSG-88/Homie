from __future__ import annotations

from typing import Optional

from homie_core.backend.protocol import (
    BackendProtocol,
    EditResult,
    FileContent,
    FileInfo,
    GrepMatch,
)


class CompositeBackend:
    """Routes filesystem operations to different backends based on path prefix.

    Paths are matched against registered *routes* using longest-prefix-first
    order.  Unmatched paths fall through to the *default* backend.

    Example::

        backend = CompositeBackend(
            default=LocalFilesystemBackend("/home/user"),
            routes={
                "/vault": EncryptedVaultBackend(...),
                "/tmp":   StateBackend(),
            },
        )

    A call to ``read("/vault/secret.txt")`` resolves to the vault backend with
    the relative path ``"/secret.txt"``.
    """

    def __init__(
        self,
        default: BackendProtocol,
        routes: dict[str, BackendProtocol] | None = None,
    ) -> None:
        self._default = default
        # Sort routes longest-prefix-first so more specific routes win
        self._routes: list[tuple[str, BackendProtocol]] = sorted(
            (routes or {}).items(),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _route(self, path: str) -> tuple[BackendProtocol, str]:
        """Return ``(backend, relative_path)`` for *path*.

        Tries each route prefix from longest to shortest.  If none matches
        the path is sent to the default backend unchanged.

        The *relative_path* returned has the matched prefix stripped, leaving
        a ``"/"``-rooted path within the target backend.
        """
        for prefix, backend in self._routes:
            # Exact match: "/vault" for path "/vault" -> relative "/"
            if path == prefix:
                return backend, "/"
            # Prefix match: "/vault/..." -> strip prefix, keep leading /
            if path.startswith(prefix + "/"):
                relative = path[len(prefix):]  # keeps the leading /
                return backend, relative

        return self._default, path

    # ------------------------------------------------------------------
    # ls — special: aggregates for root
    # ------------------------------------------------------------------

    def ls(self, path: str = "/") -> list[FileInfo]:
        """List entries under *path*.

        When *path* is ``"/"`` the results are the union of:
        - entries from the default backend's root, and
        - a virtual :class:`~homie_core.backend.protocol.FileInfo` for each
          top-level route directory (e.g. ``"vault"`` for ``"/vault"``).

        For any other path the call is delegated to the appropriate backend.
        """
        if path == "/":
            seen_names: set[str] = set()
            entries: list[FileInfo] = []

            # Entries from default backend root
            for entry in self._default.ls("/"):
                if entry.name not in seen_names:
                    seen_names.add(entry.name)
                    entries.append(entry)

            # Virtual directory entries for each top-level route prefix
            for prefix, _backend in self._routes:
                # Only top-level routes contribute a virtual dir at root
                # e.g. "/vault" -> name "vault"
                stripped = prefix.lstrip("/")
                if "/" not in stripped and stripped:  # single component
                    if stripped not in seen_names:
                        seen_names.add(stripped)
                        entries.append(
                            FileInfo(
                                path=prefix,
                                name=stripped,
                                is_dir=True,
                            )
                        )

            return entries

        backend, relative = self._route(path)
        return backend.ls(relative)

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        backend, relative = self._route(path)
        return backend.read(relative, offset=offset, limit=limit)

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(self, path: str, content: str) -> None:
        backend, relative = self._route(path)
        backend.write(relative, content)

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
        backend, relative = self._route(path)
        return backend.edit(relative, old, new, replace_all=replace_all)

    # ------------------------------------------------------------------
    # glob — searches all backends, restores path prefixes
    # ------------------------------------------------------------------

    def glob(self, pattern: str) -> list[str]:
        """Search *all* backends and return merged results.

        Paths from routed backends have their prefix restored so callers
        always see fully-qualified composite paths.
        """
        results: list[str] = list(self._default.glob(pattern))

        for prefix, backend in self._routes:
            for match in backend.glob(pattern):
                # Restore the route prefix so paths are composite-rooted
                # match is like "/secret.txt", prefix is like "/vault"
                composite_path = prefix + match if match.startswith("/") else prefix + "/" + match
                results.append(composite_path)

        return results

    # ------------------------------------------------------------------
    # grep — searches all backends, restores path prefixes
    # ------------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        path: str = "/",
        include: Optional[str] = None,
    ) -> list[GrepMatch]:
        """Search across backends, scoped by *path*.

        When *path* is ``"/"`` every backend is searched and results are
        merged with composite-rooted paths.  For a specific path the call
        is routed to the appropriate backend.
        """
        if path == "/":
            results: list[GrepMatch] = list(
                self._default.grep(pattern, path="/", include=include)
            )

            for prefix, backend in self._routes:
                for match in backend.grep(pattern, path="/", include=include):
                    # Rewrite the path to include the route prefix
                    composite_path = (
                        prefix + match.path
                        if match.path.startswith("/")
                        else prefix + "/" + match.path
                    )
                    results.append(
                        GrepMatch(
                            path=composite_path,
                            line_number=match.line_number,
                            line=match.line,
                        )
                    )

            return results

        # Scoped search: route to the correct backend
        backend, relative = self._route(path)
        raw_results = backend.grep(pattern, path=relative, include=include)

        # If path matched a route, restore the prefix on all returned paths
        if backend is not self._default:
            # Find the matched prefix for this path
            matched_prefix = ""
            for prefix, b in self._routes:
                if b is backend:
                    matched_prefix = prefix
                    break
            return [
                GrepMatch(
                    path=matched_prefix + m.path if m.path.startswith("/") else matched_prefix + "/" + m.path,
                    line_number=m.line_number,
                    line=m.line,
                )
                for m in raw_results
            ]

        return raw_results
