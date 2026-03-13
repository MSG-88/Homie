"""Handler for /folder slash command — folder awareness."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _get_folder_service(ctx):
    """Build FolderService from context config."""
    from pathlib import Path
    from homie_core.vault.schema import create_cache_db
    from homie_core.folders import FolderService
    cfg = ctx.get("config")
    if not cfg:
        return None, "No configuration loaded."
    storage = Path(cfg.storage.path)
    cache_conn = create_cache_db(storage / "cache.db")
    return FolderService(cache_conn=cache_conn), None


def _parse_folder_args(args: str) -> tuple[str, dict[str, str]]:
    """Parse a path (possibly quoted) and --flag values from args.

    Returns (path_string, {flag: value}).
    """
    import shlex
    try:
        tokens = shlex.split(args, posix=False)
        # shlex with posix=False keeps quotes; strip them
        tokens = [t.strip("'\"") for t in tokens]
    except ValueError:
        tokens = args.strip().split()

    path_parts: list[str] = []
    flags: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        if tokens[i].startswith("--") and i + 1 < len(tokens):
            flags[tokens[i][2:]] = tokens[i + 1]
            i += 2
        else:
            path_parts.append(tokens[i])
            i += 1
    return " ".join(path_parts).strip("'\""), flags


def _handle_folder_watch(args: str, **ctx) -> str:
    if not args.strip():
        return "Usage: /folder watch <path> [--label <label>] [--interval <seconds>]"
    try:
        from pathlib import Path
        path_str, flags = _parse_folder_args(args)
        label = flags.get("label")
        interval = int(flags.get("interval", "300"))

        target = Path(path_str).resolve()
        if not target.is_dir():
            return f"Error: '{target}' is not a directory."

        service, err = _get_folder_service(ctx)
        if err:
            return err
        service.add_watch(str(target), label=label, scan_interval=interval)
        return f"Watching: {target} (interval: {interval}s)"
    except Exception as e:
        return f"Could not watch folder: {e}"


def _handle_folder_list(args: str, **ctx) -> str:
    try:
        service, err = _get_folder_service(ctx)
        if err:
            return err
        watches = service.list_watches()
        if not watches:
            return "No folders watched. Use: /folder watch <path>"
        lines = ["**Watched Folders:**"]
        for w in watches:
            label = f" ({w['label']})" if w.get("label") else ""
            lines.append(f"  {w['path']}{label}  [{w['file_count']} files, every {w['scan_interval']}s]")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not list folders: {e}"


def _handle_folder_scan(args: str, **ctx) -> str:
    try:
        service, err = _get_folder_service(ctx)
        if err:
            return err
        result = service.scan_tick()
        return f"Scan: {result}"
    except Exception as e:
        return f"Could not scan folders: {e}"


def _handle_folder_unwatch(args: str, **ctx) -> str:
    path_str = args.strip().strip("'\"")
    if not path_str:
        return "Usage: /folder unwatch <path>"
    try:
        from pathlib import Path
        target = Path(path_str).resolve()
        service, err = _get_folder_service(ctx)
        if err:
            return err
        removed = service.remove_watch(str(target))
        if removed:
            return f"Unwatched: {target}"
        return f"Not found: {target}"
    except Exception as e:
        return f"Could not unwatch folder: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="folder",
        description="Folder awareness (watch, list, scan, unwatch)",
        args_spec="watch|list|scan|unwatch",
        subcommands={
            "watch": SlashCommand(name="watch", description="Add a folder to watch", args_spec="<path> [--label <label>] [--interval <secs>]", handler_fn=_handle_folder_watch),
            "list": SlashCommand(name="list", description="List watched folders", handler_fn=_handle_folder_list),
            "scan": SlashCommand(name="scan", description="Force immediate scan", handler_fn=_handle_folder_scan),
            "unwatch": SlashCommand(name="unwatch", description="Remove a folder watch", args_spec="<path>", handler_fn=_handle_folder_unwatch),
        },
    ))
