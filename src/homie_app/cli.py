from __future__ import annotations

import argparse
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="homie", description="Homie AI — Local Personal Assistant")
    subparsers = parser.add_subparsers(dest="command")

    # homie start
    start_parser = subparsers.add_parser("start", help="Start the assistant")
    start_parser.add_argument("--config", type=str, help="Path to config file")
    start_parser.add_argument("--no-voice", action="store_true", help="Disable voice pipeline")
    start_parser.add_argument("--no-tray", action="store_true", help="Disable system tray")

    # homie init
    init_parser = subparsers.add_parser("init", help="Initialize Homie for first use")
    init_parser.add_argument("--auto", action="store_true", help="Fully automatic setup")

    # homie model
    model_parser = subparsers.add_parser("model", help="Model management")
    model_sub = model_parser.add_subparsers(dest="model_command")
    model_sub.add_parser("list", help="List installed models")
    dl = model_sub.add_parser("download", help="Download a model from HuggingFace")
    dl.add_argument("repo_id", type=str)
    dl.add_argument("--filename", type=str, default="")
    add = model_sub.add_parser("add", help="Register a local model file")
    add.add_argument("path", type=str)
    add.add_argument("--name", type=str, required=True)
    add.add_argument("--format", type=str, default="gguf", choices=["gguf", "safetensors", "cloud"])
    add.add_argument("--params", type=str, default="unknown")
    rm = model_sub.add_parser("remove", help="Remove a model")
    rm.add_argument("name", type=str)
    sw = model_sub.add_parser("switch", help="Switch active model")
    sw.add_argument("name", type=str)
    model_sub.add_parser("benchmark", help="Benchmark current model")

    # homie plugin
    plugin_parser = subparsers.add_parser("plugin", help="Plugin management")
    plugin_sub = plugin_parser.add_subparsers(dest="plugin_command")
    plugin_sub.add_parser("list", help="List all plugins")
    en = plugin_sub.add_parser("enable", help="Enable a plugin")
    en.add_argument("name", type=str)
    dis = plugin_sub.add_parser("disable", help="Disable a plugin")
    dis.add_argument("name", type=str)

    # homie backup / restore
    backup = subparsers.add_parser("backup", help="Create encrypted backup")
    backup.add_argument("--to", type=str, required=True, dest="backup_path")
    restore = subparsers.add_parser("restore", help="Restore from backup")
    restore.add_argument("--from", type=str, required=True, dest="restore_path")

    # homie chat (interactive)
    subparsers.add_parser("chat", help="Start interactive chat mode")

    # homie daemon
    daemon_parser = subparsers.add_parser("daemon", help="Run as always-active background daemon")
    daemon_parser.add_argument("--config", type=str, help="Path to config file")

    return parser


def _validate_model_entry(entry, cfg) -> str | None:
    """Check if a model entry is actually usable. Returns error message or None."""
    if entry.format == "cloud":
        if not cfg.llm.api_key:
            return (
                f"Cloud model '{entry.name}' requires an API key.\n"
                f"  Set it via: HOMIE_API_KEY env var, or re-run 'homie init'."
            )
        if not cfg.llm.api_base_url:
            return (
                f"Cloud model '{entry.name}' has no endpoint configured.\n"
                f"  Re-run 'homie init' to set up the cloud provider."
            )
    else:
        model_path = Path(entry.path)
        if not model_path.exists():
            return (
                f"Local model file not found: {entry.path}\n"
                f"  Download it with: homie model download {entry.repo_id or '<repo_id>'}\n"
                f"  Or re-run 'homie init' to configure a different model."
            )
    return None


def _pick_usable_model(registry, cfg):
    """Find the best usable model: active first, then any valid one."""
    from homie_core.model.registry import ModelEntry

    # Try active model first
    entry = registry.get_active()
    if entry:
        error = _validate_model_entry(entry, cfg)
        if not error:
            return entry, None
        # Active model isn't usable — try others
        print(f"  Active model not usable: {error}")

    # Try all registered models
    for candidate in registry.list_models():
        if candidate.active:
            continue  # Already tried
        error = _validate_model_entry(candidate, cfg)
        if not error:
            print(f"  Falling back to: {candidate.name}")
            return candidate, None

    # Try config fallback
    if cfg.llm.model_path:
        fallback = ModelEntry(
            name=cfg.llm.model_path if cfg.llm.backend == "cloud" else "Qwen3.5-35B-A3B",
            path=cfg.llm.model_path,
            format=cfg.llm.backend,
            params="cloud" if cfg.llm.backend == "cloud" else "35B-A3B",
        )
        error = _validate_model_entry(fallback, cfg)
        if not error:
            return fallback, None

    return None, "No usable model found. Run 'homie init' to set up."


def _load_model_engine(cfg):
    """Load the model engine from local file or cloud API."""
    from homie_core.model.engine import ModelEngine
    from homie_core.model.registry import ModelRegistry

    engine = ModelEngine()

    # Check registry for a usable model
    registry = ModelRegistry(Path(cfg.storage.path) / cfg.storage.models_dir)
    registry.initialize()

    entry, error = _pick_usable_model(registry, cfg)
    if not entry:
        print(f"  {error}")
        return None, None

    try:
        if entry.format == "cloud":
            print(f"  Connecting to cloud API: {entry.path}")
            print(f"  Endpoint: {cfg.llm.api_base_url}")
            engine.load(entry, api_key=cfg.llm.api_key, base_url=cfg.llm.api_base_url)
            print(f"  Connected!")
        else:
            print(f"  Loading model: {entry.name} ({entry.format})")
            print(f"  Path: {entry.path}")
            print(f"  Context: {cfg.llm.context_length:,} tokens")
            engine.load(entry, n_ctx=cfg.llm.context_length, n_gpu_layers=cfg.llm.gpu_layers)
    except FileNotFoundError as e:
        print(f"  Failed: {e}")
        print(f"  Make sure the model file exists or install llama-server.")
        return None, None
    except ConnectionError as e:
        print(f"  Connection failed: {e}")
        return None, None
    except TimeoutError as e:
        print(f"  Timed out loading model: {e}")
        return None, None
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return None, None

    print(f"  Model loaded successfully!")
    return engine, entry


def cmd_chat(args, config=None):
    from homie_core.config import load_config
    from homie_core.memory.working import WorkingMemory
    from homie_core.brain.orchestrator import BrainOrchestrator
    from homie_app.prompts.system import SYSTEM_PROMPT

    cfg = config or load_config(args.config if hasattr(args, 'config') else None)
    print("=" * 50)
    print("  Homie AI v0.1.0 — Interactive Chat")
    print("=" * 50)

    print("\n[Loading model...]")
    engine, entry = _load_model_engine(cfg)
    if not engine:
        print("  No model found. Run 'homie init' first, or 'homie model add <path>'.")
        return

    wm = WorkingMemory()
    brain = BrainOrchestrator(model_engine=engine, working_memory=wm)
    brain.set_system_prompt(SYSTEM_PROMPT)

    user_name = cfg.user_name or "User"
    print(f"\nHey{' ' + user_name if user_name != 'User' else ''}! Type your message or 'quit' to exit.\n")

    while True:
        try:
            user_input = input(f"{user_name}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.lower() in ("quit", "exit", ":q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        try:
            import sys
            from homie_app.loading import CLILoadingSpinner, get_random_thinking_message

            # Show spinner until first token arrives
            spinner = CLILoadingSpinner(style="random")
            spinner.start()
            first_token = True
            try:
                for token in brain.process_stream(user_input):
                    if first_token:
                        spinner.stop()
                        sys.stdout.write("Homie> ")
                        sys.stdout.flush()
                        first_token = False
                    sys.stdout.write(token)
                    sys.stdout.flush()
            except Exception:
                if not first_token:
                    raise
                # Stream not supported — fall back to blocking generate
                spinner.stop()
                sys.stdout.write(f"Homie> {brain.process(user_input)}")
                sys.stdout.flush()
            finally:
                if first_token:
                    spinner.stop()
            print("\n")
        except ConnectionError as e:
            print(f"Homie> Connection failed: {e}\n")
        except Exception as e:
            print(f"Homie> [Error: {e}]\n")

    engine.unload()
    print("Model unloaded.")


def cmd_model(args, config=None):
    from homie_core.config import load_config
    from homie_core.model.registry import ModelRegistry
    cfg = config or load_config()
    registry = ModelRegistry(Path(cfg.storage.path) / cfg.storage.models_dir)
    registry.initialize()

    if args.model_command == "list":
        models = registry.list_models()
        if not models:
            print("No models installed. Run 'homie model download <repo_id>' or 'homie init'.")
        for m in models:
            active = " [ACTIVE]" if m.active else ""
            print(f"  {m.name} ({m.params}, {m.format}){active}")
            if m.format == "cloud":
                print(f"    Endpoint: cloud API")
            else:
                print(f"    Path: {m.path}")

    elif args.model_command == "add":
        path = Path(args.path)
        if not path.exists():
            print(f"Error: File not found: {path}")
            return
        registry.register(args.name, path, format=args.format, params=args.params)
        print(f"Registered model '{args.name}' from {path}")

    elif args.model_command == "remove":
        registry.remove(args.name)
        print(f"Removed model '{args.name}'")

    elif args.model_command == "switch":
        if not registry.get(args.name):
            print(f"Error: Model '{args.name}' not found")
            return
        registry.set_active(args.name)
        print(f"Switched to model '{args.name}'")

    else:
        print("Usage: homie model {list|download|add|remove|switch|benchmark}")


def cmd_plugin(args, config=None):
    from homie_core.plugins.manager import PluginManager
    mgr = PluginManager()

    if args.plugin_command == "list":
        plugins = mgr.list_plugins()
        if not plugins:
            print("No plugins registered.")
        for p in plugins:
            status = "enabled" if p["enabled"] else "disabled"
            print(f"  {p['name']} ({status}) - {p['description']}")

    elif args.plugin_command == "enable":
        if mgr.enable(args.name):
            print(f"Enabled plugin '{args.name}'")
        else:
            print(f"Failed to enable plugin '{args.name}'")

    elif args.plugin_command == "disable":
        if mgr.disable(args.name):
            print(f"Disabled plugin '{args.name}'")
        else:
            print(f"Failed to disable plugin '{args.name}'")

    else:
        print("Usage: homie plugin {list|enable|disable}")


def cmd_backup(args, config=None):
    from homie_core.config import load_config
    from homie_core.storage.backup import BackupManager
    import getpass
    cfg = config or load_config()
    passphrase = getpass.getpass("Backup passphrase: ")
    mgr = BackupManager()
    out = mgr.create_backup(Path(cfg.storage.path), Path(args.backup_path), passphrase)
    print(f"Backup created at {out}")


def cmd_restore(args, config=None):
    from homie_core.config import load_config
    from homie_core.storage.backup import BackupManager
    import getpass
    cfg = config or load_config()
    passphrase = getpass.getpass("Backup passphrase: ")
    mgr = BackupManager()
    mgr.restore_backup(Path(args.restore_path), Path(cfg.storage.path), passphrase)
    print(f"Restored to {cfg.storage.path}")


def cmd_daemon(args, config=None):
    from homie_app.daemon import HomieDaemon
    daemon = HomieDaemon(config_path=getattr(args, 'config', None))
    daemon.start()


def main(argv: list[str] | None = None):
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "chat": cmd_chat,
        "model": cmd_model,
        "plugin": cmd_plugin,
        "backup": cmd_backup,
        "restore": cmd_restore,
        "daemon": cmd_daemon,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    elif args.command == "start":
        cmd_chat(args)
    elif args.command == "init":
        print("Initializing Homie AI...")
        from homie_app.init import run_init
        run_init(auto=getattr(args, 'auto', False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
