from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper functions (imported by console.py and other modules — do not remove)
# ---------------------------------------------------------------------------

def _validate_model_entry(entry, cfg) -> str | None:
    """Check if a model entry is actually usable. Returns error message or None."""
    if entry.format == "hf":
        api_key = cfg.llm.api_key or os.environ.get("HF_KEY", "")
        if not api_key:
            return (
                f"HF model '{entry.name}' requires an API key.\n"
                f"  Set it via: HF_KEY env var, or re-run 'homie init'."
            )
        return None
    elif entry.format == "cloud":
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
        if entry.format == "hf":
            print(f"  Connecting to HF Inference API: {entry.path}")
            api_key = cfg.llm.api_key or os.environ.get("HF_KEY", "")
            engine.load(entry, api_key=api_key)
            print(f"  Connected to Hugging Face!")
        elif entry.format == "cloud":
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


def _create_inference_router(cfg, engine):
    """Wrap the model engine in an InferenceRouter with Qubrid fallback."""
    from homie_core.inference.router import InferenceRouter
    import os

    # Try to get Qubrid API key from environment first
    qubrid_key = os.environ.get("HOMIE_QUBRID_API_KEY", "")

    # Fall back to vault if available
    if not qubrid_key:
        try:
            from homie_core.vault.secure_vault import SecureVault
            vault = SecureVault()
            if vault.is_unlocked:
                cred = vault.get_credential(provider="qubrid", account_id="default")
                if cred:
                    qubrid_key = cred.token
        except Exception:
            pass

    router = InferenceRouter(
        config=cfg,
        model_engine=engine,
        qubrid_api_key=qubrid_key,
    )

    # Show fallback banner if applicable
    banner = router.fallback_banner
    if banner:
        print(f"\n  {banner}\n")

    return router


def _init_intelligence_stack(cfg):
    """Initialize the full intelligence stack: memories, plugins, RAG, tools.

    Returns (wm, em, sm, tool_registry, rag, plugin_mgr) with graceful fallbacks.
    """
    from homie_core.memory.working import WorkingMemory
    from homie_core.brain.tool_registry import ToolRegistry

    wm = WorkingMemory()
    em = None
    sm = None
    rag = None
    plugin_mgr = None
    tool_registry = ToolRegistry()

    storage_path = Path(cfg.storage.path).expanduser()

    # Initialize database + semantic memory
    try:
        from homie_core.storage.database import Database
        from homie_core.memory.semantic import SemanticMemory
        db = Database(storage_path / cfg.storage.db_name)
        db.initialize()
        sm = SemanticMemory(db)
        print("  [+] Semantic memory loaded")
    except Exception as e:
        print(f"  [-] Semantic memory unavailable: {e}")
        db = None

    # Initialize vector store + episodic memory
    try:
        from homie_core.storage.vectors import VectorStore
        from homie_core.memory.episodic import EpisodicMemory
        vs = VectorStore(storage_path / cfg.storage.chroma_dir)
        vs.initialize()
        em = EpisodicMemory(db, vs) if db else None
        if em:
            print("  [+] Episodic memory loaded")

        # Initialize RAG pipeline with vector store
        try:
            from homie_core.rag.pipeline import RagPipeline
            rag = RagPipeline(vector_store=vs)
            print("  [+] RAG pipeline ready")
        except Exception as e:
            print(f"  [-] RAG unavailable: {e}")
    except Exception as e:
        print(f"  [-] Vector store unavailable: {e}")

    # Load plugins and register as tools
    try:
        from homie_core.plugins.manager import PluginManager
        plugin_mgr = PluginManager()

        # Load built-in plugins
        builtin_dir = Path(__file__).parent / "plugins"
        if builtin_dir.exists():
            loaded = plugin_mgr.load_from_directory(builtin_dir)
            if loaded > 0:
                print(f"  [+] {loaded} plugins discovered")

        # Enable plugins from config
        enabled_plugins = getattr(cfg, 'plugins', None)
        if enabled_plugins and hasattr(enabled_plugins, 'enabled'):
            for name in enabled_plugins.enabled:
                if plugin_mgr.enable(name):
                    print(f"      Enabled: {name}")

        # Register plugin tools with tool registry
        _register_plugin_tools(plugin_mgr, tool_registry)
    except Exception as e:
        print(f"  [-] Plugin system unavailable: {e}")

    # Register comprehensive built-in tools (memory, system, files, git, clipboard, etc.)
    try:
        from homie_core.brain.builtin_tools import register_builtin_tools
        register_builtin_tools(
            registry=tool_registry,
            working_memory=wm,
            semantic_memory=sm,
            episodic_memory=em,
            plugin_manager=plugin_mgr,
            storage_path=str(storage_path),
            rag_pipeline=rag,
        )
        tool_count = len(tool_registry.list_tools())
        print(f"  [+] {tool_count} tools registered")
    except Exception as e:
        print(f"  [-] Built-in tools unavailable: {e}")
        # Fall back to meta-tools only
        _register_meta_tools(tool_registry, sm, em, wm)

    return wm, em, sm, tool_registry, rag, plugin_mgr


def _register_plugin_tools(plugin_mgr, tool_registry):
    """Convert plugins into callable tools for the agentic loop.

    Creates query-based tools from each plugin's known intents.
    Plugins are auto-enabled for tool registration.
    """
    from homie_core.brain.tool_registry import Tool, ToolParam
    import json

    # Known plugin intents (maps plugin name → {intent: description})
    _PLUGIN_INTENTS = {
        "git": {
            "status": "Get current git status (changed/staged files)",
            "log": "Get recent git commit log",
            "branch": "Get current git branch name",
            "diff": "Get git diff summary of changes",
        },
        "terminal": {
            "history": "Get recent shell command history",
        },
        "system": {
            "status": "Get system status (CPU, RAM, disk usage)",
            "processes": "List top running processes by resource usage",
        },
        "clipboard": {
            "history": "Get recent clipboard history",
            "search": "Search clipboard history for text",
        },
        "notes": {
            "search": "Search local markdown/text notes",
            "recent": "List recently modified notes",
        },
    }

    for plugin_info in plugin_mgr.list_plugins():
        p_name = plugin_info["name"]
        intents = _PLUGIN_INTENTS.get(p_name, {})

        # Auto-enable plugins so they're available as tools
        if not plugin_info["enabled"]:
            plugin_mgr.enable(p_name)

        for intent, desc in intents.items():
            def make_executor(plugin_name, intent_name):
                def executor(**kwargs):
                    result = plugin_mgr.query_plugin(plugin_name, intent_name, kwargs)
                    if result.success:
                        data = result.data
                        if isinstance(data, (dict, list)):
                            return json.dumps(data, indent=2, default=str)[:2000]
                        return str(data)[:2000]
                    return f"Error: {result.error}"
                return executor

            # Build params based on intent
            params = []
            if intent in ("search",):
                params.append(ToolParam(name="query", description="Search query", type="string"))
            if intent in ("history", "log", "recent", "processes"):
                params.append(ToolParam(name="n", description="Number of results", type="string", required=False, default="10"))

            tool = Tool(
                name=f"{p_name}_{intent}",
                description=desc,
                params=params,
                execute=make_executor(p_name, intent),
                category=p_name,
            )
            tool_registry.register(tool)


def _register_meta_tools(tool_registry, sm, em, wm):
    """Register built-in meta-tools for memory management."""
    from homie_core.brain.tool_registry import Tool, ToolParam

    # Tool: remember a fact
    def remember_fact(fact: str = "", confidence: str = "0.7", **kwargs):
        if not sm or not fact:
            return "Cannot store facts — semantic memory not available."
        conf = float(confidence) if confidence else 0.7
        sm.learn(fact, confidence=conf, tags=["user_explicit"])
        return f"Remembered: {fact}"

    tool_registry.register(Tool(
        name="remember",
        description="Store a fact about the user for future reference",
        params=[
            ToolParam(name="fact", description="The fact to remember", type="string"),
            ToolParam(name="confidence", description="Confidence level 0-1", type="string", required=False, default="0.7"),
        ],
        execute=remember_fact,
        category="memory",
    ))

    # Tool: recall facts
    def recall_facts(query: str = "", **kwargs):
        if not sm:
            return "Semantic memory not available."
        facts = sm.get_facts(min_confidence=0.3)
        if not facts:
            return "No facts stored yet."
        if query:
            from homie_core.brain.cognitive_arch import _tf_idf_relevance
            texts = [f["fact"] for f in facts]
            scores = _tf_idf_relevance(query, texts)
            scored = sorted(zip(facts, scores), key=lambda x: x[1], reverse=True)
            relevant = [f for f, s in scored[:5] if s > 0.05]
            if relevant:
                return "\n".join(f"- {f['fact']} (confidence: {f['confidence']:.0%})" for f in relevant)
            return "No relevant facts found."
        return "\n".join(f"- {f['fact']} (confidence: {f['confidence']:.0%})" for f in facts[:10])

    tool_registry.register(Tool(
        name="recall",
        description="Search stored facts and memories about the user",
        params=[
            ToolParam(name="query", description="What to search for", type="string", required=False, default=""),
        ],
        execute=recall_facts,
        category="memory",
    ))

    # Tool: forget a fact
    def forget_fact(topic: str = "", **kwargs):
        if not sm or not topic:
            return "Cannot forget — semantic memory not available or no topic given."
        sm.forget_topic(topic)
        return f"Forgotten facts about: {topic}"

    tool_registry.register(Tool(
        name="forget",
        description="Forget stored facts about a topic (privacy)",
        params=[
            ToolParam(name="topic", description="The topic to forget", type="string"),
        ],
        execute=forget_fact,
        category="memory",
    ))


def _init_watchdog(cfg):
    """Initialize the self-healing watchdog."""
    from homie_core.self_healing.watchdog import HealthWatchdog
    from homie_core.self_healing.probes.config_probe import ConfigProbe

    if not getattr(cfg, 'self_healing', None) or not cfg.self_healing.enabled:
        return None

    storage_path = Path(cfg.storage.path)
    wd = HealthWatchdog(
        db_path=storage_path / "health.db",
        probe_interval=cfg.self_healing.probe_interval,
    )

    # Register config probe (always available)
    config_path = Path("homie.config.yaml")
    if config_path.exists():
        wd.register_probe(ConfigProbe(config=cfg, config_path=config_path))

    return wd


def _init_adaptive_learner(cfg):
    """Initialize the adaptive learning engine."""
    from homie_core.adaptive_learning.learner import AdaptiveLearner

    if not getattr(cfg, 'adaptive_learning', None) or not cfg.adaptive_learning.enabled:
        return None

    storage_path = Path(cfg.storage.path)
    al_cfg = cfg.adaptive_learning
    learner = AdaptiveLearner(
        db_path=storage_path / "learning.db",
        learning_rate_explicit=al_cfg.preference.learning_rate_explicit,
        learning_rate_implicit=al_cfg.preference.learning_rate_implicit,
        cache_max_entries=al_cfg.performance.cache_max_entries,
        cache_ttl=al_cfg.performance.cache_ttl_default,
    )
    return learner


# ---------------------------------------------------------------------------
# CLI entry point (slim — all interactive commands live in Console)
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="homie", description="Homie AI — Local Personal Assistant")
    parser.add_argument("--version", action="version", version="homie-ai 0.1.0")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice pipeline")
    parser.add_argument("--no-tray", action="store_true", help="Disable system tray")

    subparsers = parser.add_subparsers(dest="command")
    start_parser = subparsers.add_parser("start", help="Start the assistant (default)")
    start_parser.add_argument("--config", type=str, help="Path to config file", dest="start_config")
    start_parser.add_argument("--no-voice", action="store_true", dest="start_no_voice")
    start_parser.add_argument("--no-tray", action="store_true", dest="start_no_tray")

    daemon_parser = subparsers.add_parser("daemon", help="Run as background daemon")
    daemon_parser.add_argument("--config", type=str, help="Path to config file", dest="daemon_config")
    daemon_parser.add_argument("--headless", action="store_true", help="No console output, log to file")

    return parser


def main(argv: list[str] | None = None):
    parser = create_parser()
    args = parser.parse_args(argv)

    # Daemon mode — run background service
    if getattr(args, "command", None) == "daemon":
        from homie_app.daemon import run_daemon
        run_daemon(
            config_path=getattr(args, "daemon_config", None),
            headless=getattr(args, "headless", False),
        )
        return

    config_path = getattr(args, "start_config", None) or getattr(args, "config", None)
    no_voice = getattr(args, "start_no_voice", False) or getattr(args, "no_voice", False)
    no_tray = getattr(args, "start_no_tray", False) or getattr(args, "no_tray", False)

    from homie_core.config import load_config
    cfg = load_config(config_path)

    from homie_app.console.console import Console
    console = Console(
        config=cfg,
        config_path=config_path,
        no_voice=no_voice,
        no_tray=no_tray,
    )
    console.run()


if __name__ == "__main__":
    main()
