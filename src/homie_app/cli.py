from __future__ import annotations

import argparse
import os
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
    add.add_argument("--format", type=str, default="gguf", choices=["gguf", "safetensors", "cloud", "hf"])
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

    # homie insights
    insights_parser = subparsers.add_parser("insights", help="Show usage analytics and stats")
    insights_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    insights_parser.add_argument("--compact", action="store_true", help="Show compact summary")

    # homie schedule
    schedule_parser = subparsers.add_parser("schedule", help="Manage scheduled tasks")
    schedule_sub = schedule_parser.add_subparsers(dest="schedule_command")
    schedule_sub.add_parser("list", help="List scheduled jobs")
    sched_add = schedule_sub.add_parser("add", help="Add a scheduled job")
    sched_add.add_argument("name", type=str, help="Job name")
    sched_add.add_argument("schedule", type=str, help="Schedule: '30m', 'every 2h', '0 9 * * *', 'daily'")
    sched_add.add_argument("prompt", type=str, help="What Homie should do")
    sched_add.add_argument("--max-repeats", type=int, default=None, help="Max repetitions (None=infinite)")
    sched_rm = schedule_sub.add_parser("remove", help="Remove a scheduled job")
    sched_rm.add_argument("job_id", type=str, help="Job ID to remove")

    # homie skills
    skills_parser = subparsers.add_parser("skills", help="List available skills")

    # homie connections
    sub_connections = subparsers.add_parser("connections", help="List provider connections")
    sub_connections.set_defaults(func=cmd_connections)

    # homie consent-log
    sub_consent = subparsers.add_parser("consent-log", help="Show consent audit trail")
    sub_consent.add_argument("provider", help="Provider name (gmail, slack, etc.)")
    sub_consent.set_defaults(func=cmd_consent_log)

    # homie vault
    sub_vault = subparsers.add_parser("vault", help="Vault management")
    vault_sub = sub_vault.add_subparsers(dest="vault_cmd")
    vault_status = vault_sub.add_parser("status", help="Show vault health")
    vault_status.set_defaults(func=cmd_vault_status)

    # homie connect / disconnect
    sub_connect = subparsers.add_parser("connect", help="Connect a provider")
    sub_connect.add_argument("provider", help="Provider name (gmail, slack, etc.)")
    sub_connect.set_defaults(func=cmd_connect)

    sub_disconnect = subparsers.add_parser("disconnect", help="Disconnect a provider")
    sub_disconnect.add_argument("provider", help="Provider name")
    sub_disconnect.set_defaults(func=cmd_disconnect)

    # homie email
    email_parser = subparsers.add_parser("email", help="Email management")
    email_sub = email_parser.add_subparsers(dest="email_command")
    email_summary_parser = email_sub.add_parser("summary", help="Email summary")
    email_summary_parser.add_argument("--days", type=int, default=1, help="Days to summarize")
    email_sub.add_parser("sync", help="Force sync now")
    email_sub.add_parser("config", help="Show email settings")

    # homie folder
    folder_parser = subparsers.add_parser("folder", help="Folder awareness")
    folder_sub = folder_parser.add_subparsers(dest="folder_command")
    fw = folder_sub.add_parser("watch", help="Add a folder to watch")
    fw.add_argument("path", type=str, help="Directory path to watch")
    fw.add_argument("--label", type=str, default=None, help="Friendly label")
    fw.add_argument("--interval", type=int, default=300, help="Scan interval in seconds")
    folder_sub.add_parser("list", help="List watched folders")
    fs = folder_sub.add_parser("scan", help="Force immediate scan")
    fs.add_argument("--path", type=str, default=None, help="Scan specific folder only")
    fu = folder_sub.add_parser("unwatch", help="Remove a folder watch")
    fu.add_argument("path", type=str, help="Directory path to unwatch")

    # homie social
    social_parser = subparsers.add_parser("social", help="Social messaging")
    social_sub = social_parser.add_subparsers(dest="social_command")
    social_sub.add_parser("channels", help="List channels")
    sr = social_sub.add_parser("recent", help="Recent messages from a channel")
    sr.add_argument("channel", type=str, help="Channel ID")
    sr.add_argument("--limit", type=int, default=20)

    # homie sm (social media profiles)
    sm_parser = subparsers.add_parser("sm", help="Social media operations")
    sm_sub = sm_parser.add_subparsers(dest="sm_command")
    sm_feed = sm_sub.add_parser("feed", help="Get social media feed")
    sm_feed.add_argument("--platform", default="all", help="Platform name or 'all'")
    sm_feed.add_argument("--limit", type=int, default=20)
    sm_profile = sm_sub.add_parser("profile", help="Get profile info")
    sm_profile.add_argument("--platform", required=True)
    sm_profile.add_argument("--username", default="")
    sm_sub.add_parser("scan", help="Full profile scan + intelligence")
    sm_publish = sm_sub.add_parser("publish", help="Publish a post")
    sm_publish.add_argument("platform")
    sm_publish.add_argument("content")
    sm_dms = sm_sub.add_parser("dms", help="View DM conversations")
    sm_dms.add_argument("platform")
    sm_dms.add_argument("--conversation", default="")
    sm_send = sm_sub.add_parser("send-dm", help="Send a DM")
    sm_send.add_argument("platform")
    sm_send.add_argument("recipient")
    sm_send.add_argument("text")

    # homie browser
    browser_parser = subparsers.add_parser("browser", help="Browser history")
    browser_sub = browser_parser.add_subparsers(dest="browser_command")
    br_enable = browser_sub.add_parser("enable", help="Enable browser history tracking")
    br_enable.add_argument("--browsers", default="chrome")
    browser_sub.add_parser("disable", help="Disable browser history tracking")
    br_config = browser_sub.add_parser("config", help="Configure browser history")
    br_config.add_argument("--exclude", default="")
    br_config.add_argument("--retention", type=int, default=0)
    br_history = browser_sub.add_parser("history", help="View browsing history")
    br_history.add_argument("--limit", type=int, default=50)
    br_history.add_argument("--domain", default="")
    browser_sub.add_parser("scan", help="Full history scan")
    browser_sub.add_parser("patterns", help="Browsing patterns analysis")

    return parser


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


def _handle_meta_command(command: str, brain, wm, sm, em, cfg) -> str | None:
    """Handle slash commands. Returns response string, or None if not a command."""
    cmd = command.strip().lower()

    if cmd == "/status":
        lines = ["**Homie Status**"]
        lines.append(f"  Memory: {len(wm.get_conversation())} messages this session")
        if sm:
            facts = sm.get_facts(min_confidence=0.0)
            lines.append(f"  Facts stored: {len(facts)}")
        if em:
            lines.append("  Episodic memory: active")
        lines.append(f"  User: {cfg.user_name or 'Unknown'}")
        return "\n".join(lines)

    if cmd == "/learn":
        stats = brain._cognitive._learning.get_session_stats()
        lines = ["**Session Learning Stats**"]
        lines.append(f"  Interactions: {stats['interactions']}")
        lines.append(f"  Facts learned: {stats['facts_learned']}")
        if stats['facts']:
            for f in stats['facts']:
                lines.append(f"    - {f}")
        return "\n".join(lines)

    if cmd == "/facts":
        if not sm:
            return "Semantic memory not available."
        facts = sm.get_facts(min_confidence=0.3)
        if not facts:
            return "No facts stored yet. Chat with me and I'll learn about you!"
        lines = ["**What I know about you:**"]
        for f in facts[:15]:
            lines.append(f"  - {f['fact']} ({f['confidence']:.0%} confident)")
        return "\n".join(lines)

    if cmd.startswith("/remember "):
        fact = command[10:].strip()
        if sm and fact:
            sm.learn(fact, confidence=0.9, tags=["user_explicit"])
            return f"Got it, I'll remember: {fact}"
        return "Could not store that — semantic memory not available."

    if cmd.startswith("/forget "):
        topic = command[8:].strip()
        if sm and topic:
            sm.forget_topic(topic)
            return f"Forgotten everything about: {topic}"
        return "Could not forget — semantic memory not available."

    if cmd == "/clear":
        wm.clear()
        return "Conversation cleared. Fresh start!"

    if cmd == "/insights":
        try:
            from homie_core.analytics.insights import InsightsEngine
            engine = InsightsEngine(Path(cfg.storage.path).expanduser())
            insights = engine.generate_insights(days=30)
            return engine.format_terminal(insights)
        except Exception as e:
            return f"Could not generate insights: {e}"

    if cmd.startswith("/schedule "):
        parts = command[10:].strip().split(maxsplit=2)
        if len(parts) < 3:
            return "Usage: /schedule <name> <schedule> <prompt>\n  Example: /schedule remind_break every 2h Take a break and stretch"
        try:
            from homie_core.scheduler.cron import JobStore
            job_store = JobStore()
            job = job_store.create_job(name=parts[0], prompt=parts[2], schedule=parts[1])
            return f"Scheduled: '{job.name}' ({parts[1]}) — next run: {job.next_run}"
        except Exception as e:
            return f"Could not schedule: {e}"

    if cmd == "/skills":
        try:
            from homie_core.skills.loader import SkillLoader
            loader = SkillLoader()
            skills = loader.scan()
            if not skills:
                return "No skills installed. Drop SKILL.md files into ~/.homie/skills/"
            return loader.build_skills_index()
        except Exception as e:
            return f"Could not load skills: {e}"

    if cmd == "/connections":
        try:
            from homie_core.vault.secure_vault import SecureVault
            vault = SecureVault()
            vault.unlock()
            connections = vault.get_all_connections()
            vault.lock()
            if not connections:
                return "No connections configured."
            lines = ["**Connections:**"]
            for c in connections:
                icon = "+" if c.connected else "-"
                lines.append(f"  [{icon}] {c.provider}: {c.display_label or 'no label'}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not check connections: {e}"

    if cmd.startswith("/consent-log "):
        provider = command[13:].strip()
        if not provider:
            return "Usage: /consent-log <provider>"
        try:
            from datetime import datetime
            from homie_core.vault.secure_vault import SecureVault
            vault = SecureVault()
            vault.unlock()
            history = vault.get_consent_history(provider)
            vault.lock()
            if not history:
                return f"No consent history for '{provider}'."
            lines = [f"**Consent log for {provider}:**"]
            for entry in history:
                dt = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M")
                lines.append(f"  {dt}  {entry.action}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not check consent log: {e}"

    if cmd == "/vault":
        try:
            from homie_core.vault.secure_vault import SecureVault
            vault = SecureVault()
            vault.unlock()
            connections = vault.get_all_connections()
            active = sum(1 for c in connections if c.connected)
            has_pw = vault.has_password
            vault.lock()
            return (
                f"**Vault Status:**\n"
                f"  Connections: {active} active / {len(connections)} total\n"
                f"  Password: {'set' if has_pw else 'not set'}"
            )
        except Exception as e:
            return f"Could not check vault: {e}"

    if cmd.startswith("/connect "):
        provider = command[9:].strip()
        if provider == "gmail":
            return "Use the CLI command: homie connect gmail"
        return f"Provider '{provider}' not yet supported. Available: gmail"

    elif cmd.startswith("/email"):
        parts = cmd.split()
        if len(parts) > 1 and parts[1] == "summary":
            return "Use the CLI command: homie email summary"
        elif len(parts) > 1 and parts[1] == "sync":
            return "Use the CLI command: homie email sync"
        else:
            return "Email commands: /email summary, /email sync, or use CLI: homie email {summary|sync|config}"

    if cmd.startswith("/disconnect "):
        provider = command[12:].strip()
        return f"To disconnect {provider}, use the CLI: homie disconnect {provider}"

    if cmd == "/help":
        return (
            "**Homie Commands:**\n"
            "  /status      — Show system status\n"
            "  /facts       — Show what I know about you\n"
            "  /learn       — Show what I learned this session\n"
            "  /remember    — Store a fact (e.g., /remember I prefer dark mode)\n"
            "  /forget      — Forget a topic (e.g., /forget work)\n"
            "  /insights    — Show usage analytics (sessions, topics, streaks)\n"
            "  /schedule    — Create scheduled task (e.g., /schedule name every_2h prompt)\n"
            "  /skills      — List installed skills\n"
            "  /connections — Show connected providers\n"
            "  /consent-log — Show consent audit trail (e.g., /consent-log gmail)\n"
            "  /vault       — Show vault status\n"
            "  /connect     — Connect a provider (e.g., /connect gmail)\n"
            "  /email       — Email commands (summary, sync)\n"
            "  /disconnect  — Disconnect a provider (e.g., /disconnect gmail)\n"
            "  /clear       — Clear conversation (fresh start)\n"
            "  /help        — Show this help\n"
            "  quit         — Exit chat"
        )

    return None


def cmd_chat(args, config=None):
    from homie_core.config import load_config
    from homie_core.brain.orchestrator import BrainOrchestrator
    from homie_app.prompts.system import build_system_prompt

    cfg = config or load_config(args.config if hasattr(args, 'config') else None)
    print("=" * 50)
    print("  Homie AI v0.1.0 — Interactive Chat")
    print("=" * 50)

    print("\n[Loading model...]")
    engine, entry = _load_model_engine(cfg)
    if not engine:
        print("  No model found. Run 'homie init' first, or 'homie model add <path>'.")
        return

    print("\n[Initializing intelligence...]")
    wm, em, sm, tool_registry, rag, plugin_mgr = _init_intelligence_stack(cfg)

    # Build brain with full intelligence stack
    brain = BrainOrchestrator(
        model_engine=engine,
        working_memory=wm,
        episodic_memory=em,
        semantic_memory=sm,
        tool_registry=tool_registry if tool_registry.list_tools() else None,
        rag_pipeline=rag,
    )

    # Build dynamic system prompt with user context
    user_name = cfg.user_name or "User"
    known_facts = []
    if sm:
        try:
            facts = sm.get_facts(min_confidence=0.5)
            known_facts = [f["fact"] for f in facts[:10]]
        except Exception:
            pass

    system_prompt = build_system_prompt(
        user_name=user_name,
        known_facts=known_facts if known_facts else None,
    )
    brain.set_system_prompt(system_prompt)

    print(f"\nHey{' ' + user_name if user_name != 'User' else ''}! I'm ready. Type /help for commands or just start chatting.\n")

    while True:
        try:
            user_input = input(f"{user_name}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break
        if user_input.lower() in ("quit", "exit", ":q"):
            break
        if not user_input:
            continue

        # Handle meta-commands
        if user_input.startswith("/"):
            result = _handle_meta_command(user_input, brain, wm, sm, em, cfg)
            if result is not None:
                print(f"Homie> {result}\n")
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

    # Consolidate session into episodic memory before exiting
    print("[Saving session...]")
    try:
        summary = brain.consolidate_session()
        if summary:
            print(f"  Session saved: {summary}")
        else:
            print("  Nothing to save.")
    except Exception:
        print("  Could not save session.")

    engine.unload()
    print("Goodbye!")


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


def cmd_insights(args, config=None):
    from homie_core.config import load_config
    from homie_core.analytics.insights import InsightsEngine
    cfg = config or load_config()
    engine = InsightsEngine(Path(cfg.storage.path).expanduser())
    insights = engine.generate_insights(days=args.days)
    if getattr(args, 'compact', False):
        print(engine.format_compact(insights))
    else:
        print(engine.format_terminal(insights))


def cmd_schedule(args, config=None):
    from homie_core.config import load_config
    from homie_core.scheduler.cron import JobStore
    cfg = config or load_config()
    storage = Path(cfg.storage.path).expanduser()
    job_store = JobStore(path=storage / "scheduler" / "jobs.json")

    if args.schedule_command == "list":
        jobs = job_store.list_jobs()
        if not jobs:
            print("No scheduled jobs.")
            return
        for j in jobs:
            status = "enabled" if j.enabled else "disabled"
            print(f"  [{j.id[:8]}] {j.name} ({status})")
            print(f"    Schedule: {j.schedule}  Next: {j.next_run or 'N/A'}")
            print(f"    Prompt: {j.prompt[:80]}")
    elif args.schedule_command == "add":
        job = job_store.create_job(
            name=args.name,
            prompt=args.prompt,
            schedule=args.schedule,
            max_repeats=args.max_repeats,
        )
        print(f"Created job '{job.name}' (id: {job.id[:8]})")
        print(f"  Next run: {job.next_run}")
    elif args.schedule_command == "remove":
        if job_store.delete_job(args.job_id):
            print(f"Removed job {args.job_id}")
        else:
            print(f"Job not found: {args.job_id}")
    else:
        print("Usage: homie schedule {list|add|remove}")


def cmd_skills(args, config=None):
    from homie_core.skills.loader import SkillLoader
    loader = SkillLoader()
    skills = loader.scan()
    if not skills:
        print("No skills installed.")
        print("Drop SKILL.md files into ~/.homie/skills/ to add custom skills.")
        return
    print(loader.build_skills_index())


def cmd_connections(args, config=None):
    """List all connection statuses."""
    from homie_core.vault.secure_vault import SecureVault
    vault = SecureVault()
    try:
        vault.unlock()
        connections = vault.get_all_connections()
        if not connections:
            print("No connections configured. Use 'homie connect <provider>' to add one.")
            return
        for c in connections:
            icon = "+" if c.connected else "-"
            mode = f" ({c.connection_mode})" if c.connected else ""
            print(f"  [{icon}] {c.provider}: {c.display_label or 'no label'}{mode}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vault.lock()


def cmd_consent_log(args, config=None):
    """Show consent audit trail for a provider."""
    from datetime import datetime
    from homie_core.vault.secure_vault import SecureVault
    provider = args.provider
    vault = SecureVault()
    try:
        vault.unlock()
        history = vault.get_consent_history(provider)
        if not history:
            print(f"No consent history for '{provider}'.")
            return
        for entry in history:
            dt = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M")
            scopes_str = ", ".join(entry.scopes) if entry.scopes else ""
            reason_str = f"  reason: {entry.reason}" if entry.reason else ""
            print(f"  {dt}  {entry.action:<15} {scopes_str}{reason_str}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vault.lock()


def cmd_vault_status(args, config=None):
    """Show vault health and statistics."""
    from pathlib import Path
    from homie_core.vault.secure_vault import SecureVault
    vault = SecureVault()
    try:
        vault.unlock()
        connections = vault.get_all_connections()
        active = sum(1 for c in connections if c.connected)
        vault_path = vault._storage_dir
        vault_size = (vault_path / "vault.db").stat().st_size if (vault_path / "vault.db").exists() else 0
        cache_size = (vault_path / "cache.db").stat().st_size if (vault_path / "cache.db").exists() else 0

        print(f"  Vault DB: {vault_size / 1024:.1f} KB")
        print(f"  Cache DB: {cache_size / 1024:.1f} KB")
        print(f"  Connections: {active} active / {len(connections)} total")
        print(f"  Password: {'set' if vault.has_password else 'not set'}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vault.lock()


def cmd_connect(args, config=None):
    """Connect a provider via OAuth or API key."""
    from homie_core.config import load_config
    cfg = config or load_config()
    provider = args.provider
    if provider == "slack":
        _connect_slack(args, config=cfg)
        return
    if provider == "blog":
        _connect_blog(args, config=cfg)
        return
    _SM_PLATFORMS = {
        "twitter": ("https://twitter.com/i/oauth2/authorize",
                     "https://api.twitter.com/2/oauth2/token",
                     ["tweet.read", "tweet.write", "users.read", "dm.read", "dm.write", "offline.access"], 8551),
        "reddit": ("https://www.reddit.com/api/v1/authorize",
                    "https://www.reddit.com/api/v1/access_token",
                    ["identity", "read", "submit", "privatemessages"], 8552),
        "linkedin": ("https://www.linkedin.com/oauth/v2/authorization",
                      "https://www.linkedin.com/oauth/v2/accessToken",
                      ["r_liteprofile", "r_emailaddress", "w_member_social"], 8553),
        "facebook": ("https://www.facebook.com/v19.0/dialog/oauth",
                      "https://graph.facebook.com/v19.0/oauth/access_token",
                      ["public_profile", "email", "pages_read_engagement", "pages_manage_posts"], 8554),
    }
    if provider in _SM_PLATFORMS:
        auth_url, token_url, scopes, port = _SM_PLATFORMS[provider]
        _connect_social_media(args, config=cfg, platform=provider,
                              auth_url=auth_url, token_url=token_url,
                              scopes=scopes, port=port)
        return
    if provider != "gmail":
        print(f"Provider '{provider}' not yet supported. Available: gmail, slack, twitter, reddit, linkedin, facebook, blog")
        return

    from homie_core.email.oauth import GmailOAuth, GMAIL_SCOPES
    from homie_core.vault.secure_vault import SecureVault
    import time

    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()

    # Check for existing OAuth client credentials
    client_cred = vault.get_credential("gmail", account_id="oauth_client")
    if client_cred:
        client_id = client_cred.access_token
        client_secret = client_cred.refresh_token
    else:
        print("\nGmail OAuth Setup")
        print("=" * 40)
        print("You need a Google Cloud OAuth client ID.")
        print("1. Go to https://console.cloud.google.com/apis/credentials")
        print("2. Create an OAuth 2.0 Client ID (Desktop app)")
        print("3. Enable the Gmail API\n")
        client_id = input("Client ID: ").strip()
        client_secret = input("Client Secret: ").strip()
        if not client_id or not client_secret:
            print("Cancelled.")
            vault.lock()
            return

        vault.store_credential(
            provider="gmail", account_id="oauth_client",
            token_type="oauth_client",
            access_token=client_id,
            refresh_token=client_secret,
            scopes=GMAIL_SCOPES,
        )

    oauth = GmailOAuth(client_id=client_id, client_secret=client_secret)

    print("\nOpening browser for Google authorization...")
    auth_url = oauth.get_auth_url(use_local_server=True)

    import webbrowser
    webbrowser.open(auth_url)

    code = oauth.wait_for_redirect(timeout=120)

    if not code:
        # Fallback: try alternate port
        print("\nPrimary port unavailable. Trying alternate port...")
        alt_url = oauth.get_auth_url(alt_port=True)
        webbrowser.open(alt_url)
        code = oauth.wait_for_redirect_alt(timeout=120)
        if not code:
            print("\nAuthorization failed. Please check your firewall settings and try again.")
            vault.lock()
            return
        tokens = oauth.exchange(code, alt_port=True)
    else:
        tokens = oauth.exchange(code, use_local_server=True)

    # Get profile to determine account email
    from types import SimpleNamespace
    from homie_core.email.gmail_provider import GmailProvider
    provider_instance = GmailProvider(account_id="pending")
    temp_cred = SimpleNamespace(
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
        provider="gmail", account_id="pending",
    )
    provider_instance.authenticate(temp_cred, client_id=client_id, client_secret=client_secret)
    profile = provider_instance.get_profile()
    email_addr = profile["emailAddress"]

    vault.store_credential(
        provider="gmail", account_id=email_addr,
        token_type="oauth2",
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token", ""),
        scopes=GMAIL_SCOPES,
        expires_at=time.time() + tokens.get("expires_in", 3600),
    )

    vault.set_connection_status("gmail", connected=True, label=email_addr)
    vault.log_consent("gmail", "connected", scopes=GMAIL_SCOPES)

    print(f"\nConnected: {email_addr}")
    print("Run `homie email sync` to fetch recent emails, or they'll sync automatically.")
    vault.lock()


def cmd_email_summary(args, config=None):
    """Print email summary."""
    from homie_core.config import load_config
    from homie_core.email import EmailService
    from homie_core.vault.secure_vault import SecureVault
    from homie_core.vault.schema import create_cache_db

    cfg = config or load_config()
    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()
    cache_conn = create_cache_db(storage / "cache.db")

    service = EmailService(vault=vault, cache_conn=cache_conn)
    accounts = service.initialize()
    if not accounts:
        print("No email accounts connected. Run: homie connect gmail")
        vault.lock()
        return

    summary = service.get_summary(days=int(getattr(args, "days", 1)))
    print(f"\nEmail Summary (last {getattr(args, 'days', 1)} day(s)):")
    print(f"  Total: {summary['total']}")
    print(f"  Unread: {summary['unread']}")
    if summary['high_priority']:
        print(f"\n  High Priority:")
        for item in summary['high_priority'][:5]:
            print(f"    - {item['sender']}: {item['subject']}")
    vault.lock()


def cmd_email_sync(args, config=None):
    """Force immediate email sync."""
    from homie_core.config import load_config
    from homie_core.email import EmailService
    from homie_core.vault.secure_vault import SecureVault
    from homie_core.vault.schema import create_cache_db

    cfg = config or load_config()
    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()
    cache_conn = create_cache_db(storage / "cache.db")

    service = EmailService(vault=vault, cache_conn=cache_conn)
    accounts = service.initialize()
    if not accounts:
        print("No email accounts connected. Run: homie connect gmail")
        vault.lock()
        return

    print("Syncing...")
    result = service.sync_tick()
    print(result)
    vault.lock()


def cmd_email_config(args, config=None):
    """Show email sync configuration."""
    from homie_core.config import load_config
    from homie_core.vault.schema import create_cache_db

    cfg = config or load_config()
    storage = Path(cfg.storage.path)
    cache_conn = create_cache_db(storage / "cache.db")

    rows = cache_conn.execute("SELECT * FROM email_config").fetchall()
    if not rows:
        print("No email configuration found. Defaults apply (5 min sync, high-priority notifications).")
        return

    for row in rows:
        print(f"\nAccount: {row[0]}")
        print(f"  Check interval: {row[1]}s")
        print(f"  Notify priority: {row[2]}")
        print(f"  Quiet hours: {row[3] or 'none'}-{row[4] or 'none'}")
        print(f"  Auto-trash spam: {'yes' if row[5] else 'no'}")
    cache_conn.close()


def _connect_slack(args, config=None):
    """Connect Slack workspace via OAuth."""
    from homie_core.social.oauth import SlackOAuth, SLACK_SCOPES
    from homie_core.vault.secure_vault import SecureVault
    import time

    cfg = config
    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()

    client_cred = vault.get_credential("slack", account_id="oauth_client")
    if client_cred:
        client_id = client_cred.access_token
        client_secret = client_cred.refresh_token
    else:
        print("\nSlack OAuth Setup")
        print("=" * 40)
        print("You need a Slack App with OAuth configured.")
        print("1. Go to https://api.slack.com/apps")
        print("2. Create a new app (or use existing)")
        print("3. Under OAuth & Permissions, add redirect URL: http://localhost:8549/callback")
        print("4. Add the required Bot Token Scopes\n")
        client_id = input("Client ID: ").strip()
        client_secret = input("Client Secret: ").strip()
        if not client_id or not client_secret:
            print("Cancelled.")
            vault.lock()
            return

        vault.store_credential(
            provider="slack", account_id="oauth_client",
            token_type="oauth_client",
            access_token=client_id,
            refresh_token=client_secret,
            scopes=SLACK_SCOPES,
        )

    oauth = SlackOAuth(client_id=client_id, client_secret=client_secret)

    print("\nOpening browser for Slack authorization...")
    auth_url = oauth.get_auth_url()

    import webbrowser
    webbrowser.open(auth_url)

    code = oauth.wait_for_redirect(timeout=120)
    if not code:
        print("\nAuthorization timed out or port unavailable.")
        vault.lock()
        return
    tokens = oauth.exchange(code)

    team_name = tokens.get("team", {}).get("name", "unknown")
    team_id = tokens.get("team", {}).get("id", "unknown")
    bot_token = tokens.get("access_token", "")

    vault.store_credential(
        provider="slack", account_id=team_id,
        token_type="oauth2",
        access_token=bot_token,
        refresh_token="",
        scopes=SLACK_SCOPES,
    )

    vault.set_connection_status("slack", connected=True, label=team_name)
    vault.log_consent("slack", "connected", scopes=SLACK_SCOPES)

    print(f"\nConnected: {team_name} ({team_id})")
    print("Run `homie social channels` to see available channels.")
    vault.lock()


def cmd_folder(args, config=None):
    """Folder awareness commands."""
    from homie_core.config import load_config
    from homie_core.vault.schema import create_cache_db

    cfg = config or load_config()
    storage = Path(cfg.storage.path)
    cache_conn = create_cache_db(storage / "cache.db")

    from homie_core.folders import FolderService
    service = FolderService(cache_conn=cache_conn)

    sub = args.folder_command
    if sub == "watch":
        target = Path(args.path).resolve()
        if not target.is_dir():
            print(f"Error: '{target}' is not a directory.")
            return
        service.add_watch(str(target), label=args.label, scan_interval=args.interval)
        print(f"Watching: {target} (interval: {args.interval}s)")
    elif sub == "list":
        watches = service.list_watches()
        if not watches:
            print("No folders watched. Use: homie folder watch <path>")
        else:
            for w in watches:
                label = f" ({w['label']})" if w.get("label") else ""
                print(f"  {w['path']}{label}  [{w['file_count']} files, every {w['scan_interval']}s]")
    elif sub == "scan":
        result = service.scan_tick()
        print(f"Scan: {result}")
    elif sub == "unwatch":
        target = Path(args.path).resolve()
        removed = service.remove_watch(str(target))
        if removed:
            print(f"Unwatched: {target}")
        else:
            print(f"Not found: {target}")
    else:
        print("Usage: homie folder {watch|list|scan|unwatch}")
    cache_conn.close()


def _connect_social_media(args, config=None, platform="", auth_url="", token_url="",
                          scopes=None, port=8551):
    """Generic OAuth flow for social media platforms."""
    import webbrowser
    from homie_core.social_media.oauth import SocialMediaOAuth
    from homie_core.vault.secure_vault import SecureVault

    cfg = config
    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()

    client_cred = vault.get_credential(platform, account_id="oauth_client")
    if client_cred:
        client_id = client_cred.access_token
        client_secret = client_cred.refresh_token
    else:
        print(f"\n--- {platform.title()} OAuth Setup ---")
        print(f"Create an app at the {platform.title()} developer portal.")
        print(f"Set redirect URI to: http://localhost:{port}/callback\n")
        client_id = input("Client ID: ").strip()
        client_secret = input("Client Secret: ").strip()
        if not client_id or not client_secret:
            print("Aborted.")
            vault.lock()
            return
        vault.store_credential(
            provider=platform, account_id="oauth_client",
            token_type="oauth_client",
            access_token=client_id, refresh_token=client_secret,
            scopes=scopes or [],
        )

    oauth = SocialMediaOAuth(
        platform=platform, client_id=client_id, client_secret=client_secret,
        auth_url=auth_url, token_url=token_url,
        scopes=scopes or [], redirect_port=port,
    )

    url = oauth.get_auth_url()
    print(f"\nOpening browser for {platform.title()} authorization...")
    webbrowser.open(url)

    print(f"Waiting for redirect on port {port} (timeout: 120s)...")
    code = oauth.wait_for_redirect(timeout=120)
    if not code:
        print("Timed out waiting for authorization.")
        vault.lock()
        return

    print("Exchanging code for token...")
    tokens = oauth.exchange(code)
    access_token = tokens.get("access_token", "")
    refresh_token = tokens.get("refresh_token", "")

    if not access_token:
        print("Failed to get access token.")
        vault.lock()
        return

    account_id = tokens.get("user_id", tokens.get("name", f"{platform}_user"))
    vault.store_credential(
        provider=platform, account_id=account_id,
        token_type="oauth2",
        access_token=access_token, refresh_token=refresh_token,
        scopes=scopes or [],
    )
    vault.set_connection_status(platform, connected=True, label=account_id)
    vault.log_consent(platform, "connected", scopes=scopes or [])
    vault.lock()
    print(f"\n{platform.title()} connected successfully!")


def _connect_blog(args, config=None):
    """Connect a blog via RSS/Atom feed URL."""
    from homie_core.vault.secure_vault import SecureVault

    cfg = config
    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()

    url = input("Blog RSS/Atom feed URL: ").strip()
    if not url:
        print("Aborted.")
        vault.lock()
        return

    vault.store_credential(
        provider="blog", account_id="blog_feed",
        token_type="rss", access_token=url, refresh_token="",
        scopes=[],
    )
    vault.set_connection_status("blog", connected=True, label=url[:50])
    vault.lock()
    print(f"Blog feed connected: {url}")


def cmd_sm(args, config=None):
    """Social media profile commands."""
    import json
    from homie_core.config import load_config
    from homie_core.vault.secure_vault import SecureVault

    cfg = config or load_config()
    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()

    from homie_core.social_media import SocialMediaService
    service = SocialMediaService(vault=vault)
    service.initialize()

    sub = args.sm_command
    if sub == "feed":
        results = service.get_feed(platform=args.platform, limit=args.limit)
        if not results:
            print("No posts found.")
        for post in results:
            print(f"  [{post.get('platform')}] {post.get('author')}: {post.get('content', '')[:100]}")
    elif sub == "profile":
        info = service.get_profile(args.platform, username=args.username or None)
        print(json.dumps(info, indent=2))
    elif sub == "scan":
        print("Scanning all connected profiles...")
        result = service.scan_profiles()
        print(json.dumps(result, indent=2))
    elif sub == "publish":
        result = service.publish(args.platform, args.content)
        print(json.dumps(result, indent=2))
    elif sub == "dms":
        if args.conversation:
            msgs = service.get_dms(args.platform, args.conversation)
            for m in msgs:
                print(f"  [{m.get('sender')}]: {m.get('content', '')[:100]}")
        else:
            convos = service.get_conversations(args.platform)
            for c in convos:
                print(f"  [{c.get('id')}] {', '.join(c.get('participants', []))}: "
                      f"{c.get('last_message_preview', '')[:50]}")
    elif sub == "send-dm":
        result = service.send_dm(args.platform, args.recipient, args.text)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: homie sm {feed|profile|scan|publish|dms|send-dm}")
    vault.lock()


def cmd_browser(args, config=None):
    """Browser history commands."""
    import json
    from homie_core.config import load_config
    from homie_core.vault.secure_vault import SecureVault

    cfg = config or load_config()
    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()

    from homie_core.browser import BrowserHistoryService
    service = BrowserHistoryService(vault=vault)
    service.initialize()

    sub = args.browser_command
    if sub == "enable":
        browsers = [b.strip() for b in args.browsers.split(",")]
        result = service.configure(enabled=True, browsers=browsers)
        print(f"Browser history enabled for: {', '.join(result['browsers'])}")
    elif sub == "disable":
        service.configure(enabled=False)
        print("Browser history disabled.")
    elif sub == "config":
        kwargs = {}
        if args.exclude:
            kwargs["exclude_domains"] = [d.strip() for d in args.exclude.split(",")]
        if args.retention:
            kwargs["retention_days"] = args.retention
        if kwargs:
            result = service.configure(**kwargs)
        else:
            result = service.get_config()
        print(json.dumps(result, indent=2))
    elif sub == "history":
        entries = service.get_history(limit=args.limit, domain=args.domain or None)
        if not entries:
            print("No history entries found.")
        for e in entries:
            print(f"  [{e.get('browser')}] {e.get('title', '')[:50]} — {e.get('url', '')[:80]}")
    elif sub == "scan":
        print("Scanning browser history...")
        result = service.scan()
        print(json.dumps(result, indent=2))
    elif sub == "patterns":
        result = service.get_patterns()
        print(json.dumps(result, indent=2))
    else:
        print("Usage: homie browser {enable|disable|config|history|scan|patterns}")
    vault.lock()


def cmd_social(args, config=None):
    """Social messaging commands."""
    from homie_core.config import load_config
    from homie_core.vault.secure_vault import SecureVault

    cfg = config or load_config()
    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()

    from homie_core.social import SocialService
    service = SocialService(vault=vault)
    workspaces = service.initialize()

    if not workspaces:
        print("No social platforms connected. Run: homie connect slack")
        vault.lock()
        return

    sub = args.social_command
    if sub == "channels":
        channels = service.list_channels()
        if not channels:
            print("No channels found.")
        else:
            for ch in channels:
                private = " (private)" if ch.get("is_private") else ""
                print(f"  {ch['id']}  #{ch['name']}{private}  ({ch.get('member_count', '?')} members)")
    elif sub == "recent":
        messages = service.get_messages(args.channel, limit=args.limit)
        if not messages:
            print(f"No messages in {args.channel}. Run `homie start` to sync first.")
        else:
            for m in messages:
                print(f"  [{m.get('sender', '?')}] {m.get('text', '')[:120]}")
    else:
        print("Usage: homie social {channels|recent}")
    vault.lock()


def cmd_disconnect(args, config=None):
    """Disconnect a provider with user confirmation."""
    from homie_core.vault.secure_vault import SecureVault
    provider = args.provider
    vault = SecureVault()
    try:
        vault.unlock()
        cred = vault.get_credential(provider)
        if not cred:
            print(f"No active connection for '{provider}'.")
            return
        print(f"Disconnecting {provider}...")
        print("  1) Disconnect (keep credentials encrypted, can reconnect later)")
        print("  2) Disconnect and delete credentials permanently")
        print("  3) Cancel")
        choice = input("Choose [1/2/3]: ").strip()
        if choice == "1":
            vault.deactivate_credential(cred.id)
            vault.log_consent(provider, "disconnected", reason="user_initiated")
            vault.set_connection_status(provider, connected=False)
            print(f"  {provider} disconnected. Credentials kept encrypted.")
        elif choice == "2":
            confirm = input(f"Permanently delete {provider} credentials? (yes/no): ")
            if confirm.lower() == "yes":
                vault.delete_credential(cred.id)
                vault.log_consent(provider, "disconnected", reason="user_deleted")
                vault.set_connection_status(provider, connected=False)
                print(f"  {provider} credentials permanently deleted.")
            else:
                print("  Cancelled.")
        else:
            print("  Cancelled.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vault.lock()


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
        "insights": cmd_insights,
        "schedule": cmd_schedule,
        "skills": cmd_skills,
        "connections": cmd_connections,
        "consent-log": cmd_consent_log,
        "connect": cmd_connect,
        "disconnect": cmd_disconnect,
        "email": lambda args, cfg=None: {
            "summary": cmd_email_summary,
            "sync": cmd_email_sync,
            "config": cmd_email_config,
        }.get(args.email_command, lambda a, c=None: print("Usage: homie email {summary|sync|config}"))(args, cfg),
        "folder": cmd_folder,
        "social": cmd_social,
        "sm": cmd_sm,
        "browser": cmd_browser,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    elif args.command == "vault":
        if hasattr(args, 'func'):
            args.func(args)
        else:
            print("Usage: homie vault {status}")
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
