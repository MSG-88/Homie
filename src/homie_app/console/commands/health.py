"""Handler for /health slash command — system health check."""
from __future__ import annotations

from homie_app.console.router import SlashCommand, SlashCommandRouter


def _check_inference(ctx: dict) -> tuple[str, str]:
    """Check inference engine health. Returns (status, detail)."""
    brain = ctx.get("brain")
    if not brain:
        return "UNAVAILABLE", "Brain/orchestrator not loaded"
    try:
        # Check if the underlying model engine is reachable
        engine = getattr(brain, "_engine", None) or getattr(
            getattr(brain, "_cognitive", None), "_engine", None
        )
        if engine is None:
            return "UNKNOWN", "Engine reference not accessible"
        # If the engine has a health-check method, use it
        if hasattr(engine, "is_running"):
            running = engine.is_running()
            return ("OK", "Engine running") if running else ("DEGRADED", "Engine not running")
        return "OK", "Engine present"
    except Exception as e:
        return "ERROR", str(e)


def _check_memory(ctx: dict) -> tuple[str, str]:
    """Check memory subsystem health."""
    sm = ctx.get("sm")
    em = ctx.get("em")
    wm = ctx.get("wm")
    parts = []
    if wm:
        try:
            msgs = len(wm.get_conversation())
            parts.append(f"working:{msgs} msgs")
        except Exception as e:
            return "ERROR", f"Working memory error: {e}"
    else:
        return "UNAVAILABLE", "Working memory not loaded"
    if sm:
        try:
            facts = sm.get_facts(min_confidence=0.0)
            parts.append(f"semantic:{len(facts)} facts")
        except Exception as e:
            return "DEGRADED", f"Semantic memory error: {e}"
    else:
        parts.append("semantic:unavailable")
    if em:
        parts.append("episodic:ok")
    else:
        parts.append("episodic:unavailable")
    return "OK", ", ".join(parts)


def _check_voice(ctx: dict) -> tuple[str, str]:
    """Check voice pipeline health."""
    voice = ctx.get("voice_manager") or ctx.get("voice")
    if voice is None:
        return "UNAVAILABLE", "Voice manager not loaded"
    try:
        running = getattr(voice, "_running", None)
        if running is None:
            return "UNKNOWN", "Cannot determine voice state"
        return ("OK", "Voice pipeline active") if running else ("IDLE", "Voice pipeline stopped")
    except Exception as e:
        return "ERROR", str(e)


def _check_plugins(ctx: dict) -> tuple[str, str]:
    """Check plugin manager health."""
    plugin_mgr = ctx.get("plugin_manager") or ctx.get("plugins")
    if plugin_mgr is None:
        return "UNAVAILABLE", "Plugin manager not loaded"
    try:
        all_plugins = plugin_mgr.list_plugins()
        enabled = plugin_mgr.list_enabled()
        return "OK", f"{len(enabled)}/{len(all_plugins)} plugins enabled"
    except Exception as e:
        return "ERROR", str(e)


def _check_knowledge_graph(ctx: dict) -> tuple[str, str]:
    """Check knowledge graph health."""
    kg = ctx.get("kg") or ctx.get("knowledge_graph")
    brain = ctx.get("brain")
    # Try to reach the KG via brain's cognitive arch
    if kg is None and brain:
        kg = getattr(getattr(brain, "_cognitive", None), "_kg", None)
    if kg is None:
        return "UNAVAILABLE", "Knowledge graph not loaded"
    try:
        # A lightweight probe — just count nodes if possible
        if hasattr(kg, "entity_count"):
            count = kg.entity_count()
            return "OK", f"{count} entities"
        return "OK", "Knowledge graph present"
    except Exception as e:
        return "ERROR", str(e)


_STATUS_ICONS = {
    "OK": "[OK]",
    "IDLE": "[IDLE]",
    "DEGRADED": "[DEGRADED]",
    "UNAVAILABLE": "[UNAVAILABLE]",
    "UNKNOWN": "[UNKNOWN]",
    "ERROR": "[ERROR]",
}


def _handle_health(args: str, **ctx) -> str:
    checks = [
        ("Inference", _check_inference(ctx)),
        ("Memory", _check_memory(ctx)),
        ("Voice", _check_voice(ctx)),
        ("Plugins", _check_plugins(ctx)),
        ("Knowledge Graph", _check_knowledge_graph(ctx)),
    ]

    lines = ["**Homie Health Check**", ""]
    overall_ok = True
    for name, (status, detail) in checks:
        icon = _STATUS_ICONS.get(status, f"[{status}]")
        lines.append(f"  {icon:<14} {name:<18} {detail}")
        if status in ("ERROR", "DEGRADED"):
            overall_ok = False

    lines.append("")
    lines.append("Overall: " + ("Healthy" if overall_ok else "Degraded — check details above"))
    return "\n".join(lines)


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="health",
        description="Show health status of inference, memory, voice, plugins, knowledge graph",
        args_spec="",
        handler_fn=_handle_health,
    ))
