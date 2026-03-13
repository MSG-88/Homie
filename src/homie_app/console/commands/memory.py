"""Handlers for memory-related slash commands: /status, /learn, /facts, /remember, /forget, /clear."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_status(args: str, **ctx) -> str:
    wm = ctx.get("wm")
    sm = ctx.get("sm")
    em = ctx.get("em")
    cfg = ctx.get("config")

    lines = ["**Homie Status**"]
    if wm:
        lines.append(f"  Memory: {len(wm.get_conversation())} messages this session")
    if sm:
        try:
            facts = sm.get_facts(min_confidence=0.0)
            lines.append(f"  Facts stored: {len(facts)}")
        except Exception:
            lines.append("  Facts: unavailable")
    if em:
        lines.append("  Episodic memory: active")
    lines.append(f"  User: {getattr(cfg, 'user_name', 'Unknown') or 'Unknown'}")
    return "\n".join(lines)


def _handle_learn(args: str, **ctx) -> str:
    brain = ctx.get("brain")
    if not brain:
        return "Brain not loaded."
    try:
        stats = brain._cognitive._learning.get_session_stats()
        lines = ["**Session Learning Stats**"]
        lines.append(f"  Interactions: {stats['interactions']}")
        lines.append(f"  Facts learned: {stats['facts_learned']}")
        if stats.get("facts"):
            for f in stats["facts"]:
                lines.append(f"    - {f}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not get learning stats: {e}"


def _handle_facts(args: str, **ctx) -> str:
    sm = ctx.get("sm")
    if not sm:
        return "Semantic memory not available."
    facts = sm.get_facts(min_confidence=0.3)
    if not facts:
        return "No facts stored yet. Chat with me and I'll learn about you!"
    lines = ["**What I know about you:**"]
    for f in facts[:15]:
        lines.append(f"  - {f['fact']} ({f['confidence']:.0%} confident)")
    return "\n".join(lines)


def _handle_remember(args: str, **ctx) -> str:
    sm = ctx.get("sm")
    fact = args.strip()
    if sm and fact:
        sm.learn(fact, confidence=0.9, tags=["user_explicit"])
        return f"Got it, I'll remember: {fact}"
    if not fact:
        return "Usage: /remember <fact to store>"
    return "Could not store that — semantic memory not available."


def _handle_forget(args: str, **ctx) -> str:
    sm = ctx.get("sm")
    topic = args.strip()
    if sm and topic:
        sm.forget_topic(topic)
        return f"Forgotten everything about: {topic}"
    if not topic:
        return "Usage: /forget <topic>"
    return "Could not forget — semantic memory not available."


def _handle_clear(args: str, **ctx) -> str:
    wm = ctx.get("wm")
    if wm:
        wm.clear()
    return "Conversation cleared. Fresh start!"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    commands = [
        ("status", "Show system status", "", _handle_status),
        ("learn", "Show what I learned this session", "", _handle_learn),
        ("facts", "Show stored facts about you", "", _handle_facts),
        ("remember", "Store a fact (e.g., /remember I prefer dark mode)", "<fact>", _handle_remember),
        ("forget", "Forget a topic (e.g., /forget work)", "<topic>", _handle_forget),
        ("clear", "Clear conversation (fresh start)", "", _handle_clear),
    ]
    for name, desc, args_spec, fn in commands:
        router.register(SlashCommand(name=name, description=desc, args_spec=args_spec, handler_fn=fn))
