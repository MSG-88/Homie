"""Handler for /suggestions slash command — on-demand proactive insights."""
from __future__ import annotations

from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_suggestions(args: str, **ctx) -> str:
    """Display context-aware suggestions and pending follow-ups."""
    lines: list[str] = []

    # Try to get proactive intelligence from context
    proactive = ctx.get("proactive_intelligence")

    if proactive:
        suggestions = proactive.get_suggestions()
        if suggestions:
            lines.append("**Suggestions:**")
            for s in suggestions:
                lines.append(f"  - {s}")
            lines.append("")

        # Show pending follow-ups
        pending = proactive.followup_tracker.get_pending()
        if pending:
            lines.append(f"**Pending Follow-ups ({len(pending)}):**")
            for fu in pending[:10]:
                due = f" (due: {fu.due_by})" if fu.due_by else ""
                lines.append(f"  - {fu.text}{due}")
            lines.append("")

        # Show detected patterns
        patterns = proactive.pattern_detector.get_patterns(min_occurrences=3)
        if patterns:
            lines.append("**Detected Patterns:**")
            for p in patterns[:5]:
                lines.append(f"  - '{p.action}' at {p.typical_hour}:00 ({p.occurrences}x)")
            lines.append("")

    if not lines:
        lines.append("No proactive insights available yet.")
        lines.append("Homie learns your patterns over time. Keep using it and suggestions will appear here.")

    return "\n".join(lines)


def _handle_followups(args: str, **ctx) -> str:
    """Manage tracked follow-ups."""
    proactive = ctx.get("proactive_intelligence")
    if not proactive:
        return "Proactive intelligence not available."

    tracker = proactive.followup_tracker

    parts = args.strip().split(maxsplit=1)
    subcmd = parts[0].lower() if parts else "list"

    if subcmd == "list" or not subcmd:
        items = tracker.list_all()
        active = [f for f in items if not f.dismissed]
        if not active:
            return "No tracked follow-ups."
        lines = [f"**Follow-ups ({len(active)} active):**"]
        for f in active:
            status = "[surfaced]" if f.surfaced else "[pending]"
            due = f" (due: {f.due_by})" if f.due_by else ""
            lines.append(f"  {status} {f.text}{due}  [id: {f.id}]")
        return "\n".join(lines)

    elif subcmd == "dismiss" and len(parts) > 1:
        fid = parts[1].strip()
        tracker.dismiss(fid)
        return f"Dismissed follow-up: {fid}"

    elif subcmd == "add" and len(parts) > 1:
        text = parts[1].strip()
        added = tracker.ingest(f"I need to {text}", source="manual")
        if added:
            return f"Tracked: {added[0].text}"
        return "Could not extract a follow-up from that text. Try: /followups add review the PR"

    elif subcmd == "cleanup":
        removed = tracker.cleanup()
        return f"Cleaned up {removed} old follow-ups."

    else:
        return ("Usage:\n"
                "  /followups           — list all follow-ups\n"
                "  /followups add <text> — manually add a follow-up\n"
                "  /followups dismiss <id> — dismiss a follow-up\n"
                "  /followups cleanup   — remove old dismissed items")


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="suggestions",
        description="Context-aware proactive suggestions and insights",
        handler_fn=_handle_suggestions,
    ))
    router.register(SlashCommand(
        name="followups",
        description="View and manage tracked follow-ups",
        args_spec="[list|add|dismiss|cleanup]",
        handler_fn=_handle_followups,
    ))
