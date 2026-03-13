"""Handler for /email slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _short_sender(sender: str) -> str:
    """Extract readable sender name: 'Foo Bar <foo@bar.com>' -> 'Foo Bar'."""
    import re
    match = re.match(r'^"?([^"<]+)"?\s*<', sender)
    if match:
        return match.group(1).strip()
    return sender.split("@")[0] if "@" in sender else sender


def _make_email_service(**ctx):
    """Create an EmailService from console context."""
    from homie_core.email import EmailService
    from homie_core.vault.secure_vault import SecureVault
    from pathlib import Path
    import sqlite3

    cfg = ctx.get("config")
    vault = ctx.get("vault")
    if not vault:
        vault = SecureVault()
        vault.unlock()

    storage_path = Path(cfg.storage.path).expanduser()
    cache_conn = sqlite3.connect(str(storage_path / "cache.db"))

    model_engine = ctx.get("model_engine")
    service = EmailService(vault, cache_conn, model_engine=model_engine)
    accounts = service.initialize()
    return service, accounts


def _handle_email_summary(args: str, **ctx) -> str:
    try:
        service, accounts = _make_email_service(**ctx)
        if not accounts:
            return "No email accounts connected. Use /connect gmail first."

        days = 1
        if args.strip():
            try:
                days = int(args.strip().split("--days")[-1].strip() if "--days" in args else args.strip())
            except ValueError:
                days = 1

        summary = service.get_summary(days=days)
        lines = [f"**Email Summary ({days} day{'s' if days > 1 else ''}):**"]
        lines.append(f"  Total: {summary.get('total', 0)}")
        lines.append(f"  Unread: {summary.get('unread', 0)}")
        hp = summary.get("high_priority", [])
        if hp:
            lines.append(f"  High priority: {len(hp)}")
            for msg in hp[:5]:
                lines.append(f"    - {msg.get('subject', '(no subject)')}")
        return "\n".join(lines)
    except Exception as e:
        return f"Email summary failed: {e}"


def _handle_email_sync(args: str, **ctx) -> str:
    try:
        service, accounts = _make_email_service(**ctx)
        if not accounts:
            return "No email accounts connected. Use /connect gmail first."

        result = service.sync_tick()
        return f"Sync complete: {result}"
    except Exception as e:
        return f"Email sync failed: {e}"


def _handle_email_triage(args: str, **ctx) -> str:
    """Batch-triage unread emails using AI analysis."""
    try:
        import json
        service, accounts = _make_email_service(**ctx)
        if not accounts:
            return "No email accounts connected. Use /connect gmail first."

        max_emails = 15
        account = "all"
        if args.strip():
            parts = args.strip().split()
            for p in parts:
                if p.isdigit():
                    max_emails = min(int(p), 15)
                elif "@" in p:
                    account = p

        from homie_core.email.classifier import clean_snippet

        result = service.triage(account=account, max_emails=max_emails)

        if isinstance(result, dict) and "action_needed" in result:
            # LLM-powered triage
            lines = [f"**Email Triage** ({result.get('status', '')})\n"]

            action = result.get("action_needed", [])
            if action:
                lines.append(f"**Action Required ({len(action)}):**")
                for e in action[:10]:
                    lines.append(f"  [{e.get('llm_priority', 'medium')}] {_short_sender(e.get('sender', ''))}")
                    lines.append(f"    {e.get('subject', '')}")
                    lines.append(f"    -> {e.get('intent', '')}")
                lines.append("")

            important = result.get("important", [])
            if important:
                lines.append(f"**Important ({len(important)}):**")
                for e in important[:10]:
                    lines.append(f"  {_short_sender(e.get('sender', ''))}: {e.get('subject', '')}")
                    lines.append(f"    {clean_snippet(e.get('summary', ''))[:120]}")
                lines.append("")

            spam = result.get("likely_spam", [])
            if spam:
                lines.append(f"**Likely Spam ({len(spam)}):**")
                for e in spam[:5]:
                    lines.append(f"  {_short_sender(e.get('sender', ''))}: {e.get('subject', '')}")

            return "\n".join(lines)

        # Heuristic-only fallback — format nicely instead of raw JSON
        emails = result.get("emails", [])
        if not emails:
            return "No unread emails."

        # Bucket by priority
        high = [e for e in emails if e.get("priority") == "high"]
        medium = [e for e in emails if e.get("priority") == "medium"]
        low = [e for e in emails if e.get("priority") == "low"]

        lines = [f"**Email Triage** ({len(emails)} unread)\n"]

        if high:
            lines.append(f"**High Priority ({len(high)}):**")
            for e in high:
                lines.append(f"  {_short_sender(e.get('sender', ''))}: {e.get('subject', '')}")
                snippet = clean_snippet(e.get("snippet", ""))
                if snippet:
                    lines.append(f"    {snippet[:120]}")
            lines.append("")

        if medium:
            lines.append(f"**Medium ({len(medium)}):**")
            for e in medium:
                lines.append(f"  {_short_sender(e.get('sender', ''))}: {e.get('subject', '')}")
                snippet = clean_snippet(e.get("snippet", ""))
                if snippet:
                    lines.append(f"    {snippet[:120]}")
            lines.append("")

        if low:
            lines.append(f"**Low Priority ({len(low)}):**")
            for e in low:
                lines.append(f"  {_short_sender(e.get('sender', ''))}: {e.get('subject', '')}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Email triage failed: {e}"


def _handle_email_digest(args: str, **ctx) -> str:
    """Generate an AI-powered email digest."""
    try:
        service, accounts = _make_email_service(**ctx)
        if not accounts:
            return "No email accounts connected. Use /connect gmail first."

        days = 1
        if args.strip():
            try:
                days = int(args.strip())
            except ValueError:
                days = 1

        result = service.get_intelligent_digest(days=days)
        if isinstance(result, str):
            return result
        import json
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Email digest failed: {e}"


def _handle_email_config(args: str, **ctx) -> str:
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        creds = vault.list_credentials("gmail")
        if not creds:
            return "No Gmail accounts configured. Use /connect gmail."
        lines = ["**Email Configuration:**"]
        for c in creds:
            lines.append(f"  Account: {c.account_id}")
            lines.append(f"  Active: {c.active}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not read email config: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="email",
        description="Email operations (summary, sync, triage, digest, config)",
        args_spec="summary|sync|triage|digest|config",
        subcommands={
            "summary": SlashCommand(name="summary", description="Email summary", args_spec="[--days N]", handler_fn=_handle_email_summary),
            "sync": SlashCommand(name="sync", description="Force sync now", handler_fn=_handle_email_sync),
            "triage": SlashCommand(name="triage", description="AI-powered email triage", args_spec="[max_emails] [account]", handler_fn=_handle_email_triage),
            "digest": SlashCommand(name="digest", description="AI-powered email digest", args_spec="[days]", handler_fn=_handle_email_digest),
            "config": SlashCommand(name="config", description="Show email settings", handler_fn=_handle_email_config),
        },
    ))
