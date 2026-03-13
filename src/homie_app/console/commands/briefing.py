"""Handler for /briefing slash command — personalized daily briefing."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_briefing(args: str, **ctx) -> str:
    cfg = ctx.get("config")
    location = getattr(cfg, "location", None) if cfg else None
    user_name = getattr(cfg, "user_name", "User") or "User"

    lines = [f"**Good {'morning' if _is_morning() else 'day'}, {user_name}!**\n"]

    # Weather
    if location and location.city:
        try:
            api_key = _get_api_key("weather", ctx)
            if api_key:
                from homie_core.intelligence.weather import WeatherService
                weather = WeatherService(api_key=api_key)
                data = weather.get_current(location.city)
                if "error" not in data:
                    lines.append(f"**Weather in {data['city']}:** {data['temp']}\u00b0C, {data['description']}")
                    lines.append("")
        except Exception:
            pass

    # News
    try:
        api_key = _get_api_key("news", ctx)
        if api_key:
            from homie_core.intelligence.news import NewsService
            country = location.country.lower() if location and location.country else "us"
            news = NewsService(api_key=api_key)
            data = news.get_headlines(country=country)
            if data.get("articles"):
                lines.append("**Top Headlines:**")
                for a in data["articles"][:5]:
                    lines.append(f"  - {a['title']}")
                lines.append("")
    except Exception:
        pass

    # Email summary
    try:
        from homie_core.email import EmailService
        from pathlib import Path
        import sqlite3

        vault = ctx.get("vault")
        if vault:
            storage_path = Path(cfg.storage.path).expanduser()
            cache_conn = sqlite3.connect(str(storage_path / "cache.db"))
            email_svc = EmailService(vault, cache_conn)
            accounts = email_svc.initialize()
            if accounts:
                summary = email_svc.get_summary(days=1)
                unread = summary.get("unread", 0)
                hp = summary.get("high_priority", [])
                if unread > 0:
                    lines.append(f"**Email:** {unread} unread")
                    if hp:
                        lines.append(f"  {len(hp)} high priority:")
                        for msg in hp[:3]:
                            lines.append(f"    - {msg.get('subject', '(no subject)')}")
                    lines.append("")
    except Exception:
        pass

    if len(lines) == 1:
        lines.append("No data sources configured yet. Use /connect to set up weather, news, or email.")

    return "\n".join(lines)


def _get_api_key(provider: str, ctx: dict) -> str:
    try:
        vault = ctx.get("vault")
        if not vault:
            return ""
        cred = vault.get_credential(provider, "default")
        return cred.access_token if cred else ""
    except Exception:
        return ""


def _is_morning() -> bool:
    from datetime import datetime
    return datetime.now().hour < 12


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="briefing",
        description="Full personalized briefing (weather, news, email, schedule)",
        handler_fn=_handle_briefing,
    ))
