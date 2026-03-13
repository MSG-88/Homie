"""Handler for /news slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_news(args: str, **ctx) -> str:
    cfg = ctx.get("config")
    location = getattr(cfg, "location", None) if cfg else None

    # Determine country code from location
    country = "us"
    if location and location.country:
        country = location.country.lower()

    query = args.strip()

    # Get API key
    api_key = ""
    try:
        vault = ctx.get("vault")
        if vault:
            cred = vault.get_credential("news", "default")
            if cred:
                api_key = cred.access_token
    except Exception:
        pass

    from homie_core.intelligence.news import NewsService
    service = NewsService(api_key=api_key)
    data = service.get_headlines(country=country, query=query)
    return service.format_headlines(data)


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="news",
        description="Top headlines (e.g., /news, /news technology, /news local)",
        args_spec="[topic]",
        handler_fn=_handle_news,
    ))
