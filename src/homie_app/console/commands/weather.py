"""Handler for /weather slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_weather(args: str, **ctx) -> str:
    cfg = ctx.get("config")
    location = getattr(cfg, "location", None) if cfg else None

    # Parse args — could be "forecast", a city name, or "forecast <city>"
    text = args.strip()
    forecast = False
    city = ""

    if text.lower().startswith("forecast"):
        forecast = True
        city = text[8:].strip()
    elif text:
        city = text

    if not city and location and location.city:
        city = location.city

    if not city:
        return "No location set. Use /location set <city> first, or /weather <city>."

    # Get API key from vault
    api_key = ""
    try:
        vault = ctx.get("vault")
        if vault:
            cred = vault.get_credential("weather", "default")
            if cred:
                api_key = cred.access_token
    except Exception:
        pass

    from homie_core.intelligence.weather import WeatherService
    service = WeatherService(api_key=api_key)

    if forecast:
        data = service.get_forecast(city)
        return service.format_forecast(data)
    else:
        data = service.get_current(city)
        return service.format_current(data)


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="weather",
        description="Current weather or forecast (e.g., /weather, /weather forecast, /weather London)",
        args_spec="[forecast] [city]",
        handler_fn=_handle_weather,
    ))
