"""Handler for /location slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_location(args: str, **ctx) -> str:
    cfg = ctx.get("config")
    if not cfg:
        return "No configuration loaded."

    text = args.strip()

    # /location set <city>
    if text.lower() == "set" or text.lower().startswith("set "):
        city = text[4:].strip() if len(text) > 3 else ""
        if not city:
            return "Usage: /location set <city>"

        from homie_core.config import LocationConfig
        cfg.location = LocationConfig(city=city)
        try:
            from homie_app.init import _save_config
            _save_config(cfg, ctx.get("config_path") or "homie.config.yaml")
        except Exception:
            pass
        return f"Location set to: {city}. Refine with /settings > Location for region/country/timezone."

    # /location — show current
    loc = getattr(cfg, "location", None)
    if not loc or not loc.city:
        return "Location not set. Use /location set <city> to configure."

    lines = ["**Location:**"]
    lines.append(f"  City: {loc.city}")
    if loc.region:
        lines.append(f"  Region: {loc.region}")
    if loc.country:
        lines.append(f"  Country: {loc.country}")
    if loc.timezone:
        lines.append(f"  Timezone: {loc.timezone}")
    return "\n".join(lines)


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="location",
        description="View or set location (e.g., /location set Chennai)",
        args_spec="[set <city>]",
        handler_fn=_handle_location,
    ))
