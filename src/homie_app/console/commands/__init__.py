"""Register all slash commands with the router."""
from __future__ import annotations
from homie_app.console.router import SlashCommandRouter, SlashCommand


def register_all_commands(router: SlashCommandRouter, ctx: dict) -> None:
    """Import and register every command module."""
    ctx["_router"] = router

    from homie_app.console.commands.help import register as reg_help
    from homie_app.console.commands.memory import register as reg_memory
    from homie_app.console.commands.connect import register as reg_connect
    from homie_app.console.commands.email import register as reg_email
    from homie_app.console.commands.consent import register as reg_consent
    from homie_app.console.commands.vault import register as reg_vault
    from homie_app.console.commands.settings import register as reg_settings
    from homie_app.console.commands.model import register as reg_model
    from homie_app.console.commands.plugins import register as reg_plugins
    from homie_app.console.commands.daemon import register as reg_daemon
    from homie_app.console.commands.folder import register as reg_folder
    from homie_app.console.commands.social import register as reg_social
    from homie_app.console.commands.sm import register as reg_sm
    from homie_app.console.commands.browser import register as reg_browser
    from homie_app.console.commands.voice import register as reg_voice
    from homie_app.console.commands.backup import register as reg_backup
    from homie_app.console.commands.insights import register as reg_insights
    from homie_app.console.commands.schedule import register as reg_schedule
    from homie_app.console.commands.skills import register as reg_skills
    from homie_app.console.commands.location import register as reg_location
    from homie_app.console.commands.weather import register as reg_weather
    from homie_app.console.commands.news import register as reg_news
    from homie_app.console.commands.briefing import register as reg_briefing
    from homie_app.console.commands.desktop import register as reg_desktop
    from homie_app.console.commands.node import register as reg_node
    from homie_app.console.commands.mesh import register as reg_mesh

    reg_help(router, ctx)
    reg_memory(router, ctx)
    reg_connect(router, ctx)
    reg_email(router, ctx)
    reg_consent(router, ctx)
    reg_vault(router, ctx)
    reg_settings(router, ctx)
    reg_model(router, ctx)
    reg_plugins(router, ctx)
    reg_daemon(router, ctx)
    reg_folder(router, ctx)
    reg_social(router, ctx)
    reg_sm(router, ctx)
    reg_browser(router, ctx)
    reg_voice(router, ctx)
    reg_backup(router, ctx)
    reg_insights(router, ctx)
    reg_schedule(router, ctx)
    reg_skills(router, ctx)
    reg_location(router, ctx)
    reg_weather(router, ctx)
    reg_news(router, ctx)
    reg_briefing(router, ctx)
    reg_desktop(router, ctx)
    reg_node(router, ctx)
    reg_mesh(router, ctx)

    # Dashboard command
    def _handle_dashboard(args, **ctx):
        from homie_app.console.dashboard import show_dashboard
        from homie_app.console.rich_console import rc
        show_dashboard(rc, engine=ctx.get("engine"), config=ctx.get("config"),
                       mesh_ctx=ctx.get("_mesh_ctx"))
        return ""

    router.register(SlashCommand(
        name="dashboard",
        description="Show system dashboard with live stats",
        handler_fn=_handle_dashboard,
    ))

    # Quit is handled by Console.run() directly before router dispatch
    router.register(SlashCommand(
        name="quit",
        description="Exit Homie",
        handler_fn=lambda args, **ctx: "",
    ))
