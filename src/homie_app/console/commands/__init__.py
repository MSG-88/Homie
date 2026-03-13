"""Register all slash commands with the router."""
from __future__ import annotations
from homie_app.console.router import SlashCommandRouter, SlashCommand


def register_all_commands(router: SlashCommandRouter, ctx: dict) -> None:
    """Import and register every command module."""
    ctx["_router"] = router

    from homie_app.console.commands.help import register as reg_help
    from homie_app.console.commands.memory import register as reg_memory
    from homie_app.console.commands.connect import register as reg_connect

    reg_help(router, ctx)
    reg_memory(router, ctx)
    reg_connect(router, ctx)

    # Quit is handled by Console.run() directly before router dispatch
    router.register(SlashCommand(
        name="quit",
        description="Exit Homie",
        handler_fn=lambda args, **ctx: "",
    ))
