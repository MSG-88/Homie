"""Handler for /help slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_help(args: str, **ctx) -> str:
    router: SlashCommandRouter = ctx.get("_router")
    if not router:
        return "Help unavailable."

    if args.strip():
        cmd_name = args.strip().lstrip("/")
        commands = {c.name: c for c in router.list_commands()}
        cmd = commands.get(cmd_name)
        if not cmd:
            return f"Unknown command: /{cmd_name}"
        lines = [f"**/{cmd.name}** — {cmd.description}"]
        if cmd.subcommands:
            lines.append("\nSubcommands:")
            for sc in sorted(cmd.subcommands.values(), key=lambda s: s.name):
                lines.append(sc.format_help())
        return "\n".join(lines)

    lines = ["**Homie Commands:**"]
    for cmd in router.list_commands():
        lines.append(cmd.format_help())
    lines.append("\nType /help <command> for details on a specific command.")
    return "\n".join(lines)


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="help",
        description="Show available commands or help for a specific command",
        args_spec="[command]",
        handler_fn=_handle_help,
    ))
