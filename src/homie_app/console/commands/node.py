"""Handler for /node slash command — node identity and info."""
from __future__ import annotations
from pathlib import Path
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_node_info(args: str, **ctx) -> str:
    from homie_core.mesh.identity import load_identity
    from homie_core.mesh.capabilities import detect_capabilities
    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)
    if identity is None:
        return "No node identity found. Run `homie init` or `/node init` to create one."
    caps = detect_capabilities()
    gpu_str = f"{caps.gpu['name']} ({caps.gpu['vram_gb']} GB)" if caps.gpu else "none"
    lines = [
        "**Node Identity**",
        f"  ID:   {identity.node_id}",
        f"  Name: {identity.node_name}",
        f"  Role: {identity.role}",
        f"  Mesh: {identity.mesh_id or 'not joined'}",
        "",
        "**Capabilities**",
        f"  OS:      {caps.os}",
        f"  CPU:     {caps.cpu_cores} cores",
        f"  RAM:     {caps.ram_gb} GB",
        f"  Disk:    {caps.disk_free_gb} GB free",
        f"  GPU:     {gpu_str}",
        f"  Mic:     {'yes' if caps.has_mic else 'no'}",
        f"  Display: {'yes' if caps.has_display else 'no'}",
        f"  Score:   {caps.capability_score():.0f}",
    ]
    return "\n".join(lines)


def _handle_node_init(args: str, **ctx) -> str:
    from homie_core.mesh.identity import NodeIdentity, save_identity, load_identity
    identity_path = Path.home() / ".homie" / "node.json"
    existing = load_identity(identity_path)
    if existing and "--force" not in args:
        return (
            f"Node identity already exists: {existing.node_id} ({existing.node_name})\n"
            f"Use `/node init --force` to regenerate."
        )
    identity = NodeIdentity.generate()
    save_identity(identity, identity_path)
    return f"Node identity created:\n  ID:   {identity.node_id}\n  Name: {identity.node_name}"


def _handle_node_set_name(args: str, **ctx) -> str:
    name = args.strip()
    if not name:
        return "Usage: /node set-name <name>"
    from homie_core.mesh.identity import load_identity, save_identity
    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)
    if identity is None:
        return "No node identity. Run `/node init` first."
    identity.node_name = name
    save_identity(identity, identity_path)
    return f"Node name set to: {name}"


def _handle_node_set_role(args: str, **ctx) -> str:
    role = args.strip().lower()
    if role not in ("hub", "spoke", "standalone"):
        return "Usage: /node set-role <hub|spoke|standalone>"
    from homie_core.mesh.identity import load_identity, save_identity
    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)
    if identity is None:
        return "No node identity. Run `/node init` first."
    identity.role = role
    save_identity(identity, identity_path)
    return f"Node role set to: {role}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="node",
        description="Node identity and management",
        args_spec="<info|init|set-name|set-role>",
        handler_fn=_handle_node_info,
        subcommands={
            "info": SlashCommand(name="info", description="Show node identity", handler_fn=_handle_node_info),
            "init": SlashCommand(name="init", description="Create node identity", handler_fn=_handle_node_init),
            "set-name": SlashCommand(name="set-name", description="Set node name", args_spec="<name>", handler_fn=_handle_node_set_name),
            "set-role": SlashCommand(name="set-role", description="Force node role", args_spec="<hub|spoke>", handler_fn=_handle_node_set_role),
        },
    ))
