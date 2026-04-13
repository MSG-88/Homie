"""Handler for /mesh slash command — mesh topology and management."""
from __future__ import annotations
from pathlib import Path
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_mesh_status(args: str, **ctx) -> str:
    from homie_core.mesh.identity import load_identity
    from homie_core.mesh.registry import MeshNodeRegistry
    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)
    if identity is None:
        return "No node identity. Run `/node init` first."
    if not identity.mesh_id:
        return (
            f"**This node:** {identity.node_name} ({identity.role})\n"
            f"**Mesh:** not joined\n\n"
            f"Use `/mesh pair` (on hub) or `/mesh join --code <code>` to form a mesh."
        )
    cfg = ctx.get("config")
    if not cfg:
        return "No configuration loaded."
    db_path = Path(cfg.storage.path) / "mesh_nodes.db"
    registry = MeshNodeRegistry(db_path)
    registry.initialize()
    nodes = registry.list_all()
    lines = [
        f"**Mesh:** {identity.mesh_id}",
        f"**This node:** {identity.node_name} ({identity.role})",
        f"**Nodes:** {len(nodes) + 1}",
        "",
    ]
    for node in nodes:
        icon = "+" if node.status == "online" else "-" if node.status == "offline" else "~"
        lines.append(f"  [{icon}] {node.node_name} ({node.role}) — score: {node.capability_score:.0f} — {node.status}")
    return "\n".join(lines)


def _handle_mesh_pair(args: str, **ctx) -> str:
    from homie_core.mesh.identity import load_identity
    from homie_core.mesh.pairing import generate_pairing_code
    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)
    if identity is None:
        return "No node identity. Run `/node init` first."
    cfg = ctx.get("config")
    ttl = cfg.mesh.pairing_timeout if cfg else 300
    session = generate_pairing_code(ttl_seconds=ttl)
    return (
        f"**Pairing Code:** {session.code}\n"
        f"**Expires in:** {ttl} seconds\n\n"
        f"On the other machine, run: `/mesh join --code {session.code}`"
    )


def _handle_mesh_leave(args: str, **ctx) -> str:
    from homie_core.mesh.identity import load_identity, save_identity
    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)
    if identity is None:
        return "No node identity."
    if not identity.mesh_id:
        return "Not in a mesh."
    old_mesh = identity.mesh_id
    identity.mesh_id = None
    identity.role = "standalone"
    save_identity(identity, identity_path)
    return f"Left mesh {old_mesh}. Node is now standalone."


def _handle_mesh_health(args: str, **ctx) -> str:
    """Run mesh health checks."""
    from homie_core.mesh.health import MeshHealthChecker
    mesh_ctx = ctx.get("_mesh_ctx")
    checker = MeshHealthChecker(mesh_context=mesh_ctx)
    health = checker.run_all()
    return health.summary()


def _handle_mesh_api(args: str, **ctx) -> str:
    """Start the mesh REST API server."""
    try:
        import uvicorn
        from fastapi import FastAPI
        from homie_core.mesh.api import create_mesh_router
        from homie_core.mesh.bootstrap import bootstrap_mesh
    except ImportError:
        return "FastAPI/uvicorn not installed. Run: pip install homie-ai[app]"

    config = ctx.get("config")
    if not config:
        return "No configuration loaded."

    mesh_ctx = ctx.get("_mesh_ctx")
    if mesh_ctx is None:
        try:
            mesh_ctx = bootstrap_mesh(config)
        except Exception as e:
            return f"Mesh bootstrap failed: {e}"

    router = create_mesh_router(mesh_ctx)
    if router is None:
        return "FastAPI not available."

    port = 8721
    parts = args.strip().split()
    if "--port" in parts:
        idx = parts.index("--port")
        if idx + 1 < len(parts):
            port = int(parts[idx + 1])

    app = FastAPI(title="Homie Mesh API", version="1.0.0")
    app.include_router(router)

    print(f"Starting mesh API on http://localhost:{port}")
    print(f"  Node: {mesh_ctx.identity.node_name} ({mesh_ctx.identity.node_id[:8]}...)")
    print(f"  Score: {mesh_ctx.capabilities.capability_score():.0f}")
    print("  Press Ctrl+C to stop\n")

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        pass
    return "Mesh API stopped."


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="mesh",
        description="Mesh topology and management",
        args_spec="<status|pair|leave|nodes|health|api>",
        handler_fn=_handle_mesh_status,
        subcommands={
            "status": SlashCommand(name="status", description="Show mesh topology", handler_fn=_handle_mesh_status),
            "pair": SlashCommand(name="pair", description="Generate pairing code", handler_fn=_handle_mesh_pair),
            "leave": SlashCommand(name="leave", description="Leave current mesh", handler_fn=_handle_mesh_leave),
            "nodes": SlashCommand(name="nodes", description="List all nodes", handler_fn=_handle_mesh_status),
            "health": SlashCommand(name="health", description="Run health checks", handler_fn=_handle_mesh_health),
            "api": SlashCommand(name="api", description="Start mesh REST API", args_spec="[--port N]", handler_fn=_handle_mesh_api),
        },
    ))
