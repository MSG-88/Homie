from __future__ import annotations

from typing import Any, Optional

from homie_core.plugins.base import HomiePlugin, PluginResult


class MCPHost:
    def __init__(self):
        self._servers: dict[str, dict] = {}

    def register_server(self, name: str, config: dict) -> None:
        self._servers[name] = {"config": config, "connected": False, "tools": []}

    def unregister_server(self, name: str) -> None:
        self._servers.pop(name, None)

    def list_servers(self) -> list[dict]:
        return [
            {"name": name, "connected": info["connected"], "tool_count": len(info["tools"])}
            for name, info in self._servers.items()
        ]

    def get_tools(self, server_name: str) -> list[dict]:
        server = self._servers.get(server_name)
        if server:
            return server["tools"]
        return []

    def invoke_tool(self, server_name: str, tool_name: str, params: dict | None = None) -> PluginResult:
        server = self._servers.get(server_name)
        if not server:
            return PluginResult(success=False, error=f"MCP server '{server_name}' not found")
        if not server["connected"]:
            return PluginResult(success=False, error=f"MCP server '{server_name}' not connected")
        # Actual MCP protocol would be implemented here
        return PluginResult(success=False, error="MCP protocol not yet implemented")

    def list_all_tools(self) -> list[dict]:
        all_tools = []
        for name, info in self._servers.items():
            for tool in info["tools"]:
                all_tools.append({**tool, "server": name})
        return all_tools
