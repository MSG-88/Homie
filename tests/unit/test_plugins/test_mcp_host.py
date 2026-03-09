from homie_core.plugins.mcp_host import MCPHost


def test_register_server():
    host = MCPHost()
    host.register_server("github", {"url": "localhost:3000"})
    servers = host.list_servers()
    assert len(servers) == 1
    assert servers[0]["name"] == "github"


def test_unregister_server():
    host = MCPHost()
    host.register_server("github", {})
    host.unregister_server("github")
    assert len(host.list_servers()) == 0


def test_invoke_not_connected():
    host = MCPHost()
    host.register_server("github", {})
    result = host.invoke_tool("github", "list_repos")
    assert result.success is False


def test_list_all_tools():
    host = MCPHost()
    host.register_server("test", {"url": ""})
    host._servers["test"]["tools"] = [{"name": "tool1"}]
    tools = host.list_all_tools()
    assert len(tools) == 1
    assert tools[0]["server"] == "test"
