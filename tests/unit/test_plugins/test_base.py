from homie_core.plugins.base import HomiePlugin, PluginResult


class TestPlugin(HomiePlugin):
    name = "test"
    description = "A test plugin"
    permissions = ["read_files"]

    def on_activate(self, config):
        self.config = config
    def on_deactivate(self):
        pass
    def on_query(self, intent, params):
        return PluginResult(success=True, data=f"queried: {intent}")
    def on_action(self, action, params):
        return PluginResult(success=True, data=f"action: {action}")


def test_plugin_interface():
    p = TestPlugin()
    assert p.name == "test"
    assert p.description == "A test plugin"


def test_plugin_activate():
    p = TestPlugin()
    p.on_activate({"key": "value"})
    assert p.config == {"key": "value"}


def test_plugin_query():
    p = TestPlugin()
    result = p.on_query("search", {"q": "test"})
    assert result.success is True
    assert "queried" in result.data


def test_plugin_action():
    p = TestPlugin()
    result = p.on_action("run", {})
    assert result.success is True


def test_plugin_context_default():
    p = TestPlugin()
    assert p.on_context() == {}
