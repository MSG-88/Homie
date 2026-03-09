import pytest
from homie_core.plugins.base import HomiePlugin, PluginResult
from homie_core.plugins.manager import PluginManager


class FakePlugin(HomiePlugin):
    name = "fake"
    description = "Fake plugin"
    def on_activate(self, config): pass
    def on_deactivate(self): pass
    def on_query(self, intent, params):
        return PluginResult(success=True, data="fake result")
    def on_action(self, action, params):
        return PluginResult(success=True, data=f"did {action}")


class CrashPlugin(HomiePlugin):
    name = "crasher"
    description = "Plugin that crashes"
    def on_activate(self, config): pass
    def on_deactivate(self): pass
    def on_query(self, intent, params):
        raise RuntimeError("boom")


def test_register_and_list():
    mgr = PluginManager()
    mgr.register(FakePlugin())
    plugins = mgr.list_plugins()
    assert len(plugins) == 1
    assert plugins[0]["name"] == "fake"


def test_enable_disable():
    mgr = PluginManager()
    mgr.register(FakePlugin())
    assert mgr.enable("fake") is True
    assert "fake" in mgr.list_enabled()
    assert mgr.disable("fake") is True
    assert "fake" not in mgr.list_enabled()


def test_query_enabled_plugin():
    mgr = PluginManager()
    mgr.register(FakePlugin())
    mgr.enable("fake")
    result = mgr.query_plugin("fake", "search")
    assert result.success is True


def test_query_disabled_plugin():
    mgr = PluginManager()
    mgr.register(FakePlugin())
    result = mgr.query_plugin("fake", "search")
    assert result.success is False


def test_action_plugin():
    mgr = PluginManager()
    mgr.register(FakePlugin())
    mgr.enable("fake")
    result = mgr.action_plugin("fake", "run")
    assert result.success is True


def test_crash_handled():
    mgr = PluginManager()
    mgr.register(CrashPlugin())
    mgr.enable("crasher")
    result = mgr.query_plugin("crasher", "anything")
    assert result.success is False
    assert "boom" in result.error


def test_collect_context():
    mgr = PluginManager()
    mgr.register(FakePlugin())
    mgr.enable("fake")
    context = mgr.collect_context()
    assert isinstance(context, dict)
