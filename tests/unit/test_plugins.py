"""Unit tests for plugin system — base, manager, discovery, tool registration.

Covers:
- HomiePlugin ABC: interface methods, default implementations
- PluginResult: success/failure/error fields
- PluginManager: register, enable, disable, query, action, context collection
- PluginManager: load_from_directory (dynamic discovery)
- PluginManager: error isolation (crashing plugins don't break others)
- PluginManager: thread safety
"""
from __future__ import annotations

import threading
import textwrap
import pytest

from homie_core.plugins.base import HomiePlugin, PluginResult
from homie_core.plugins.manager import PluginManager


# ---------------------------------------------------------------------------
# Concrete plugin implementations for testing
# ---------------------------------------------------------------------------

class SimplePlugin(HomiePlugin):
    name = "simple"
    description = "A simple test plugin"
    version = "1.0.0"
    permissions = ["read"]
    activated_with: dict = {}
    deactivated: bool = False

    def on_activate(self, config: dict) -> None:
        self.activated_with = config

    def on_deactivate(self) -> None:
        self.deactivated = True

    def on_context(self) -> dict:
        return {"app": "test", "status": "running"}

    def on_query(self, intent: str, params: dict) -> PluginResult:
        return PluginResult(success=True, data=f"answer:{intent}")

    def on_action(self, action: str, params: dict) -> PluginResult:
        return PluginResult(success=True, data=f"did:{action}")


class AnotherPlugin(HomiePlugin):
    name = "another"
    description = "Second plugin"
    version = "2.0.0"
    permissions = ["write"]

    def on_activate(self, config: dict) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def on_query(self, intent: str, params: dict) -> PluginResult:
        return PluginResult(success=True, data=params.get("value", "default"))


class CrashOnActivatePlugin(HomiePlugin):
    name = "crash_activate"
    description = "Crashes on activate"

    def on_activate(self, config: dict) -> None:
        raise RuntimeError("activation failure")

    def on_deactivate(self) -> None:
        pass


class CrashOnQueryPlugin(HomiePlugin):
    name = "crash_query"
    description = "Crashes on query"

    def on_activate(self, config: dict) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def on_query(self, intent: str, params: dict) -> PluginResult:
        raise ValueError("query boom")

    def on_action(self, action: str, params: dict) -> PluginResult:
        raise ValueError("action boom")


class CrashOnContextPlugin(HomiePlugin):
    name = "crash_context"
    description = "Crashes on context"

    def on_activate(self, config: dict) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def on_context(self) -> dict:
        raise RuntimeError("context boom")


class ContextPlugin(HomiePlugin):
    name = "context_provider"
    description = "Provides context"

    def on_activate(self, config: dict) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def on_context(self) -> dict:
        return {"weather": "sunny", "time": "09:00"}


# ---------------------------------------------------------------------------
# PluginResult
# ---------------------------------------------------------------------------

class TestPluginResult:
    def test_success_result(self):
        r = PluginResult(success=True, data="some data")
        assert r.success is True
        assert r.data == "some data"
        assert r.error == ""

    def test_failure_result(self):
        r = PluginResult(success=False, error="something went wrong")
        assert r.success is False
        assert r.error == "something went wrong"

    def test_default_data_is_none(self):
        r = PluginResult(success=True)
        assert r.data is None

    def test_default_error_is_empty_string(self):
        r = PluginResult(success=True)
        assert r.error == ""

    def test_result_with_dict_data(self):
        r = PluginResult(success=True, data={"key": "value"})
        assert r.data["key"] == "value"


# ---------------------------------------------------------------------------
# HomiePlugin default methods
# ---------------------------------------------------------------------------

class TestHomiePluginDefaults:
    def test_on_context_default_empty(self):
        plugin = SimplePlugin()
        # Our SimplePlugin overrides on_context, test with the raw ABC default
        # by calling HomiePlugin's on_context via super
        class MinimalPlugin(HomiePlugin):
            name = "minimal"
            def on_activate(self, config): pass
            def on_deactivate(self): pass

        p = MinimalPlugin()
        assert p.on_context() == {}

    def test_on_query_default_not_implemented(self):
        class MinimalPlugin(HomiePlugin):
            name = "minimal"
            def on_activate(self, config): pass
            def on_deactivate(self): pass

        p = MinimalPlugin()
        result = p.on_query("test", {})
        assert result.success is False
        assert "Not implemented" in result.error

    def test_on_action_default_not_implemented(self):
        class MinimalPlugin(HomiePlugin):
            name = "minimal"
            def on_activate(self, config): pass
            def on_deactivate(self): pass

        p = MinimalPlugin()
        result = p.on_action("do_thing", {})
        assert result.success is False

    def test_plugin_metadata_attributes(self):
        plugin = SimplePlugin()
        assert plugin.name == "simple"
        assert plugin.description == "A simple test plugin"
        assert plugin.version == "1.0.0"
        assert "read" in plugin.permissions


# ---------------------------------------------------------------------------
# PluginManager — register and list
# ---------------------------------------------------------------------------

class TestPluginManagerRegister:
    def test_register_single_plugin(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        plugins = mgr.list_plugins()
        assert len(plugins) == 1

    def test_register_multiple_plugins(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        mgr.register(AnotherPlugin())
        plugins = mgr.list_plugins()
        assert len(plugins) == 2

    def test_list_plugins_has_required_fields(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        info = mgr.list_plugins()[0]
        assert "name" in info
        assert "description" in info
        assert "version" in info
        assert "enabled" in info
        assert "permissions" in info

    def test_list_plugins_not_enabled_by_default(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        info = mgr.list_plugins()[0]
        assert info["enabled"] is False

    def test_get_plugin_returns_instance(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        plugin = mgr.get_plugin("simple")
        assert plugin is not None
        assert plugin.name == "simple"

    def test_get_nonexistent_plugin_returns_none(self):
        mgr = PluginManager()
        assert mgr.get_plugin("ghost") is None

    def test_register_overwrites_same_name(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        new_plugin = SimplePlugin()
        mgr.register(new_plugin)
        assert mgr.get_plugin("simple") is new_plugin


# ---------------------------------------------------------------------------
# PluginManager — enable and disable
# ---------------------------------------------------------------------------

class TestPluginManagerEnableDisable:
    def test_enable_existing_plugin(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        result = mgr.enable("simple")
        assert result is True

    def test_enable_adds_to_enabled_list(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        mgr.enable("simple")
        assert "simple" in mgr.list_enabled()

    def test_enable_with_config_passed_to_plugin(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        mgr.enable("simple", config={"key": "value"})
        assert plugin.activated_with == {"key": "value"}

    def test_enable_nonexistent_returns_false(self):
        mgr = PluginManager()
        result = mgr.enable("ghost")
        assert result is False

    def test_enable_crash_returns_false(self):
        mgr = PluginManager()
        mgr.register(CrashOnActivatePlugin())
        result = mgr.enable("crash_activate")
        assert result is False

    def test_disable_enabled_plugin(self):
        mgr = PluginManager()
        plugin = SimplePlugin()
        mgr.register(plugin)
        mgr.enable("simple")
        result = mgr.disable("simple")
        assert result is True
        assert "simple" not in mgr.list_enabled()
        assert plugin.deactivated is True

    def test_disable_not_enabled_returns_false(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        result = mgr.disable("simple")
        assert result is False

    def test_disable_nonexistent_returns_false(self):
        mgr = PluginManager()
        result = mgr.disable("ghost")
        assert result is False

    def test_list_enabled_empty_initially(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        assert mgr.list_enabled() == []

    def test_enabled_flag_in_list_plugins(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        mgr.enable("simple")
        info = next(p for p in mgr.list_plugins() if p["name"] == "simple")
        assert info["enabled"] is True

    def test_re_enable_after_disable(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        mgr.enable("simple")
        mgr.disable("simple")
        result = mgr.enable("simple")
        assert result is True
        assert "simple" in mgr.list_enabled()


# ---------------------------------------------------------------------------
# PluginManager — query and action
# ---------------------------------------------------------------------------

class TestPluginManagerQuery:
    def test_query_enabled_plugin(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        mgr.enable("simple")
        result = mgr.query_plugin("simple", "search")
        assert result.success is True
        assert "answer:search" in result.data

    def test_query_disabled_plugin_fails(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        result = mgr.query_plugin("simple", "search")
        assert result.success is False
        assert "not found or not enabled" in result.error

    def test_query_nonexistent_plugin_fails(self):
        mgr = PluginManager()
        result = mgr.query_plugin("ghost", "intent")
        assert result.success is False

    def test_query_crash_isolated(self):
        mgr = PluginManager()
        mgr.register(CrashOnQueryPlugin())
        mgr.enable("crash_query")
        result = mgr.query_plugin("crash_query", "anything")
        assert result.success is False
        assert "query boom" in result.error

    def test_query_with_params(self):
        mgr = PluginManager()
        mgr.register(AnotherPlugin())
        mgr.enable("another")
        result = mgr.query_plugin("another", "get", params={"value": "hello"})
        assert result.success is True
        assert result.data == "hello"

    def test_query_default_empty_params(self):
        mgr = PluginManager()
        mgr.register(AnotherPlugin())
        mgr.enable("another")
        result = mgr.query_plugin("another", "get")
        assert result.success is True


class TestPluginManagerAction:
    def test_action_enabled_plugin(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        mgr.enable("simple")
        result = mgr.action_plugin("simple", "run")
        assert result.success is True
        assert "did:run" in result.data

    def test_action_disabled_plugin_fails(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        result = mgr.action_plugin("simple", "run")
        assert result.success is False

    def test_action_crash_isolated(self):
        mgr = PluginManager()
        mgr.register(CrashOnQueryPlugin())
        mgr.enable("crash_query")
        result = mgr.action_plugin("crash_query", "do_it")
        assert result.success is False
        assert "action boom" in result.error

    def test_action_nonexistent_plugin_fails(self):
        mgr = PluginManager()
        result = mgr.action_plugin("ghost", "action")
        assert result.success is False


# ---------------------------------------------------------------------------
# PluginManager — context collection
# ---------------------------------------------------------------------------

class TestPluginManagerContext:
    def test_collect_context_empty_when_none_enabled(self):
        mgr = PluginManager()
        mgr.register(ContextPlugin())
        ctx = mgr.collect_context()
        assert ctx == {}

    def test_collect_context_from_enabled_plugin(self):
        mgr = PluginManager()
        mgr.register(ContextPlugin())
        mgr.enable("context_provider")
        ctx = mgr.collect_context()
        assert "context_provider" in ctx
        assert ctx["context_provider"]["weather"] == "sunny"

    def test_collect_context_crash_isolated(self):
        mgr = PluginManager()
        mgr.register(CrashOnContextPlugin())
        mgr.register(ContextPlugin())
        mgr.enable("crash_context")
        mgr.enable("context_provider")
        ctx = mgr.collect_context()
        # crash_context should not appear, context_provider should
        assert "context_provider" in ctx
        assert "crash_context" not in ctx

    def test_collect_context_empty_dict_not_included(self):
        class EmptyContextPlugin(HomiePlugin):
            name = "empty_ctx"
            def on_activate(self, config): pass
            def on_deactivate(self): pass
            def on_context(self): return {}

        mgr = PluginManager()
        mgr.register(EmptyContextPlugin())
        mgr.enable("empty_ctx")
        ctx = mgr.collect_context()
        assert "empty_ctx" not in ctx

    def test_collect_context_multiple_plugins(self):
        mgr = PluginManager()
        mgr.register(ContextPlugin())
        mgr.register(SimplePlugin())
        mgr.enable("context_provider")
        mgr.enable("simple")
        ctx = mgr.collect_context()
        assert "context_provider" in ctx
        assert "simple" in ctx


# ---------------------------------------------------------------------------
# PluginManager — load_from_directory
# ---------------------------------------------------------------------------

class TestPluginManagerLoadFromDirectory:
    def test_load_from_nonexistent_directory_returns_zero(self, tmp_path):
        mgr = PluginManager()
        count = mgr.load_from_directory(tmp_path / "no_such_dir")
        assert count == 0

    def test_load_from_empty_directory_returns_zero(self, tmp_path):
        mgr = PluginManager()
        count = mgr.load_from_directory(tmp_path)
        assert count == 0

    def test_load_plugin_from_file(self, tmp_path):
        plugin_code = textwrap.dedent("""
            from homie_core.plugins.base import HomiePlugin, PluginResult

            class MyDynamicPlugin(HomiePlugin):
                name = "my_dynamic"
                description = "Dynamically loaded"
                version = "0.1.0"
                permissions = []

                def on_activate(self, config):
                    pass

                def on_deactivate(self):
                    pass
        """)
        plugin_file = tmp_path / "my_dynamic_plugin.py"
        plugin_file.write_text(plugin_code)

        mgr = PluginManager()
        count = mgr.load_from_directory(tmp_path)
        assert count == 1
        assert mgr.get_plugin("my_dynamic") is not None

    def test_load_ignores_non_plugin_files(self, tmp_path):
        (tmp_path / "helper.py").write_text("x = 1\n")
        mgr = PluginManager()
        count = mgr.load_from_directory(tmp_path)
        assert count == 0

    def test_load_broken_plugin_file_no_crash(self, tmp_path):
        bad_file = tmp_path / "broken_plugin.py"
        bad_file.write_text("this is not valid python !!!")
        mgr = PluginManager()
        count = mgr.load_from_directory(tmp_path)
        assert count == 0  # broken file silently skipped

    def test_load_multiple_plugins_from_directory(self, tmp_path):
        plugin_template = textwrap.dedent("""
            from homie_core.plugins.base import HomiePlugin

            class {cls}(HomiePlugin):
                name = "{name}"
                description = "Test"
                version = "0.1.0"
                permissions = []
                def on_activate(self, config): pass
                def on_deactivate(self): pass
        """)
        for i in range(3):
            (tmp_path / f"plugin_{i}_plugin.py").write_text(
                plugin_template.format(cls=f"Plugin{i}", name=f"plugin_{i}")
            )
        mgr = PluginManager()
        count = mgr.load_from_directory(tmp_path)
        assert count == 3


# ---------------------------------------------------------------------------
# PluginManager — thread safety
# ---------------------------------------------------------------------------

class TestPluginManagerThreadSafety:
    def test_concurrent_register_no_crash(self):
        mgr = PluginManager()
        errors = []

        def register_plugin(i):
            class DynPlugin(HomiePlugin):
                name = f"dyn_{i}"
                def on_activate(self, config): pass
                def on_deactivate(self): pass
            try:
                mgr.register(DynPlugin())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_plugin, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_enable_disable_no_crash(self):
        mgr = PluginManager()
        mgr.register(SimplePlugin())
        errors = []

        def toggle():
            try:
                for _ in range(20):
                    mgr.enable("simple")
                    mgr.disable("simple")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
