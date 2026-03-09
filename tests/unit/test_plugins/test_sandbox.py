import time
from homie_core.plugins.sandbox import PluginSandbox
from homie_core.plugins.base import PluginResult


def test_execute_success():
    sandbox = PluginSandbox()
    result = sandbox.execute(lambda: PluginResult(success=True, data="ok"))
    assert result.success is True


def test_execute_crash():
    sandbox = PluginSandbox()
    def crasher():
        raise ValueError("crash!")
    result = sandbox.execute(crasher)
    assert result.success is False
    assert "crash" in result.error.lower()


def test_execute_timeout():
    sandbox = PluginSandbox(timeout_seconds=0.1)
    def slow():
        time.sleep(5)
    result = sandbox.execute(slow)
    assert result.success is False
    assert "timed out" in result.error.lower()


def test_crash_log():
    sandbox = PluginSandbox()
    sandbox.execute(lambda: 1/0)
    log = sandbox.get_crash_log()
    assert len(log) == 1
