from __future__ import annotations

import pytest
from homie_core.middleware.base import HomieMiddleware
from homie_core.middleware.stack import MiddlewareStack


class OrderTrackerMW(HomieMiddleware):
    def __init__(self, name: str, order: int, log: list):
        self.name = name
        self.order = order
        self._log = log

    def before_turn(self, message: str, state: dict) -> str | None:
        self._log.append(f"before:{self.name}")
        return message

    def after_turn(self, response: str, state: dict) -> str:
        self._log.append(f"after:{self.name}")
        return response

    def wrap_tool_call(self, name: str, args: dict) -> dict | None:
        self._log.append(f"wrap_call:{self.name}")
        return args

    def wrap_tool_result(self, name: str, result) -> object:
        self._log.append(f"wrap_result:{self.name}")
        return result


def test_empty_stack_apply_tools():
    stack = MiddlewareStack()
    tools = [{"name": "t1"}]
    assert stack.apply_tools(tools) == tools


def test_empty_stack_apply_prompt():
    stack = MiddlewareStack()
    assert stack.apply_prompt("hello") == "hello"


def test_empty_stack_run_before_turn():
    stack = MiddlewareStack()
    assert stack.run_before_turn("msg", {}) == "msg"


def test_empty_stack_run_after_turn():
    stack = MiddlewareStack()
    assert stack.run_after_turn("resp", {}) == "resp"


def test_empty_stack_run_wrap_tool_call():
    stack = MiddlewareStack()
    args = {"x": 1}
    assert stack.run_wrap_tool_call("tool", args) == args


def test_empty_stack_run_wrap_tool_result():
    stack = MiddlewareStack()
    assert stack.run_wrap_tool_result("tool", "result") == "result"


def test_middleware_sorted_by_order_on_init():
    log = []
    mw_high = OrderTrackerMW("high", order=200, log=log)
    mw_low = OrderTrackerMW("low", order=10, log=log)
    stack = MiddlewareStack([mw_high, mw_low])
    stack.run_before_turn("msg", {})
    assert log == ["before:low", "before:high"]


def test_add_maintains_sorted_order():
    log = []
    mw_high = OrderTrackerMW("high", order=200, log=log)
    mw_low = OrderTrackerMW("low", order=10, log=log)
    stack = MiddlewareStack([mw_high])
    stack.add(mw_low)
    stack.run_before_turn("msg", {})
    assert log == ["before:low", "before:high"]


def test_after_turn_runs_in_reverse_order():
    log = []
    mw_a = OrderTrackerMW("a", order=10, log=log)
    mw_b = OrderTrackerMW("b", order=20, log=log)
    stack = MiddlewareStack([mw_a, mw_b])
    stack.run_after_turn("resp", {})
    assert log == ["after:b", "after:a"]


def test_wrap_tool_result_runs_in_reverse_order():
    log = []
    mw_a = OrderTrackerMW("a", order=10, log=log)
    mw_b = OrderTrackerMW("b", order=20, log=log)
    stack = MiddlewareStack([mw_a, mw_b])
    stack.run_wrap_tool_result("tool", "res")
    assert log == ["wrap_result:b", "wrap_result:a"]


def test_before_turn_blocking_stops_chain():
    log = []

    class BlockMW(HomieMiddleware):
        name = "blocker"
        order = 10

        def before_turn(self, message: str, state: dict) -> str | None:
            log.append("block")
            return None

    mw_after = OrderTrackerMW("after_block", order=20, log=log)
    stack = MiddlewareStack([BlockMW(), mw_after])
    result = stack.run_before_turn("msg", {})
    assert result is None
    assert log == ["block"]


def test_wrap_tool_call_blocking_stops_chain():
    log = []

    class BlockMW(HomieMiddleware):
        name = "blocker"
        order = 10

        def wrap_tool_call(self, name: str, args: dict) -> dict | None:
            log.append("block")
            return None

    mw_after = OrderTrackerMW("after_block", order=20, log=log)
    stack = MiddlewareStack([BlockMW(), mw_after])
    result = stack.run_wrap_tool_call("tool", {"a": 1})
    assert result is None
    assert log == ["block"]


def test_apply_tools_chains_in_order():
    class AddToolMW(HomieMiddleware):
        def __init__(self, tool_name: str, order: int):
            self.name = f"add_{tool_name}"
            self.order = order
            self._tool_name = tool_name

        def modify_tools(self, tools: list) -> list:
            return tools + [{"name": self._tool_name}]

    stack = MiddlewareStack([AddToolMW("b", 20), AddToolMW("a", 10)])
    result = stack.apply_tools([])
    assert result == [{"name": "a"}, {"name": "b"}]


def test_apply_prompt_chains_in_order():
    class PrefixMW(HomieMiddleware):
        def __init__(self, prefix: str, order: int):
            self.name = f"prefix_{prefix}"
            self.order = order
            self._prefix = prefix

        def modify_prompt(self, prompt: str) -> str:
            return self._prefix + prompt

    stack = MiddlewareStack([PrefixMW("B:", 20), PrefixMW("A:", 10)])
    result = stack.apply_prompt("msg")
    assert result == "B:A:msg"
