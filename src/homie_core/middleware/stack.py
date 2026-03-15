from __future__ import annotations

from typing import Any, Optional

from homie_core.middleware.base import HomieMiddleware


class MiddlewareStack:
    def __init__(self, middleware: list[HomieMiddleware] | None = None) -> None:
        self._stack: list[HomieMiddleware] = sorted(
            middleware or [], key=lambda mw: mw.order
        )

    def add(self, mw: HomieMiddleware) -> None:
        self._stack.append(mw)
        self._stack.sort(key=lambda m: m.order)

    def apply_tools(self, tools: list) -> list:
        for mw in self._stack:
            tools = mw.modify_tools(tools)
        return tools

    def apply_prompt(self, prompt: str) -> str:
        for mw in self._stack:
            prompt = mw.modify_prompt(prompt)
        return prompt

    def run_before_turn(self, message: str, state: dict) -> str | None:
        for mw in self._stack:
            result = mw.before_turn(message, state)
            if result is None:
                return None
            message = result
        return message

    def run_after_turn(self, response: str, state: dict) -> str:
        for mw in reversed(self._stack):
            response = mw.after_turn(response, state)
        return response

    def run_wrap_tool_call(self, name: str, args: dict) -> dict | None:
        for mw in self._stack:
            result = mw.wrap_tool_call(name, args)
            if result is None:
                return None
            args = result
        return args

    def run_wrap_tool_result(self, name: str, result: Any) -> Any:
        for mw in reversed(self._stack):
            result = mw.wrap_tool_result(name, result)
        return result
