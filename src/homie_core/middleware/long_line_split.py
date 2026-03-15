from __future__ import annotations

from homie_core.middleware.base import HomieMiddleware
from homie_core.config import HomieConfig


class LongLineSplitMiddleware(HomieMiddleware):
    name = "long_line_split"
    order = 85

    def __init__(self, config: HomieConfig):
        self._threshold = config.context.long_line_threshold

    def wrap_tool_result(self, name: str, result: str) -> str:
        lines = result.splitlines()
        output_lines = []
        for i, line in enumerate(lines, 1):
            if len(line) <= self._threshold:
                output_lines.append(line)
            else:
                chunks = [line[j:j + self._threshold] for j in range(0, len(line), self._threshold)]
                for k, chunk in enumerate(chunks, 1):
                    output_lines.append(f"[line {i}.{k}] {chunk}")
        return "\n".join(output_lines)
