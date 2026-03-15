from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class PipelineStage(str, Enum):
    PERCEIVED = "on_perceived"
    CLASSIFIED = "on_classified"
    RETRIEVED = "on_retrieved"
    PROMPT_BUILT = "on_prompt_built"
    REFLECTED = "on_reflected"


@dataclass
class RetrievalBundle:
    facts: list
    episodes: list
    documents: list


HookCallback = Callable[[PipelineStage, Any], Any]


class HookRegistry:
    def __init__(self) -> None:
        self._hooks: dict[PipelineStage, list[HookCallback]] = {}

    def register(self, stage: PipelineStage, callback: HookCallback) -> None:
        self._hooks.setdefault(stage, []).append(callback)

    def emit(self, stage: PipelineStage, data: Any) -> Any:
        for hook in self._hooks.get(stage, []):
            result = hook(stage, data)
            if result is not None:
                data = result
        return data
