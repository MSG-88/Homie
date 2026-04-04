# src/homie_core/meta_learning/strategies.py
"""Concrete reasoning strategies: DirectPrompt, ChainOfThought, ToolAugmented."""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger(__name__)


class ReasoningStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def cost_estimate(self) -> float: ...
    @abstractmethod
    def apply(self, query: str, context: dict[str, Any]) -> dict[str, Any]: ...
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "planning": self.name, "cost_estimate": self.cost_estimate, "agents": [], "tools": []}


class DirectPromptStrategy(ReasoningStrategy):
    @property
    def name(self): return "direct_prompt"
    @property
    def cost_estimate(self): return 0.1
    def apply(self, query, context):
        return {"guidance": "Answer directly and concisely.", "context_budget_factor": 0.6, "planning": "reactive", "tools": []}
    def to_dict(self):
        return {"name": self.name, "planning": "reactive", "cost_estimate": self.cost_estimate, "agents": ["conversationalist"], "tools": []}


class ChainOfThoughtStrategy(ReasoningStrategy):
    @property
    def name(self): return "chain_of_thought"
    @property
    def cost_estimate(self): return 0.5
    def apply(self, query, context):
        complexity = context.get("complexity", "moderate")
        if complexity in ("complex", "deep"):
            guidance = ("Think step by step: 1) Identify the core question. "
                        "2) List relevant facts. 3) Reason through possibilities. "
                        "4) Synthesise a clear answer. Show reasoning before conclusion.")
            return {"guidance": guidance, "context_budget_factor": 1.0, "max_tokens_factor": 1.5,
                    "temperature": 0.5, "planning": "chain_of_thought", "tools": []}
        return {"guidance": "Think briefly step by step before answering.",
                "context_budget_factor": 1.0, "max_tokens_factor": 1.2,
                "temperature": 0.5, "planning": "chain_of_thought", "tools": []}
    def to_dict(self):
        return {"name": self.name, "planning": "chain_of_thought", "cost_estimate": self.cost_estimate, "agents": ["reasoner"], "tools": []}


class ToolAugmentedStrategy(ReasoningStrategy):
    def __init__(self, available_tools=None):
        self._available_tools = available_tools or ["web_search", "rag", "editor", "calculator"]
    @property
    def name(self): return "tool_augmented"
    @property
    def cost_estimate(self): return 0.9
    def apply(self, query, context):
        selected, q = [], query.lower()
        if any(kw in q for kw in ("search", "find", "latest", "news")): selected.append("web_search")
        if any(kw in q for kw in ("document", "file", "pdf", "article")): selected.append("rag")
        if any(kw in q for kw in ("code", "script", "function", "debug")): selected.append("editor")
        if any(kw in q for kw in ("calculate", "compute", "math")): selected.append("calculator")
        if not selected: selected = ["web_search", "rag"]
        selected = [t for t in selected if t in self._available_tools]
        return {"guidance": f"Use available tools: {', '.join(selected)}. Cite tool results.",
                "context_budget_factor": 1.3, "max_tokens_factor": 1.5,
                "temperature": 0.6, "planning": "tool_augmented", "tools": selected}
    def to_dict(self):
        return {"name": self.name, "planning": "tool_augmented", "cost_estimate": self.cost_estimate,
                "agents": ["researcher"], "tools": list(self._available_tools)}


BUILTIN_STRATEGIES: list[ReasoningStrategy] = [DirectPromptStrategy(), ChainOfThoughtStrategy(), ToolAugmentedStrategy()]

def get_strategy_by_name(name: str) -> ReasoningStrategy | None:
    for s in BUILTIN_STRATEGIES:
        if s.name == name: return s
    return None
