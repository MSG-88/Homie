# src/homie_core/meta_learning/strategy_selector.py
"""Strategy Selector -- epsilon-greedy and UCB1 multi-armed bandits."""
from __future__ import annotations
import logging, math, random, time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from .strategies import BUILTIN_STRATEGIES, ReasoningStrategy, get_strategy_by_name

log = logging.getLogger(__name__)
EXPLORE_RATE = 0.15


@dataclass
class ArmRecord:
    attempts: int = 0; successes: int = 0
    total_duration_ms: float = 0.0; total_quality: float = 0.0
    total_satisfaction: float = 0.0; last_used: float = 0.0

    @property
    def success_rate(self): return self.successes / self.attempts if self.attempts else 0.0
    @property
    def avg_quality(self): return self.total_quality / self.attempts if self.attempts else 0.0
    @property
    def avg_satisfaction(self): return self.total_satisfaction / self.attempts if self.attempts else 0.0
    @property
    def avg_duration_ms(self): return self.total_duration_ms / self.attempts if self.attempts else 0.0
    def reward(self): return 0.40*self.success_rate + 0.35*self.avg_quality + 0.25*self.avg_satisfaction


class SelectionAlgorithm(str, Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    UCB1 = "ucb1"


def _eps_select(arms, eps):
    keys = list(arms.keys())
    if not keys: raise ValueError("No arms")
    return random.choice(keys) if random.random() < eps else max(keys, key=lambda k: arms[k].reward())


def _ucb1_select(arms):
    keys = list(arms.keys())
    if not keys: raise ValueError("No arms")
    total = sum(a.attempts for a in arms.values())
    untried = [k for k in keys if arms[k].attempts == 0]
    if untried: return random.choice(untried)
    return max(keys, key=lambda k: arms[k].reward() + math.sqrt(2.0*math.log(total)/arms[k].attempts))


_DEFAULT_STRATEGIES = {
    "code_generation": [
        {"agents": ["coder"], "planning": "chain_of_thought", "tools": ["editor"]},
        {"agents": ["coder", "reviewer"], "planning": "plan_and_execute", "tools": ["editor", "linter"]},
    ],
    "research": [
        {"agents": ["researcher"], "planning": "breadth_first", "tools": ["web_search"]},
        {"agents": ["researcher", "summariser"], "planning": "iterative_deepening", "tools": ["web_search", "rag"]},
    ],
    "conversation": [{"agents": ["conversationalist"], "planning": "reactive", "tools": []}],
}


def _skey(s):
    if isinstance(s, ReasoningStrategy): return s.name
    return f"{','.join(sorted(s.get('agents',[])))}" + f"|{s.get('planning',s.get('name',''))}" + f"|{','.join(sorted(s.get('tools',[])))}"


class StrategySelector:
    """Bandit-based strategy selector with SQLite persistence."""

    def __init__(self, storage=None, explore_rate=EXPLORE_RATE, algorithm=SelectionAlgorithm.UCB1):
        self._storage, self._explore_rate, self._algorithm = storage, explore_rate, algorithm
        self._records: dict[str, dict[str, ArmRecord]] = {}
        self._strategies: dict[str, dict[str, dict]] = {}
        self._reasoning: dict[str, ReasoningStrategy] = {}
        for tt, strats in _DEFAULT_STRATEGIES.items():
            for s in strats: self._strategies.setdefault(tt, {})[_skey(s)] = s
        for s in BUILTIN_STRATEGIES:
            self._reasoning[s.name] = s
            for tt in ("conversation","code_generation","research","general","analysis","creative"):
                self._strategies.setdefault(tt, {})[_skey(s)] = s.to_dict()
        if storage:
            try:
                for r in storage.load_strategies():
                    self._strategies.setdefault(r["task_type"], {})[r["strategy_key"]] = r["strategy"]
                for r in storage.load_strategy_records():
                    self._records.setdefault(r["task_type"], {})[r["strategy_key"]] = ArmRecord(
                        attempts=r["attempts"], successes=r["successes"],
                        total_duration_ms=r["total_duration_ms"], total_quality=r["total_quality"],
                        total_satisfaction=r.get("total_satisfaction", 0.0))
            except Exception: log.warning("Failed to load meta-learning state", exc_info=True)

    def select_strategy(self, task_type, context=None):
        cands = self._strategies.get(task_type, {})
        if not cands: return {"agents": [], "planning": "reactive", "tools": []}
        arm_map = self._records.setdefault(task_type, {})
        for k in cands: arm_map.setdefault(k, ArmRecord())
        chosen = _eps_select(arm_map, self._explore_rate) if self._algorithm == SelectionAlgorithm.EPSILON_GREEDY else _ucb1_select(arm_map)
        return dict(cands[chosen])

    def select_reasoning_strategy(self, task_type, context=None):
        sel = self.select_strategy(task_type, context)
        name = sel.get("planning") or sel.get("name", "")
        return self._reasoning.get(name) or self._reasoning.get(_skey(sel))

    def record_outcome(self, task_type, strategy, success, metrics=None):
        metrics = metrics or {}; key = _skey(strategy)
        arm = self._records.setdefault(task_type, {}).setdefault(key, ArmRecord())
        arm.attempts += 1
        if success: arm.successes += 1
        arm.total_duration_ms += metrics.get("duration_ms", 0.0)
        arm.total_quality += metrics.get("quality", 0.0)
        arm.total_satisfaction += metrics.get("satisfaction", 0.0)
        arm.last_used = time.time()
        self._strategies.setdefault(task_type, {})[key] = strategy.to_dict() if isinstance(strategy, ReasoningStrategy) else strategy
        if self._storage:
            try: self._storage.upsert_strategy_record(task_type=task_type, strategy_key=key, attempts=arm.attempts, successes=arm.successes, total_duration_ms=arm.total_duration_ms, total_quality=arm.total_quality, total_satisfaction=arm.total_satisfaction, last_used=str(arm.last_used))
            except: pass

    def get_strategy_stats(self, task_type):
        return {k: {"attempts": r.attempts, "success_rate": round(r.success_rate,4), "avg_duration_ms": round(r.avg_duration_ms,2), "avg_quality": round(r.avg_quality,4), "avg_satisfaction": round(r.avg_satisfaction,4), "reward": round(r.reward(),4)} for k,r in self._records.get(task_type,{}).items()}

    def register_strategy(self, task_type, strategy):
        key = _skey(strategy)
        if isinstance(strategy, ReasoningStrategy):
            self._strategies.setdefault(task_type, {})[key] = strategy.to_dict()
            self._reasoning[key] = strategy
        else: self._strategies.setdefault(task_type, {})[key] = strategy

    @property
    def algorithm(self): return self._algorithm
    @algorithm.setter
    def algorithm(self, v): self._algorithm = v
    @property
    def explore_rate(self): return self._explore_rate
    @explore_rate.setter
    def explore_rate(self, v): self._explore_rate = max(0.0, min(1.0, v))
