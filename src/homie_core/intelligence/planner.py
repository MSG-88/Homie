"""Hierarchical Task Network (HTN) planner for Homie AI task decomposition."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """A single task node in the HTN plan."""

    name: str
    is_primitive: bool = False
    cost: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionRule:
    """Rule that decomposes an abstract task into ordered subtasks."""

    abstract_task: str
    subtasks: list[str] = field(default_factory=list)
    preconditions: dict[str, Any] = field(default_factory=dict)
    priority: int = 0


class HTNPlanner:
    """Hierarchical Task Network planner.

    Decomposes abstract tasks into primitive tasks via decomposition rules,
    respecting preconditions and depth limits.
    """

    def __init__(self, max_depth: int = 50) -> None:
        self.max_depth = max_depth
        self._rules: dict[str, list[DecompositionRule]] = {}
        self._primitives: dict[str, Task] = {}

    def add_rule(self, rule: DecompositionRule) -> None:
        """Register a decomposition rule, keeping rules sorted by priority descending."""
        if rule.abstract_task not in self._rules:
            self._rules[rule.abstract_task] = []
        self._rules[rule.abstract_task].append(rule)
        self._rules[rule.abstract_task].sort(key=lambda r: r.priority, reverse=True)

    def get_rules(self, task_name: str) -> list[DecompositionRule]:
        """Return all decomposition rules registered for a task name."""
        return list(self._rules.get(task_name, []))

    def mark_primitive(self, name: str, cost: float = 1.0, **metadata: Any) -> None:
        """Register a task name as primitive with an associated cost."""
        self._primitives[name] = Task(
            name=name, is_primitive=True, cost=cost, metadata=metadata
        )

    def _check_preconditions(
        self, rule: DecompositionRule, state: dict[str, Any]
    ) -> bool:
        """Return True if all preconditions in the rule are satisfied by state."""
        for key, value in rule.preconditions.items():
            if state.get(key) != value:
                return False
        return True

    def _decompose(
        self, task_name: str, state: dict[str, Any], depth: int
    ) -> list[Task] | None:
        """Recursively decompose a task into primitive tasks.

        Returns None if no valid decomposition exists or depth is exceeded.
        """
        if depth > self.max_depth:
            return None

        # If the task is primitive, return it directly.
        if task_name in self._primitives:
            return [self._primitives[task_name]]

        # Try each rule for this task (already sorted by priority).
        rules = self.get_rules(task_name)
        if not rules:
            return None

        for rule in rules:
            if not self._check_preconditions(rule, state):
                continue

            plan: list[Task] = []
            success = True
            for subtask_name in rule.subtasks:
                result = self._decompose(subtask_name, state, depth + 1)
                if result is None:
                    success = False
                    break
                plan.extend(result)

            if success:
                return plan

        return None

    def plan(
        self, goal: str, state: dict[str, Any] | None = None
    ) -> list[Task] | None:
        """Plan how to achieve a goal by decomposing it into primitive tasks.

        Args:
            goal: The top-level abstract task name.
            state: World state used to evaluate preconditions.

        Returns:
            An ordered list of primitive Tasks, or None if planning fails.
        """
        if state is None:
            state = {}
        return self._decompose(goal, state, depth=0)

    def estimate_cost(self, plan: list[Task] | None) -> float:
        """Sum the costs of all tasks in a plan."""
        if plan is None:
            return 0.0
        return sum(t.cost for t in plan)

    def serialize(self) -> dict[str, Any]:
        """Serialize the planner configuration to a JSON-compatible dict."""
        rules_data = []
        for rule_list in self._rules.values():
            for rule in rule_list:
                rules_data.append(
                    {
                        "abstract_task": rule.abstract_task,
                        "subtasks": rule.subtasks,
                        "preconditions": rule.preconditions,
                        "priority": rule.priority,
                    }
                )

        primitives_data = {}
        for name, task in self._primitives.items():
            primitives_data[name] = {
                "cost": task.cost,
                "metadata": task.metadata,
            }

        return {
            "max_depth": self.max_depth,
            "rules": rules_data,
            "primitives": primitives_data,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> HTNPlanner:
        """Reconstruct an HTNPlanner from a serialized dict."""
        planner = cls(max_depth=data.get("max_depth", 50))

        for rule_data in data.get("rules", []):
            planner.add_rule(
                DecompositionRule(
                    abstract_task=rule_data["abstract_task"],
                    subtasks=rule_data["subtasks"],
                    preconditions=rule_data.get("preconditions", {}),
                    priority=rule_data.get("priority", 0),
                )
            )

        for name, prim_data in data.get("primitives", {}).items():
            planner.mark_primitive(
                name, cost=prim_data.get("cost", 1.0), **prim_data.get("metadata", {})
            )

        return planner
