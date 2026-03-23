"""Goal, ThoughtChain, and ThoughtStep data models for the planning engine."""

from __future__ import annotations

import json
import uuid
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ThoughtStep:
    """A single step within a chain-of-thought plan."""

    id: str
    reasoning: str
    action: str
    expected_outcome: str
    agent: str  # which agent handles this
    dependencies: list[str]  # step IDs that must complete first
    result: Optional[dict] = None
    status: str = "pending"  # pending, active, complete, failed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ThoughtStep:
        return cls(**data)


@dataclass
class ThoughtChain:
    """A chain of thought steps decomposing a goal into actionable work."""

    goal: str
    steps: list[ThoughtStep]
    current_step: int = 0
    status: str = "thinking"  # thinking, executing, complete, failed, replanning

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ThoughtChain:
        steps = [ThoughtStep.from_dict(s) for s in data["steps"]]
        return cls(
            goal=data["goal"],
            steps=steps,
            current_step=data.get("current_step", 0),
            status=data.get("status", "thinking"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> ThoughtChain:
        return cls.from_dict(json.loads(raw))

    def get_ready_steps(self) -> list[ThoughtStep]:
        """Return steps whose dependencies are all complete and that are still pending."""
        completed_ids = {s.id for s in self.steps if s.status == "complete"}
        return [
            s
            for s in self.steps
            if s.status == "pending"
            and all(dep in completed_ids for dep in s.dependencies)
        ]

    def advance(self) -> Optional[ThoughtStep]:
        """Move to the next ready step, or return None if blocked/done."""
        ready = self.get_ready_steps()
        if not ready:
            return None
        step = ready[0]
        step.status = "active"
        self.current_step = self.steps.index(step)
        self.status = "executing"
        return step

    @property
    def is_complete(self) -> bool:
        return all(s.status == "complete" for s in self.steps)

    @property
    def has_failed(self) -> bool:
        return any(s.status == "failed" for s in self.steps)


@dataclass
class Goal:
    """A top-level or sub-goal with its associated thought chain."""

    id: str
    description: str
    parent_id: Optional[str] = None
    thought_chain: Optional[ThoughtChain] = None
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    outcome: Optional[str] = None
    lessons_learned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "parent_id": self.parent_id,
            "thought_chain": self.thought_chain.to_dict() if self.thought_chain else None,
            "priority": self.priority,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "outcome": self.outcome,
            "lessons_learned": self.lessons_learned,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Goal:
        tc = data.get("thought_chain")
        return cls(
            id=data["id"],
            description=data["description"],
            parent_id=data.get("parent_id"),
            thought_chain=ThoughtChain.from_dict(tc) if tc else None,
            priority=data.get("priority", 5),
            created_at=data.get("created_at", time.time()),
            completed_at=data.get("completed_at"),
            outcome=data.get("outcome"),
            lessons_learned=data.get("lessons_learned", []),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> Goal:
        return cls.from_dict(json.loads(raw))

    @staticmethod
    def new_id() -> str:
        return f"goal-{uuid.uuid4().hex[:12]}"
