"""Hub election — deterministic leader election based on capability score."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ElectionCandidate:
    node_id: str
    capability_score: float
    created_at: str


def elect_hub(candidates: list[ElectionCandidate]) -> Optional[ElectionCandidate]:
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: (-c.capability_score, c.created_at))[0]
