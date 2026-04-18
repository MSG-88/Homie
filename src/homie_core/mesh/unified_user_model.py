from __future__ import annotations
from typing import Optional
from homie_core.mesh.node_context import NodeContext


class UnifiedUserModel:
    def __init__(self):
        self._contexts: dict[str, NodeContext] = {}

    @property
    def node_contexts(self) -> dict[str, NodeContext]:
        return dict(self._contexts)

    @property
    def active_nodes(self) -> list[str]:
        return list(self._contexts.keys())

    @property
    def primary_node(self) -> Optional[str]:
        if not self._contexts:
            return None
        candidates = [(nid, ctx) for nid, ctx in self._contexts.items() if not ctx.is_idle]
        if not candidates:
            return max(self._contexts, key=lambda n: self._contexts[n].last_updated)
        return max(candidates, key=lambda item: item[1].minutes_active + (item[1].flow_score * 10))[0]

    def update_node(self, context: NodeContext) -> None:
        self._contexts[context.node_id] = context

    def remove_node(self, node_id: str) -> None:
        self._contexts.pop(node_id, None)

    def activity_summary(self) -> str:
        if not self._contexts:
            return "No devices connected"
        return ", ".join(ctx.summary() for ctx in self._contexts.values())

    def to_context_block(self) -> str:
        if not self._contexts:
            return ""
        primary = self.primary_node
        lines = ["[CROSS-DEVICE CONTEXT]"]
        for nid, ctx in self._contexts.items():
            lines.append(f"  {ctx.summary()}{' (primary)' if nid == primary else ''}")
        return "\n".join(lines)
