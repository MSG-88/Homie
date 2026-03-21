"""Tiered context assembler for RAG-augmented prompts.

Assembles retrieved chunks, graph context, conversation history, and user profile
into a single context string that fits within a token budget.

Priority tiers (highest to lowest):
    Tier 1: Top reranked chunks (direct retrieval hits)
    Tier 2: Knowledge-graph context for entities mentioned in the query
    Tier 3: Recent conversation turns (last 3 exchanges = 6 messages)
    Tier 4: User profile key-value pairs

Lost-in-middle mitigation: tier-1 chunks are split between the start and end
of the assembled context, with lower-priority tiers sandwiched in between.
"""
from __future__ import annotations

from typing import Optional

from homie_core.knowledge.graph import KnowledgeGraph


class ContextAssembler:
    """Assembles context within a token budget using tiered priority."""

    def __init__(self, graph: Optional[KnowledgeGraph] = None) -> None:
        self._graph = graph

    def assemble(
        self,
        query: str,
        chunks: list[dict],           # {"content": str, "source": str, "score": float}
        conversation: list[dict],      # recent conversation turns
        user_profile: dict | None,     # user facts
        token_budget: int = 4096,      # in estimated tokens (chars/4)
    ) -> str:
        """Assemble context with tiered priority.

        Tier 1: Top reranked chunks (direct retrieval hits) — up to 60% of budget.
        Tier 2: Graph context for mentioned entities — up to 75% of budget.
        Tier 3: Recent conversation (last 3 turns) — up to 90% of budget.
        Tier 4: User profile — up to 100% of budget.

        Lost-in-middle mitigation: high-relevance tier-1 chunks placed at both
        start and end of the assembled context.

        Returns:
            A single string with all sections separated by double newlines.
        """
        char_budget = token_budget * 4
        sections: list[tuple[str, str]] = []  # (tier, content_block)
        used = 0

        # ------------------------------------------------------------------
        # Tier 1: Retrieved chunks (highest priority — up to 60% of budget)
        # ------------------------------------------------------------------
        for chunk in chunks:
            content = chunk.get("content", "")
            source = chunk.get("source", "")
            block = f"[Source: {source}]\n{content}"
            if used + len(block) > char_budget * 0.6:
                break
            sections.append(("tier1", block))
            used += len(block)

        # ------------------------------------------------------------------
        # Tier 2: Graph entity context (up to 75% of budget)
        # ------------------------------------------------------------------
        if self._graph:
            mentioned = self._graph.entities_mentioned_in(query)
            for entity in mentioned[:3]:
                ctx = self._graph.context_for_entity(entity.id)
                if ctx and used + len(ctx) <= char_budget * 0.75:
                    sections.append(("tier2", f"[Entity: {entity.name}]\n{ctx}"))
                    used += len(ctx)

        # ------------------------------------------------------------------
        # Tier 3: Recent conversation (last 6 messages = 3 exchanges)
        # ------------------------------------------------------------------
        if conversation:
            recent = conversation[-6:]
            conv_text = "\n".join(
                f"{m.get('role', '')}: {m.get('content', '')[:200]}"
                for m in recent
            )
            if used + len(conv_text) <= char_budget * 0.9:
                sections.append(("tier3", f"[Recent Conversation]\n{conv_text}"))
                used += len(conv_text)

        # ------------------------------------------------------------------
        # Tier 4: User profile
        # ------------------------------------------------------------------
        if user_profile:
            profile_text = "\n".join(
                f"- {k}: {v}" for k, v in user_profile.items() if v
            )
            if profile_text and used + len(profile_text) <= char_budget:
                sections.append(("tier4", f"[User Profile]\n{profile_text}"))

        if not sections:
            return ""

        # ------------------------------------------------------------------
        # Lost-in-middle mitigation: split tier-1 between start and end
        # ------------------------------------------------------------------
        tier1_blocks = [block for tier, block in sections if tier == "tier1"]
        other_blocks = [block for tier, block in sections if tier != "tier1"]

        if len(tier1_blocks) >= 2:
            mid = len(tier1_blocks) // 2
            ordered = tier1_blocks[:mid] + other_blocks + tier1_blocks[mid:]
        else:
            ordered = tier1_blocks + other_blocks

        return "\n\n".join(ordered)
