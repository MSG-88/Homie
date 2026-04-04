"""ContextEnricherMiddleware — injects live context from multiple sources into
the system prompt.

Sources:
- KnowledgeGraph: recently accessed entities
- email_summary_fn: callable returning a short email digest string
- behavioral_summary_fn: callable returning a short activity string
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

from homie_core.middleware.base import HomieMiddleware
from homie_core.knowledge.graph import KnowledgeGraph


class ContextEnricherMiddleware(HomieMiddleware):
    """Injects live context from multiple sources into the system prompt."""

    name = "context_enricher"
    order = 25

    def __init__(
        self,
        graph: Optional[KnowledgeGraph] = None,
        email_summary_fn: Optional[callable] = None,
        behavioral_summary_fn: Optional[callable] = None,
    ):
        self._graph = graph
        self._email_summary = email_summary_fn
        self._behavioral_summary = behavioral_summary_fn

    def modify_prompt(self, system_prompt: str) -> str:
        blocks: list[str] = []

        # ------------------------------------------------------------------ #
        # Graph-based active context
        # ------------------------------------------------------------------ #
        if self._graph:
            stats = self._graph.stats()
            if stats.get("entity_count", 0) > 0:
                recent = self._graph.find_entities(limit=5)
                if recent:
                    entity_lines = [
                        f"- {e.name} ({e.entity_type})" for e in recent[:5]
                    ]
                    blocks.append("Recent topics:\n" + "\n".join(entity_lines))

        # ------------------------------------------------------------------ #
        # Email context
        # ------------------------------------------------------------------ #
        if self._email_summary:
            try:
                summary = self._email_summary()
                if summary:
                    blocks.append(f"Email: {summary}")
            except Exception as e:
                logger.warning("Email context enrichment failed: %s", e)

        # ------------------------------------------------------------------ #
        # Behavioral context
        # ------------------------------------------------------------------ #
        if self._behavioral_summary:
            try:
                summary = self._behavioral_summary()
                if summary:
                    blocks.append(f"Activity: {summary}")
            except Exception as e:
                logger.warning("Behavioral context enrichment failed: %s", e)

        if blocks:
            return system_prompt + "\n[LIVE CONTEXT]\n" + "\n".join(blocks) + "\n"
        return system_prompt
