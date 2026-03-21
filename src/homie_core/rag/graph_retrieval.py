"""Graph-expanded retrieval for RAG pipeline.

Augments retrieval results by injecting knowledge-graph context for entities
mentioned in the query — up to a configurable budget of extra chunks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from homie_core.knowledge.graph import KnowledgeGraph


@dataclass
class RetrievalChunk:
    """A single retrieved piece of content."""

    content: str
    source: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


def graph_expand(
    query: str,
    chunks: list[RetrievalChunk],
    graph: Optional[KnowledgeGraph],
    budget: int = 3,
) -> list[RetrievalChunk]:
    """Expand retrieval results using knowledge graph.

    Algorithm:
    1. Find entities mentioned in query (substring match against entity names).
    2. For each entity, generate a natural-language context summary.
    3. Add as synthetic RetrievalChunk objects — up to *budget* extra chunks.
    4. Original chunks are always returned first; expansions are appended.

    Args:
        query: The user query string.
        chunks: Existing retrieval results.
        graph: KnowledgeGraph instance, or None to skip expansion entirely.
        budget: Maximum number of graph-context chunks to add.

    Returns:
        The original chunks list (unchanged) plus any added graph context chunks.
        If graph is None or no entities are found, returns the original list unchanged.
    """
    if graph is None or not chunks:
        return chunks

    mentioned = graph.entities_mentioned_in(query)
    if not mentioned:
        return chunks

    expanded = list(chunks)
    added = 0
    for entity in mentioned:
        if added >= budget:
            break
        context = graph.context_for_entity(entity.id)
        if context and context.strip():
            expanded.append(
                RetrievalChunk(
                    content=context,
                    source=f"knowledge_graph:{entity.name}",
                    score=entity.confidence * 0.8,  # slightly lower than direct hits
                    metadata={
                        "entity_id": entity.id,
                        "entity_type": entity.entity_type,
                    },
                )
            )
            added += 1

    return expanded
