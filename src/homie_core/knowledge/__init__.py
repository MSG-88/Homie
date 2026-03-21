"""Knowledge Graph package for Homie AI.

Provides a SQLite-backed triple store for entity and relationship tracking,
with pattern-based and spaCy-powered entity extraction, email indexing,
and session persistence.
"""
from __future__ import annotations

from homie_core.knowledge.models import Entity, Relationship
from homie_core.knowledge.graph import KnowledgeGraph
from homie_core.knowledge.extractor import EntityExtractor
from homie_core.knowledge.email_indexer import EmailIndexer
from homie_core.knowledge.session_persistence import SessionPersistence

__all__ = [
    "Entity",
    "Relationship",
    "KnowledgeGraph",
    "EntityExtractor",
    "EmailIndexer",
    "SessionPersistence",
]
