"""Entity extractor for the Homie Knowledge Graph.

Two-tier extraction strategy:
1. Pattern-based (always available): regex patterns for emails, URLs, file
   paths, import statements, dates, and capitalized phrases.
2. spaCy NER (optional): PERSON, ORG, GPE/LOC, DATE entities when
   en_core_web_sm is installed.

Both tiers are merged; duplicates by (name, entity_type) are deduplicated.
"""
from __future__ import annotations

import re
from typing import Optional

from homie_core.knowledge.models import Entity, Relationship

# ---------------------------------------------------------------------------
# Regex patterns for pattern-based extraction
# ---------------------------------------------------------------------------

_RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
_RE_URL = re.compile(r"https?://[^\s\"'<>]+")
_RE_FILE_PATH = re.compile(
    r"(?:"
    r"(?:[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*)"  # Windows
    r"|(?:/(?:[^/\s]+/)*[^/\s]+\.[a-zA-Z0-9]{1,10})"                  # Unix abs
    r"|(?:\.{1,2}/(?:[^\s/]+/)*[^\s/]+\.[a-zA-Z0-9]{1,10})"           # relative
    r")"
)
_RE_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+([\w.]+)", re.MULTILINE
)
_RE_DATE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}"          # ISO: 2024-01-15
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"             # US/EU: 15/01/2024
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"[\s.,]+\d{1,2}[,\s]+\d{4}"                  # Jan 15, 2024
    r")\b",
    re.IGNORECASE,
)
_RE_CAPITALIZED = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


class EntityExtractor:
    """Extract entities and relationships from free text.

    Parameters
    ----------
    use_model:
        If True (default), attempt to load spaCy ``en_core_web_sm``.
        Falls back to pattern-only extraction silently if the model is
        unavailable.
    """

    def __init__(self, use_model: bool = True) -> None:
        self._nlp = None
        if use_model:
            try:
                import spacy  # type: ignore

                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, text: str, source: str = ""
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from *text*.

        Returns a tuple of (entities, relationships).  Relationship
        extraction is currently limited to co-occurrence within the same
        sentence.
        """
        if self._nlp:
            return self._extract_with_model(text, source)
        return self._extract_with_patterns(text, source)

    # ------------------------------------------------------------------
    # Pattern-based extraction
    # ------------------------------------------------------------------

    def _extract_with_patterns(
        self, text: str, source: str
    ) -> tuple[list[Entity], list[Relationship]]:
        entities: list[Entity] = []
        seen: set[tuple[str, str]] = set()

        def _add(name: str, etype: str, extra: Optional[dict] = None) -> Optional[Entity]:
            key = (name.lower(), etype)
            if key in seen:
                return None
            seen.add(key)
            e = Entity(
                name=name,
                entity_type=etype,
                source=source or "extraction",
                attributes=extra or {},
                confidence=0.7,
            )
            entities.append(e)
            return e

        # Email addresses -> Person
        for m in _RE_EMAIL.finditer(text):
            _add(m.group(), "person", {"email": m.group()})

        # URLs -> Location
        for m in _RE_URL.finditer(text):
            _add(m.group(), "location", {"url": m.group()})

        # File paths -> Document
        for m in _RE_FILE_PATH.finditer(text):
            _add(m.group(), "document", {"path": m.group()})

        # import statements -> Tool
        for m in _RE_IMPORT.finditer(text):
            pkg = m.group(1).split(".")[0]
            if pkg and pkg not in {"__future__", "typing"}:
                _add(pkg, "tool")

        # Dates -> Event
        for m in _RE_DATE.finditer(text):
            _add(m.group().strip(), "event", {"date_text": m.group()})

        # Capitalized multi-word phrases -> potential Person/Concept
        for m in _RE_CAPITALIZED.finditer(text):
            phrase = m.group(1)
            # Skip if already captured
            if (phrase.lower(), "person") not in seen and (phrase.lower(), "concept") not in seen:
                _add(phrase, "person", {"inferred": True})

        relationships: list[Relationship] = []
        return entities, relationships

    # ------------------------------------------------------------------
    # spaCy NER extraction
    # ------------------------------------------------------------------

    def _extract_with_model(
        self, text: str, source: str
    ) -> tuple[list[Entity], list[Relationship]]:
        """Run spaCy NER, then merge with pattern-based results."""
        assert self._nlp is not None

        entities: list[Entity] = []
        seen: set[tuple[str, str]] = set()

        def _add(name: str, etype: str, extra: Optional[dict] = None) -> Optional[Entity]:
            key = (name.lower(), etype)
            if key in seen:
                return None
            seen.add(key)
            e = Entity(
                name=name,
                entity_type=etype,
                source=source or "extraction",
                attributes=extra or {},
                confidence=0.85,
            )
            entities.append(e)
            return e

        doc = self._nlp(text)
        for ent in doc.ents:
            label = ent.label_
            name = ent.text.strip()
            if not name:
                continue
            if label == "PERSON":
                _add(name, "person")
            elif label == "ORG":
                _add(name, "concept", {"org": True})
            elif label in ("GPE", "LOC"):
                _add(name, "location")
            elif label == "DATE":
                _add(name, "event", {"date_text": name})

        # Merge in pattern-based results (fills in emails, URLs, etc.)
        pat_entities, pat_rels = self._extract_with_patterns(text, source)
        for pe in pat_entities:
            key = (pe.name.lower(), pe.entity_type)
            if key not in seen:
                seen.add(key)
                entities.append(pe)

        relationships: list[Relationship] = []
        return entities, relationships
