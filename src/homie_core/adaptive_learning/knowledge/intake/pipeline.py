"""Intake pipeline — coordinates scanning, extraction, and graph ingestion."""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from ..graph.store import KnowledgeGraphStore
from .deep_analyzer import DeepAnalyzer
from .scanner import SourceScanner
from .surface_extractor import SurfaceExtractor
from .value_scorer import ValueScorer

logger = logging.getLogger(__name__)


class IntakePipeline:
    """Orchestrates guided intake: scan -> surface extract -> score -> deep analyze -> graph."""

    def __init__(
        self,
        graph_store: KnowledgeGraphStore,
        inference_fn: Optional[Callable[[str], str]] = None,
        deep_pass_top_percent: int = 20,
        max_deep_files: int = 50,
    ) -> None:
        self._graph = graph_store
        self._scanner = SourceScanner()
        self._surface = SurfaceExtractor()
        self._scorer = ValueScorer(top_percent=deep_pass_top_percent)
        self._deep = DeepAnalyzer(inference_fn=inference_fn)
        self._max_deep = max_deep_files
        self._has_llm = inference_fn is not None

    def ingest(self, source: Path | str) -> dict[str, Any]:
        """Ingest a directory or file into the knowledge graph."""
        source = Path(source)
        result = {"files_scanned": 0, "entities_created": 0, "relationships_created": 0, "deep_analyzed": 0}

        # Scan
        if source.is_file():
            from .scanner import FileInfo
            files = [FileInfo(path=source, file_type=self._detect_type(source), size_bytes=source.stat().st_size, extension=source.suffix)]
        elif source.is_dir():
            files = self._scanner.scan_directory(source)
        else:
            return result

        result["files_scanned"] = len(files)

        # Surface pass
        extractions = []
        for fi in files:
            ext = self._surface.extract(fi.path, fi.file_type)
            extractions.append(ext)
            # Add entities to graph
            for entity in ext.get("entities", []):
                self._add_entity_to_graph(entity)
                result["entities_created"] += 1

        # Deep pass (if LLM available)
        if self._has_llm and extractions:
            selected = self._scorer.select_for_deep_pass(extractions)[:self._max_deep]
            for ext in selected:
                file_path = ext.get("file", "")
                try:
                    content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                    deep = self._deep.analyze(content, file_path=file_path)
                    for entity in deep.get("entities", []):
                        self._add_entity_to_graph(entity)
                        result["entities_created"] += 1
                    for rel in deep.get("relationships", []):
                        self._add_relationship_to_graph(rel)
                        result["relationships_created"] += 1
                    result["deep_analyzed"] += 1
                except Exception:
                    logger.warning("Deep analysis failed for %s", file_path)

        return result

    def _add_entity_to_graph(self, entity: dict) -> Optional[str]:
        """Add an entity to the graph, deduplicating by name."""
        name = entity.get("name", "")
        if not name:
            return None
        existing = self._graph.find_entity_by_name(name)
        if existing:
            return existing["id"]
        return self._graph.add_entity(
            name=name,
            entity_type=entity.get("type", "concept"),
        )

    def _add_relationship_to_graph(self, rel: dict) -> None:
        """Add a relationship to the graph."""
        subject_name = rel.get("subject", "")
        object_name = rel.get("object", "")
        predicate = rel.get("predicate", "related_to")
        if not subject_name or not object_name:
            return

        sub = self._graph.find_entity_by_name(subject_name)
        obj = self._graph.find_entity_by_name(object_name)
        if not sub:
            sub_id = self._graph.add_entity(subject_name, "concept")
        else:
            sub_id = sub["id"]
        if not obj:
            obj_id = self._graph.add_entity(object_name, "concept")
        else:
            obj_id = obj["id"]

        self._graph.add_relationship(sub_id, predicate, obj_id, confidence=0.75, source="deep_analysis")

    def _detect_type(self, path: Path) -> str:
        from .scanner import _EXT_TO_TYPE
        return _EXT_TO_TYPE.get(path.suffix.lower(), "unknown")
