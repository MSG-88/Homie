"""RAG Pipeline — Retrieval-Augmented Generation for Homie.

The pipeline orchestrates the full flow:
1. INGEST — chunk documents, index in hybrid search
2. RETRIEVE — query hybrid search, rerank by relevance
3. AUGMENT — build context from retrieved chunks with source attribution
4. GENERATE — (handled by CognitiveArchitecture, this module provides context)

Key design:
- Lazy indexing: files are indexed on first query or explicit index call
- Incremental updates: only re-index changed files (hash-based tracking)
- Source attribution: every piece of context traces back to file:line
- Budget-aware: retrieval respects the cognitive architecture's char budget
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from homie_core.rag.chunker import Chunk, auto_chunk
from homie_core.rag.hybrid_search import HybridSearch
from homie_core.rag.format_detector import detect_format, DocumentFormat
from homie_core.rag.parsers import PARSER_REGISTRY, ParsedDocument
# Import all parsers to trigger registration
import homie_core.rag.parsers.text
# Knowledge graph integration (optional)
try:
    from homie_core.knowledge.graph import KnowledgeGraph as _KnowledgeGraph
    from homie_core.knowledge.extractor import EntityExtractor as _EntityExtractor
    _KG_AVAILABLE = True
except ImportError:
    _KG_AVAILABLE = False
import homie_core.rag.parsers.pdf
import homie_core.rag.parsers.docx
import homie_core.rag.parsers.xlsx
import homie_core.rag.parsers.pptx
import homie_core.rag.parsers.html
import homie_core.rag.parsers.image
import homie_core.rag.parsers.email_parser
import homie_core.rag.parsers.code
import homie_core.rag.parsers.epub


# Extensions supported via the legacy text-read path (kept for compatibility)
_TEXT_READABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".c", ".cpp", ".h",
    ".java", ".rb", ".sh", ".bat",
    ".md", ".mdx", ".rst", ".txt",
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".html", ".css", ".xml", ".csv",
}
# All formats we're willing to index (expanded to include new parser formats)
_INDEXABLE_EXTENSIONS = _TEXT_READABLE_EXTENSIONS | {
    ".pdf", ".docx", ".xlsx", ".xls", ".pptx",
    ".eml", ".msg", ".epub",
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp",
    ".ps1", ".swift", ".kt", ".scala", ".r", ".sql", ".lua", ".php",
    ".pl", ".ex", ".exs", ".zig", ".nim", ".scss", ".less", ".vue", ".svelte",
    ".log", ".htm",
}
_MAX_FILE_SIZE = 2_000_000  # 2MB

# Formats that are binary/structured — skip large-file check differently
_BINARY_FORMATS = {
    DocumentFormat.PDF, DocumentFormat.DOCX, DocumentFormat.XLSX,
    DocumentFormat.PPTX, DocumentFormat.IMAGE, DocumentFormat.EPUB,
    DocumentFormat.EMAIL,
}


@dataclass
class RetrievedContext:
    """A piece of retrieved context with source attribution."""
    text: str
    source: str          # file path
    start_line: int
    end_line: int
    chunk_type: str
    relevance_score: float
    parent_section: str = ""

    def to_attributed_text(self) -> str:
        """Format with source attribution for the prompt."""
        loc = f"{Path(self.source).name}:{self.start_line}-{self.end_line}"
        if self.parent_section:
            loc = f"{loc} ({self.parent_section})"
        return f"[{loc}]\n{self.text}"


class RagPipeline:
    """Full RAG pipeline: ingest, retrieve, augment.

    Integrates with HybridSearch for retrieval and the FileIndexer's
    watchdog for real-time updates.
    """

    def __init__(
        self,
        vector_store=None,
        max_chunk_size: int = 1500,
        chunk_overlap: int = 100,
        knowledge_graph: Optional[Any] = None,
    ):
        self._search = HybridSearch(vector_store=vector_store)
        self._max_chunk_size = max_chunk_size
        self._chunk_overlap = chunk_overlap
        self._file_hashes: dict[str, str] = {}  # path -> content hash
        self._chunk_map: dict[str, Chunk] = {}   # chunk_id -> Chunk (for metadata)
        self._file_chunks: dict[str, list[str]] = {}  # file_path -> [chunk_ids]
        self._lock = threading.Lock()
        self._indexed_dirs: set[str] = set()
        # Optional knowledge graph integration
        self._knowledge_graph = knowledge_graph
        self._entity_extractor = (
            _EntityExtractor(use_model=True) if (_KG_AVAILABLE and knowledge_graph is not None) else None
        )

    # ------------------------------------------------------------------
    # INGEST
    # ------------------------------------------------------------------

    def _parse_any_document(self, path: Path) -> str:
        """Extract text content from any supported document format.

        Uses the format detector + parser registry for structured formats.
        Falls back to plain text reading for code/text/unknown formats.
        Returns empty string on failure.
        """
        fmt = detect_format(path)

        # Try the registered parser first (covers PDF, DOCX, XLSX, etc.)
        parser = PARSER_REGISTRY.get(fmt.value)
        if parser is not None:
            try:
                doc: ParsedDocument = parser(path)
                content = doc.full_text
                if content.strip():
                    return content
            except Exception:
                pass  # fall through to text read

        # Binary formats (images, PDF, DOCX, etc.) should not be text-read;
        # if the parser failed or returned empty, we give up.
        if fmt in _BINARY_FORMATS:
            return ""

        # For code, markdown, text, CSV, HTML, and unknown: read as plain text
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            return ""

    def index_file(self, path: str | Path) -> int:
        """Index a single file. Returns number of chunks indexed."""
        path = Path(path)
        if not path.is_file():
            return 0
        if path.suffix.lower() not in _INDEXABLE_EXTENSIONS:
            return 0
        if path.stat().st_size > _MAX_FILE_SIZE:
            return 0

        try:
            content = self._parse_any_document(path)
        except Exception:
            return 0

        # Backward compatibility: if new parser returned nothing, fall back
        if not content.strip() and path.suffix.lower() in _TEXT_READABLE_EXTENSIONS:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                return 0

        if not content.strip():
            return 0

        # Skip if unchanged
        content_hash = hashlib.md5(content.encode()).hexdigest()
        path_str = str(path)
        if self._file_hashes.get(path_str) == content_hash:
            return 0

        with self._lock:
            # Remove old chunks for this file
            self._remove_file_chunks(path_str)

            # Chunk the file
            chunks = auto_chunk(content, source=path_str,
                                max_chunk=self._max_chunk_size, overlap=self._chunk_overlap)

            # Index each chunk
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"rag_{hashlib.md5(f'{path_str}:{i}:{content_hash[:8]}'.encode()).hexdigest()[:16]}"
                metadata = {
                    "file_path": path_str,
                    "chunk_index": i,
                    "chunk_type": chunk.chunk_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                }
                if chunk.parent_section:
                    metadata["parent_section"] = chunk.parent_section

                self._search.index_chunk(chunk_id, chunk.to_search_text(), metadata)
                self._chunk_map[chunk_id] = chunk
                chunk_ids.append(chunk_id)

            self._file_chunks[path_str] = chunk_ids
            self._file_hashes[path_str] = content_hash

            # Feed extracted entities into the knowledge graph (if configured)
            if self._knowledge_graph is not None and self._entity_extractor is not None:
                try:
                    entities, relationships = self._entity_extractor.extract(
                        content, source=path_str
                    )
                    for entity in entities:
                        self._knowledge_graph.merge_entity(entity)
                    for rel in relationships:
                        self._knowledge_graph.add_relationship(rel)
                except Exception:
                    pass  # Knowledge graph errors must never break indexing

        return len(chunks)

    def index_directory(self, directory: str | Path, recursive: bool = True) -> int:
        """Index all files in a directory. Returns total chunks indexed."""
        directory = Path(directory)
        if not directory.is_dir():
            return 0

        self._indexed_dirs.add(str(directory))
        total = 0
        glob_fn = directory.rglob if recursive else directory.glob

        for path in glob_fn("*"):
            if path.is_file():
                total += self.index_file(path)

        return total

    def remove_file(self, path: str | Path) -> None:
        """Remove a file from the index."""
        path_str = str(path)
        with self._lock:
            self._remove_file_chunks(path_str)
            self._file_hashes.pop(path_str, None)

    def _remove_file_chunks(self, path_str: str) -> None:
        """Remove all chunks for a file (internal, must hold lock)."""
        old_ids = self._file_chunks.pop(path_str, [])
        for cid in old_ids:
            self._search.remove_chunk(cid)
            self._chunk_map.pop(cid, None)

    # ------------------------------------------------------------------
    # RETRIEVE
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_chars: int = 3000,
        file_filter: Optional[str] = None,
    ) -> list[RetrievedContext]:
        """Retrieve relevant document chunks for a query.

        Args:
            query: The user's query
            top_k: Maximum number of chunks to return
            max_chars: Character budget for retrieved context
            file_filter: Optional glob pattern to filter by file path

        Returns:
            List of RetrievedContext with source attribution
        """
        results = self._search.search(query, top_k=top_k * 2)

        # Apply file filter if specified
        if file_filter:
            import fnmatch
            results = [
                r for r in results
                if fnmatch.fnmatch(r.get("metadata", {}).get("file_path", ""), file_filter)
            ]

        # Build RetrievedContext with budget
        contexts = []
        used_chars = 0

        for r in results[:top_k]:
            chunk = self._chunk_map.get(r["id"])
            meta = r.get("metadata", {})
            text = r.get("text", "")

            if not text and chunk:
                text = chunk.text

            if not text:
                continue

            if used_chars + len(text) > max_chars:
                # Truncate last chunk to fit budget
                remaining = max_chars - used_chars
                if remaining > 100:
                    text = text[:remaining] + "..."
                else:
                    break

            ctx = RetrievedContext(
                text=text,
                source=meta.get("file_path", chunk.source if chunk else ""),
                start_line=int(meta.get("start_line", chunk.start_line if chunk else 0)),
                end_line=int(meta.get("end_line", chunk.end_line if chunk else 0)),
                chunk_type=meta.get("chunk_type", chunk.chunk_type if chunk else "text"),
                relevance_score=r.get("rrf_score", r.get("score", 0.0)),
                parent_section=meta.get("parent_section", chunk.parent_section if chunk else ""),
            )
            contexts.append(ctx)
            used_chars += len(text)

        return contexts

    # ------------------------------------------------------------------
    # AUGMENT — format retrieved context for prompt injection
    # ------------------------------------------------------------------

    def build_context_block(
        self,
        query: str,
        max_chars: int = 3000,
        top_k: int = 5,
    ) -> str:
        """Retrieve and format context for prompt injection.

        Returns a structured [DOCUMENTS] block ready to insert into
        the cognitive architecture's prompt.
        """
        contexts = self.retrieve(query, top_k=top_k, max_chars=max_chars)
        if not contexts:
            return ""

        lines = ["[DOCUMENTS]"]
        for ctx in contexts:
            lines.append(ctx.to_attributed_text())
            lines.append("")  # blank line separator

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return indexing statistics."""
        return {
            "indexed_files": len(self._file_hashes),
            "total_chunks": self._search.size,
            "indexed_dirs": list(self._indexed_dirs),
        }
