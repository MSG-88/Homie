from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMADB = True
except ImportError:
    chromadb = None
    Settings = None
    _HAS_CHROMADB = False


class VectorStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._client = None
        self._episodes = None
        self._files = None

    def initialize(self) -> None:
        if not _HAS_CHROMADB:
            raise ImportError(
                "chromadb is required for vector storage. "
                "Install with: pip install homie-ai[storage]"
            )
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._episodes = self._client.get_or_create_collection("episodes")
        self._files = self._client.get_or_create_collection("file_chunks")

    def add_episode(self, episode_id: str, text: str, metadata: dict[str, str]) -> None:
        self._episodes.upsert(ids=[episode_id], documents=[text], metadatas=[metadata or None])

    def query_episodes(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        results = self._episodes.query(query_texts=[query], n_results=n)
        return self._format_results(results)

    def delete_episodes(self, ids: list[str]) -> None:
        self._episodes.delete(ids=ids)

    def add_file_chunk(self, chunk_id: str, text: str, metadata: dict[str, str]) -> None:
        self._files.upsert(ids=[chunk_id], documents=[text], metadatas=[metadata or None])

    def query_files(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        results = self._files.query(query_texts=[query], n_results=n)
        return self._format_results(results)

    def delete_file_chunks(self, ids: list[str]) -> None:
        self._files.delete(ids=ids)

    @staticmethod
    def _format_results(results: dict) -> list[dict[str, Any]]:
        out = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for i, doc_id in enumerate(results["ids"][0]):
            entry = {
                "id": doc_id,
                "text": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            }
            if results.get("distances"):
                entry["distance"] = results["distances"][0][i]
            out.append(entry)
        return out
