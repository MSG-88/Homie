from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Optional

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    _HAS_WATCHDOG = True
except ImportError:
    Observer = None
    FileSystemEventHandler = object
    FileModifiedEvent = None
    FileCreatedEvent = None
    _HAS_WATCHDOG = False


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, indexer: FileIndexer):
        self._indexer = indexer

    def on_modified(self, event):
        if not event.is_directory:
            self._indexer.queue_file(Path(event.src_path))

    def on_created(self, event):
        if not event.is_directory:
            self._indexer.queue_file(Path(event.src_path))


class FileIndexer:
    TEXT_EXTENSIONS = {".py", ".js", ".ts", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".html", ".css", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".sh", ".bat"}
    MAX_FILE_SIZE = 1_000_000  # 1MB

    def __init__(self, vector_store=None, chunk_size: int = 500):
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._observer: Optional[Observer] = None
        self._file_hashes: dict[str, str] = {}
        self._queue: list[Path] = []
        self._lock = threading.Lock()

    def watch(self, directories: list[str | Path]) -> None:
        if not _HAS_WATCHDOG:
            raise ImportError(
                "watchdog is required for file watching. "
                "Install with: pip install homie-ai[context]"
            )
        self._observer = Observer()
        handler = FileChangeHandler(self)
        for d in directories:
            d = Path(d)
            if d.exists():
                self._observer.schedule(handler, str(d), recursive=True)
        self._observer.start()

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()

    def queue_file(self, path: Path) -> None:
        with self._lock:
            if path.suffix.lower() in self.TEXT_EXTENSIONS and path.stat().st_size <= self.MAX_FILE_SIZE:
                self._queue.append(path)

    def process_queue(self) -> int:
        with self._lock:
            files = list(self._queue)
            self._queue.clear()
        indexed = 0
        for f in files:
            if self._index_file(f):
                indexed += 1
        return indexed

    def index_directory(self, directory: str | Path) -> int:
        directory = Path(directory)
        indexed = 0
        for f in directory.rglob("*"):
            if f.is_file() and f.suffix.lower() in self.TEXT_EXTENSIONS and f.stat().st_size <= self.MAX_FILE_SIZE:
                if self._index_file(f):
                    indexed += 1
        return indexed

    def _index_file(self, path: Path) -> bool:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            return False
        content_hash = hashlib.md5(content.encode()).hexdigest()
        path_str = str(path)
        if self._file_hashes.get(path_str) == content_hash:
            return False
        self._file_hashes[path_str] = content_hash
        chunks = self._chunk_text(content)
        if self._vector_store:
            for i, chunk in enumerate(chunks):
                chunk_id = f"file_{hashlib.md5(f'{path_str}_{i}'.encode()).hexdigest()[:16]}"
                self._vector_store.add_file_chunk(chunk_id, chunk, {"file_path": path_str, "chunk_index": str(i)})
        return True

    def _chunk_text(self, text: str) -> list[str]:
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_len = 0
        for line in lines:
            current_chunk.append(line)
            current_len += len(line) + 1
            if current_len >= self._chunk_size:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks
