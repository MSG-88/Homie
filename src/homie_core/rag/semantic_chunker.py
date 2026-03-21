"""Semantic Chunker — splits text at semantic boundaries.

Uses embedding similarity between consecutive sentences to detect topic
shifts. Falls back to paragraph-based splitting when no embed_fn provided.
"""
from __future__ import annotations
import re
from typing import Callable, Optional

# Sentence splitting pattern: split after .!? followed by whitespace and capital
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def semantic_chunk(
    text: str,
    embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    threshold: float = 0.3,
    max_chunk_size: int = 1500,
    min_chunk_size: int = 100,
) -> list[str]:
    """Split text at semantic boundaries.

    1. Split into sentences
    2. If embed_fn provided: embed each, compute cosine similarity between consecutive
       sentences, split where similarity drops below threshold
    3. If no embed_fn: fall back to paragraph-based splitting
    4. Merge small chunks up to max_chunk_size

    Returns list of chunk strings.
    """
    if not text.strip():
        return []

    # Split into sentences
    sentences = _SENT_SPLIT.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text] if len(text) <= max_chunk_size else _fixed_split(text, max_chunk_size)

    if len(sentences) <= 1:
        # Even with a single sentence, try paragraph-based splitting first
        # (double newlines won't be caught by sentence regex)
        if '\n' in text:
            para_result = _paragraph_split(text, max_chunk_size, min_chunk_size)
            if len(para_result) > 1:
                return para_result
        # If still one chunk and it's too large, fixed-split it
        if len(sentences[0]) > max_chunk_size:
            return _fixed_split(sentences[0], max_chunk_size)
        return sentences

    if embed_fn is None:
        # Fallback: split by paragraphs (double newline)
        return _paragraph_split(text, max_chunk_size, min_chunk_size)

    # Embed all sentences
    embeddings = embed_fn(sentences)

    # Find split points where cosine similarity drops below threshold
    split_indices = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        if sim < threshold:
            split_indices.append(i + 1)

    # Build chunks from split points
    chunks = []
    start = 0
    for idx in split_indices:
        chunk = " ".join(sentences[start:idx])
        if chunk.strip():
            chunks.append(chunk)
        start = idx
    # Last chunk
    last = " ".join(sentences[start:])
    if last.strip():
        chunks.append(last)

    # Merge small chunks
    return _merge_small_chunks(chunks, max_chunk_size, min_chunk_size)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _paragraph_split(text: str, max_size: int, min_size: int) -> list[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    # Further split any oversized paragraphs via fixed split
    split_paragraphs: list[str] = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(p) > max_size:
            split_paragraphs.extend(_fixed_split(p, max_size))
        else:
            split_paragraphs.append(p)
    # Paragraph boundaries are intentional — use min_size=0 so we don't
    # collapse explicit paragraph separations for merging purposes.
    return _merge_small_chunks(split_paragraphs, max_size, min_size=0)


def _merge_small_chunks(chunks: list[str], max_size: int, min_size: int) -> list[str]:
    if not chunks:
        return []
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if len(merged[-1]) + len(chunk) + 1 <= max_size and len(merged[-1]) < min_size:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)
    return merged


def _fixed_split(text: str, max_size: int) -> list[str]:
    chunks = []
    for i in range(0, len(text), max_size):
        chunks.append(text[i:i + max_size])
    return chunks
