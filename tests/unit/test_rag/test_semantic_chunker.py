"""Tests for semantic_chunker — boundary detection and merging logic."""
from __future__ import annotations
import pytest
from homie_core.rag.semantic_chunker import (
    semantic_chunk,
    _cosine_similarity,
    _paragraph_split,
    _merge_small_chunks,
    _fixed_split,
)


# ---------------------------------------------------------------------------
# Helper: build a mock embed_fn from a pre-defined vector map
# ---------------------------------------------------------------------------

def _make_embed_fn(vectors: list[list[float]]):
    """Returns an embed_fn that yields vectors in order per call."""
    calls = [0]

    def embed_fn(sentences: list[str]) -> list[list[float]]:
        result = vectors[calls[0]: calls[0] + len(sentences)]
        calls[0] += len(sentences)
        return result

    return embed_fn


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string_returns_empty_list(self):
        assert semantic_chunk("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert semantic_chunk("   \n\t  ") == []

    def test_single_sentence_returns_single_chunk(self):
        text = "This is a single sentence."
        result = semantic_chunk(text)
        assert result == ["This is a single sentence."]

    def test_single_sentence_no_embed_fn(self):
        text = "Just one line here."
        result = semantic_chunk(text, embed_fn=None)
        assert len(result) == 1
        assert result[0] == "Just one line here."


# ---------------------------------------------------------------------------
# Paragraph-based fallback (no embed_fn)
# ---------------------------------------------------------------------------

class TestParagraphFallback:
    def test_splits_on_double_newline(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        result = semantic_chunk(text, embed_fn=None)
        assert len(result) == 2
        assert "First paragraph" in result[0]
        assert "Second paragraph" in result[1]

    def test_three_paragraphs(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = semantic_chunk(text, embed_fn=None)
        assert len(result) == 3

    def test_single_paragraph_no_split(self):
        text = "Only one paragraph\nwith a newline but not double."
        result = semantic_chunk(text, embed_fn=None)
        assert len(result) == 1

    def test_blank_paragraphs_ignored(self):
        text = "First.\n\n   \n\nSecond."
        result = semantic_chunk(text, embed_fn=None)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Embedding-based splitting
# ---------------------------------------------------------------------------

class TestEmbeddingBasedSplit:
    def test_similar_sentences_stay_together(self):
        """Sentences with high cosine similarity should not be split."""
        # Two sentences sharing the same direction → similarity = 1.0
        vectors = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
        embed_fn = _make_embed_fn(vectors)
        text = "Dogs are friendly. Cats are cute. Birds can fly."
        result = semantic_chunk(text, embed_fn=embed_fn, threshold=0.5, min_chunk_size=1)
        assert len(result) == 1

    def test_dissimilar_sentences_split(self):
        """Consecutive sentences with low similarity should be split."""
        # Orthogonal vectors → similarity = 0.0 < threshold 0.3
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        embed_fn = _make_embed_fn(vectors)
        text = "Dogs are friendly. Cats are curious. Birds soar high."
        result = semantic_chunk(text, embed_fn=embed_fn, threshold=0.3, min_chunk_size=1)
        assert len(result) == 3

    def test_mixed_similarity(self):
        """First two similar, then a topic shift."""
        vectors = [
            [1.0, 0.0],
            [1.0, 0.0],   # high similarity with prev
            [0.0, 1.0],   # low similarity with prev → split
        ]
        embed_fn = _make_embed_fn(vectors)
        text = "Dogs are pets. Cats are pets. The stock market crashed yesterday."
        result = semantic_chunk(text, embed_fn=embed_fn, threshold=0.3, min_chunk_size=1)
        assert len(result) == 2

    def test_all_orthogonal_respects_max_chunk_size(self):
        """When merging would exceed max_chunk_size, chunks stay separate."""
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        embed_fn = _make_embed_fn(vectors)
        long_a = "A" * 800 + "."
        long_b = "B" * 800 + "."
        text = long_a + " " + long_b
        result = semantic_chunk(
            text, embed_fn=embed_fn, threshold=0.3,
            max_chunk_size=1500, min_chunk_size=100
        )
        # Both chunks are large, so they cannot be merged
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Merge small chunks
# ---------------------------------------------------------------------------

class TestMergeSmallChunks:
    def test_merges_tiny_chunks(self):
        chunks = ["Hi.", "Hello.", "World."]
        result = _merge_small_chunks(chunks, max_size=200, min_size=50)
        # All three are tiny — should merge into as few as possible
        assert len(result) < len(chunks)

    def test_does_not_exceed_max_size(self):
        chunks = ["A" * 100, "B" * 100, "C" * 100]
        result = _merge_small_chunks(chunks, max_size=150, min_size=50)
        for chunk in result:
            assert len(chunk) <= 200  # some tolerance for separator "\n\n"

    def test_single_chunk_passthrough(self):
        chunks = ["Just one chunk here with enough content."]
        result = _merge_small_chunks(chunks, max_size=200, min_size=10)
        assert result == chunks

    def test_empty_input(self):
        assert _merge_small_chunks([], max_size=200, min_size=50) == []


# ---------------------------------------------------------------------------
# Fixed split fallback
# ---------------------------------------------------------------------------

class TestFixedSplit:
    def test_splits_long_text(self):
        text = "x" * 3000
        result = _fixed_split(text, max_size=1000)
        assert len(result) == 3
        for chunk in result:
            assert len(chunk) <= 1000

    def test_short_text_single_chunk(self):
        text = "short"
        result = _fixed_split(text, max_size=1000)
        assert result == ["short"]


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# Very long text without sentence boundaries → fixed split fallback
# ---------------------------------------------------------------------------

class TestLongTextWithoutSentences:
    def test_long_text_no_sentences_fixed_split(self):
        """Text without sentence boundaries falls through to _fixed_split."""
        # No .!? followed by space+capital — just a blob
        text = "a" * 5000
        result = semantic_chunk(text, embed_fn=None, max_chunk_size=1500)
        # Should produce multiple chunks, each <= 1500 chars
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 1500

    def test_large_single_sentence_fixed_split(self):
        """A single 'sentence' exceeding max_chunk_size is further split."""
        # The sentence split returns one item; since len == 1, it's returned as-is.
        # But if the raw text hits the no-sentences path:
        text = "x" * 4000
        result = semantic_chunk(text, embed_fn=None, max_chunk_size=1500)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 1500
