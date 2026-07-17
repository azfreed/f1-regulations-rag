from __future__ import annotations

from f1_rag.chunking.fixed_window import FixedWindowChunker
from f1_rag.chunking.regulation import RegulationChunker, _split_overlapping


def test_split_overlapping_short_text_is_single_chunk():
    assert _split_overlapping("a b c", max_tokens=100, overlap_tokens=10) == ["a b c"]


def test_split_overlapping_long_text_overlaps():
    words = " ".join(str(i) for i in range(100))
    windows = _split_overlapping(words, max_tokens=26, overlap_tokens=13)  # ~20 words / ~10 overlap
    assert len(windows) > 1
    # Consecutive windows should share overlapping words.
    first_tail = windows[0].split()[-3:]
    assert any(w in windows[1].split() for w in first_tail)


def test_regulation_chunker_ids_deterministic_and_metadata(sample_articles, sample_meta):
    chunker = RegulationChunker(max_tokens=350, overlap_tokens=60)
    a = chunker.chunk(sample_articles, sample_meta)
    b = chunker.chunk(sample_articles, sample_meta)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]  # idempotent ids
    assert {c.article_number for c in a} == {"A1.1.1", "A2.2"}
    for c in a:
        assert c.document_title == sample_meta.document_title
        assert c.issue_number == sample_meta.issue_number


def test_fixed_window_chunker_produces_chunks(sample_articles, sample_meta):
    chunker = FixedWindowChunker(max_tokens=20, overlap_tokens=6)
    chunks = chunker.chunk(sample_articles, sample_meta)
    assert len(chunks) >= 1
    # Fixed-window ids are namespaced so they never collide with regulation ids.
    reg = RegulationChunker().chunk(sample_articles, sample_meta)
    assert not (set(c.chunk_id for c in chunks) & set(c.chunk_id for c in reg))
