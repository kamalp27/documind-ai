"""Unit tests for ChunkingService."""

import pytest
from app.services.chunking_service import ChunkingService


def test_chunking_service_initialization():
    """Verify validation of chunk size and overlap configurations."""
    service = ChunkingService(chunk_size=800, chunk_overlap=120)
    assert service.chunk_size == 800
    assert service.chunk_overlap == 120

    with pytest.raises(ValueError):
        ChunkingService(chunk_size=0)

    with pytest.raises(ValueError):
        ChunkingService(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError):
        ChunkingService(chunk_size=500, chunk_overlap=-1)


def test_split_page_text_short():
    """Verify short texts that fit within chunk_size are not divided."""
    service = ChunkingService(chunk_size=500, chunk_overlap=100)
    text = "This is a concise paragraph that fits cleanly in one chunk."
    chunks = service.split_page_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_page_text_empty():
    """Verify empty or whitespace strings produce no chunks."""
    service = ChunkingService(chunk_size=500, chunk_overlap=100)
    assert service.split_page_text("") == []
    assert service.split_page_text("    \n\t   ") == []


def test_split_page_text_overlapping():
    """Verify long text produces multiple overlapping chunks."""
    service = ChunkingService(chunk_size=200, chunk_overlap=50)
    # 500 characters
    long_text = (
        "Retrieval-Augmented Generation enhances Large Language Models by grounding outputs in retrieved documents. "
        "DocuMind AI preserves page citations and chunk metadata for verifiable answers. "
        "Vector databases like ChromaDB perform nearest neighbor semantic search with sub-second latency. "
        "This project showcases production-grade AI system architecture."
    )
    chunks = service.split_page_text(long_text)
    assert len(chunks) >= 2

    # Verify overlap exists between consecutive chunks
    for i in range(len(chunks) - 1):
        # A substring from the end of chunks[i] should appear in the beginning of chunks[i+1]
        tail = chunks[i][-30:]
        assert any(word in chunks[i + 1] for word in tail.split())


def test_create_chunks_metadata_preservation():
    """Verify create_chunks preserves page numbers and sequential chunk indexes."""
    service = ChunkingService(chunk_size=300, chunk_overlap=50)
    pages = [
        {"page_number": 1, "text": "Page one first paragraph of DocuMind architecture."},
        {"page_number": 2, "text": "Page two paragraph discussing vector embeddings and citations."},
        {"page_number": 3, "text": "   "},  # Blank page should be skipped
    ]

    chunks = service.create_chunks(pages, doc_id="doc-12345", filename="report.pdf")
    assert len(chunks) == 2
    assert chunks[0].doc_id == "doc-12345"
    assert chunks[0].filename == "report.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id == "doc-12345_p1_c0"

    assert chunks[1].doc_id == "doc-12345"
    assert chunks[1].filename == "report.pdf"
    assert chunks[1].page_number == 2
    assert chunks[1].chunk_index == 1
    assert chunks[1].chunk_id == "doc-12345_p2_c1"
