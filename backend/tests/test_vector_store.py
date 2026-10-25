"""Unit tests for ChromaVectorStore."""

import pytest
from app.core.exceptions import StorageException
from app.schemas.document import TextChunk
from app.services.vector_store import ChromaVectorStore


def test_vector_store_add_and_delete_chunks(test_vector_store: ChromaVectorStore):
    """Verify adding, querying, and deleting chunks in ChromaDB."""
    chunks = [
        TextChunk(
            chunk_id="test-doc_p1_c0",
            doc_id="test-doc",
            filename="sample.pdf",
            text="First chunk for testing vector store.",
            page_number=1,
            chunk_index=0,
            char_count=39,
        ),
        TextChunk(
            chunk_id="test-doc_p2_c1",
            doc_id="test-doc",
            filename="sample.pdf",
            text="Second chunk for testing vector store.",
            page_number=2,
            chunk_index=1,
            char_count=40,
        ),
    ]
    # 384-dimensional synthetic embeddings
    embeddings = [[0.01 * i for _ in range(384)] for i in range(2)]

    # Add chunks
    test_vector_store.add_chunks(chunks, embeddings)
    count = test_vector_store.get_chunk_count_for_document("test-doc")
    assert count == 2

    # Delete chunks
    test_vector_store.delete_document_chunks("test-doc")
    count_after = test_vector_store.get_chunk_count_for_document("test-doc")
    assert count_after == 0


def test_vector_store_count_nonexistent(test_vector_store: ChromaVectorStore):
    """Verify counting a nonexistent document returns 0 without error."""
    count = test_vector_store.get_chunk_count_for_document("non-existent-doc-id")
    assert count == 0


def test_vector_store_mismatched_counts(test_vector_store: ChromaVectorStore):
    """Verify mismatch between chunk count and embedding count raises StorageException."""
    chunks = [
        TextChunk(
            chunk_id="doc1_p1_c0",
            doc_id="doc1",
            filename="a.pdf",
            text="Single chunk text",
            page_number=1,
            chunk_index=0,
            char_count=17,
        )
    ]
    embeddings = []  # Empty embeddings

    with pytest.raises(StorageException):
        test_vector_store.add_chunks(chunks, embeddings)
