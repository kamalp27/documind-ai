"""Unit tests for semantic retrieval and grounded prompt construction."""

import pytest
from app.core.exceptions import DocumentNotFoundException
from app.schemas.document import DocumentMetadata, TextChunk
from app.services.metadata_store import DocumentMetadataStore
from app.services.prompt_service import PromptService
from app.services.retrieval_service import RetrievedPassage, RetrievalService
from app.services.vector_store import ChromaVectorStore


def test_retrieval_empty_collection(test_retrieval_service: RetrievalService):
    """Verify retrieval against an empty vector collection returns empty list without error."""
    passages = test_retrieval_service.retrieve(question="What is the architecture?", top_k=4)
    assert passages == []


def test_retrieval_with_indexed_chunks(
    test_retrieval_service: RetrievalService,
    test_vector_store: ChromaVectorStore,
    test_metadata_store: DocumentMetadataStore,
):
    """Verify retrieval finds relevant chunks and assigns similarity scores."""
    # 1. Add document metadata
    doc_meta = DocumentMetadata(
        document_id="doc-test-rag",
        filename="rag_guide.pdf",
        file_size_bytes=1024,
        total_pages=2,
        total_chunks=2,
        created_at="2026-10-25T18:00:00Z",
    )
    test_metadata_store.save_document(doc_meta)

    # 2. Add chunks and embeddings to vector store
    chunks = [
        TextChunk(
            chunk_id="doc-test-rag_p1_c0",
            doc_id="doc-test-rag",
            filename="rag_guide.pdf",
            text="DocuMind AI uses local ChromaDB for vector retrieval.",
            page_number=1,
            chunk_index=0,
            char_count=53,
        ),
        TextChunk(
            chunk_id="doc-test-rag_p2_c1",
            doc_id="doc-test-rag",
            filename="rag_guide.pdf",
            text="Exact page citations are generated from chunk metadata.",
            page_number=2,
            chunk_index=1,
            char_count=56,
        ),
    ]
    # Synthetic vectors that match mock embedding generator pattern
    embeddings = [[0.05 * (i % 10) for _ in range(384)] for i in range(2)]
    test_vector_store.add_chunks(chunks, embeddings)

    # 3. Retrieve
    passages = test_retrieval_service.retrieve(question="How does ChromaDB work?", top_k=2)
    assert len(passages) >= 1
    assert passages[0].doc_id == "doc-test-rag"
    assert passages[0].filename == "rag_guide.pdf"
    assert passages[0].page_numbers == [1]
    assert passages[0].similarity_score >= 0.0


def test_retrieval_filter_by_document_id(
    test_retrieval_service: RetrievalService,
    test_vector_store: ChromaVectorStore,
    test_metadata_store: DocumentMetadataStore,
):
    """Verify document_ids filter restricts retrieval exclusively to the specified document."""
    # Register two documents
    test_metadata_store.save_document(
        DocumentMetadata(
            document_id="doc-A",
            filename="doc_a.pdf",
            file_size_bytes=500,
            total_pages=1,
            total_chunks=1,
            created_at="2026-10-25T18:00:00Z",
        )
    )
    test_metadata_store.save_document(
        DocumentMetadata(
            document_id="doc-B",
            filename="doc_b.pdf",
            file_size_bytes=500,
            total_pages=1,
            total_chunks=1,
            created_at="2026-10-25T18:00:00Z",
        )
    )

    chunks = [
        TextChunk(
            chunk_id="doc-A_p1_c0",
            doc_id="doc-A",
            filename="doc_a.pdf",
            text="Document A content about neural networks.",
            page_number=1,
            chunk_index=0,
            char_count=41,
        ),
        TextChunk(
            chunk_id="doc-B_p1_c0",
            doc_id="doc-B",
            filename="doc_b.pdf",
            text="Document B content about database indexes.",
            page_number=1,
            chunk_index=0,
            char_count=42,
        ),
    ]
    embeddings = [[0.05 * (i % 10) for _ in range(384)] for i in range(2)]
    test_vector_store.add_chunks(chunks, embeddings)

    # Filter for doc-B only
    passages = test_retrieval_service.retrieve(
        question="What is this about?",
        document_ids=["doc-B"],
    )
    assert len(passages) == 1
    assert passages[0].doc_id == "doc-B"
    assert passages[0].filename == "doc_b.pdf"


def test_retrieval_unknown_document_id_raises_not_found(test_retrieval_service: RetrievalService):
    """Verify requesting an unknown document ID raises DocumentNotFoundException."""
    with pytest.raises(DocumentNotFoundException) as exc_info:
        test_retrieval_service.retrieve(
            question="Any query",
            document_ids=["non-existent-uuid"],
        )
    assert "non-existent-uuid" in str(exc_info.value)


def test_prompt_builder_structure():
    """Verify prompt builder encloses context in XML tags and preserves citation mappings."""
    service = PromptService()
    passages = [
        RetrievedPassage(
            chunk_id="c1",
            doc_id="d1",
            filename="overview.pdf",
            page_numbers=[1],
            text="DocuMind AI provides grounded question answering.",
            distance=0.15,
            similarity_score=0.85,
        ),
        RetrievedPassage(
            chunk_id="c2",
            doc_id="d1",
            filename="overview.pdf",
            page_numbers=[2],
            text="All citations reference page numbers.",
            distance=0.25,
            similarity_score=0.75,
        ),
    ]

    system_inst, user_prompt, sources = service.build_prompt("How does it work?", passages)

    assert "DocuMind AI" in system_inst
    assert "PROMPT INJECTION DEFENSE" in system_inst
    assert "<document_context>" in user_prompt
    assert "</document_context>" in user_prompt
    assert "[Source 1] (Document: overview.pdf, Page: 1)" in user_prompt
    assert "[Source 2] (Document: overview.pdf, Page: 2)" in user_prompt
    assert "User Question: How does it work?" in user_prompt

    assert len(sources) == 2
    assert sources[0].source_id == 1
    assert sources[0].filename == "overview.pdf"
    assert sources[0].page_numbers == [1]
    assert sources[1].source_id == 2
    assert sources[1].page_numbers == [2]


def test_prompt_builder_context_truncation(monkeypatch):
    """Verify prompt builder halts adding passages when context budget is exceeded."""
    service = PromptService()
    # Force a small budget
    monkeypatch.setattr(service.settings, "MAX_CONTEXT_CHARS", 150)

    passages = [
        RetrievedPassage(
            chunk_id=f"c{i}",
            doc_id="d1",
            filename="doc.pdf",
            page_numbers=[i],
            text="This is a moderately long passage of text for context budget testing." * 2,
            distance=0.1,
            similarity_score=0.9,
        )
        for i in range(5)
    ]

    _, user_prompt, sources = service.build_prompt("Budget check?", passages)
    # Should contain only 1 source due to budget cutoff
    assert len(sources) == 1
    assert len(user_prompt) <= 1000
