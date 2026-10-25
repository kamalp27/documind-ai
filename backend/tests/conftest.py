"""Pytest shared fixtures for DocuMind AI backend tests."""

import io
from pathlib import Path
from typing import Generator
import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeminiService
from app.services.metadata_store import DocumentMetadataStore
from app.services.pdf_service import PDFService
from app.services.prompt_service import PromptService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import ChromaVectorStore


class MockEmbeddingService(EmbeddingService):
    """Mock embedding service that produces deterministic 384-dimensional vectors without downloading models."""

    def __init__(self):
        super().__init__(model_name="mock-model")

    def embed_texts(self, texts):
        if not texts:
            return []
        # Return deterministic synthetic 384-dim float vectors with positive non-zero norm
        return [[(0.05 * (i % 10) + 0.1) for _ in range(384)] for i, _ in enumerate(texts)]


class MockGeminiService(GeminiService):
    """Mock Gemini service that simulates LLM generation without network calls or API keys."""

    def __init__(self, response_text: str = "According to [1], DocuMind AI uses ChromaDB for semantic search."):
        super().__init__(api_key="mock-api-key", model_name="gemini-mock")
        self.response_text = response_text
        self.last_system_instruction = None
        self.last_user_content = None

    def generate_answer(self, system_instruction: str, user_content: str) -> str:
        self.last_system_instruction = system_instruction
        self.last_user_content = user_content
        return self.response_text


@pytest.fixture
def mock_embedding_service() -> MockEmbeddingService:
    """Fixture providing a mock embedding service for fast, offline unit testing."""
    return MockEmbeddingService()


@pytest.fixture
def mock_gemini_service() -> MockGeminiService:
    """Fixture providing a mock Gemini service returning grounded answers."""
    return MockGeminiService()


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Fixture providing an isolated temporary directory for test persistence."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def test_metadata_store(temp_data_dir: Path) -> DocumentMetadataStore:
    """Fixture providing a fresh metadata store bound to temp_data_dir."""
    meta_file = temp_data_dir / "documents_metadata.json"
    return DocumentMetadataStore(file_path=meta_file)


@pytest.fixture
def test_vector_store(temp_data_dir: Path) -> ChromaVectorStore:
    """Fixture providing a fresh ChromaDB vector store bound to temp_data_dir."""
    chroma_dir = temp_data_dir / "chroma"
    return ChromaVectorStore(persist_dir=chroma_dir, collection_name="test_collection")


@pytest.fixture
def test_document_service(
    mock_embedding_service: MockEmbeddingService,
    test_vector_store: ChromaVectorStore,
    test_metadata_store: DocumentMetadataStore,
) -> DocumentService:
    """Fixture providing an end-to-end DocumentService with isolated storage and mock embeddings."""
    return DocumentService(
        pdf_service=PDFService(max_size_bytes=10 * 1024 * 1024),
        chunking_service=ChunkingService(chunk_size=400, chunk_overlap=50),
        embedding_service=mock_embedding_service,
        vector_store=test_vector_store,
        metadata_store=test_metadata_store,
    )


@pytest.fixture
def test_retrieval_service(
    mock_embedding_service: MockEmbeddingService,
    test_vector_store: ChromaVectorStore,
    test_metadata_store: DocumentMetadataStore,
) -> RetrievalService:
    """Fixture providing a RetrievalService bound to test storage."""
    return RetrievalService(
        embedding_service=mock_embedding_service,
        vector_store=test_vector_store,
        metadata_store=test_metadata_store,
    )


@pytest.fixture
def test_rag_service(
    test_retrieval_service: RetrievalService,
    mock_gemini_service: MockGeminiService,
) -> RAGService:
    """Fixture providing an end-to-end RAGService with mocked generation."""
    return RAGService(
        retrieval_service=test_retrieval_service,
        prompt_service=PromptService(),
        gemini_service=mock_gemini_service,
    )


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a clean 2-page PDF in-memory with known text using PyMuPDF."""
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 100),
        "DocuMind AI Architecture Overview\n"
        "Page one discusses retrieval-augmented generation and neural semantic search.\n"
        "Vector databases like ChromaDB store high-dimensional embeddings for fast similarity indexing.\n"
        "This is essential for modern Generative AI engineering workflows.",
    )

    # Page 2
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (50, 100),
        "DocuMind AI Page Two Citations\n"
        "Every chunk retains source metadata such as document ID, filename, and page number.\n"
        "When generating answers, the Large Language Model cites source page numbers directly.\n"
        "This guarantees strict traceability and eliminates hallucinations.",
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def blank_pdf_bytes() -> bytes:
    """Generate a 1-page PDF in-memory with NO text (simulating a scanned document)."""
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def client_with_mock_rag(
    test_document_service: DocumentService,
    test_rag_service: RAGService,
) -> Generator[TestClient, None, None]:
    """TestClient configured with isolated storage, mock embeddings, and mock Gemini generation."""
    from app.services.document_service import get_document_service
    from app.services.rag_service import get_rag_service

    app.dependency_overrides[get_document_service] = lambda: test_document_service
    app.dependency_overrides[get_rag_service] = lambda: test_rag_service
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
