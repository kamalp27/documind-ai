"""Integration tests for RAG Chat Query API (/api/v1/chat/query)."""

import io
import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import GeminiServiceException
from app.main import app
from app.services.document_service import DocumentService, get_document_service
from app.services.gemini_service import GeminiService
from app.services.rag_service import RAGService, get_rag_service
from app.services.retrieval_service import RetrievalService




def test_chat_query_empty_question(client_with_mock_rag: TestClient):
    """Verify empty question fails Pydantic validation with 422."""
    response = client_with_mock_rag.post("/api/v1/chat/query", json={"question": ""})
    assert response.status_code == 422


def test_chat_query_whitespace_question(client_with_mock_rag: TestClient):
    """Verify whitespace-only question fails validation with 422."""
    response = client_with_mock_rag.post("/api/v1/chat/query", json={"question": "     \n\t  "})
    assert response.status_code == 422


def test_chat_query_too_long_question(client_with_mock_rag: TestClient):
    """Verify questions exceeding 1000 characters fail validation with 422."""
    response = client_with_mock_rag.post("/api/v1/chat/query", json={"question": "a" * 1001})
    assert response.status_code == 422


def test_chat_query_no_documents_returns_insufficient_evidence(client_with_mock_rag: TestClient):
    """Verify querying an empty system immediately returns insufficient evidence without error."""
    response = client_with_mock_rag.post(
        "/api/v1/chat/query",
        json={"question": "What is the policy?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["insufficient_evidence"] is True
    assert "not contain sufficient information" in data["answer"].lower()
    assert data["sources"] == []


def test_chat_query_with_indexed_document_success(
    client_with_mock_rag: TestClient,
    sample_pdf_bytes: bytes,
):
    """Verify full RAG pipeline: upload PDF, query content, receive answer with structured citations."""
    # 1. Upload a document
    files = {"file": ("architecture_doc.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    upload_res = client_with_mock_rag.post("/api/v1/documents/upload", files=files)
    assert upload_res.status_code == 201

    # 2. Query document content
    query_payload = {"question": "How does DocuMind AI handle citations?"}
    query_res = client_with_mock_rag.post("/api/v1/chat/query", json=query_payload)
    assert query_res.status_code == 200

    data = query_res.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert data["insufficient_evidence"] is False
    assert len(data["sources"]) >= 1

    # Check citation structure
    citation = data["sources"][0]
    assert citation["source_id"] == 1
    assert citation["filename"] == "architecture_doc.pdf"
    assert isinstance(citation["page_numbers"], list)
    assert len(citation["page_numbers"]) >= 1
    assert len(citation["excerpt"]) > 0


def test_chat_query_filtered_by_document_id(
    client_with_mock_rag: TestClient,
    sample_pdf_bytes: bytes,
):
    """Verify querying with document_ids filter restricts citations to that document."""
    files = {"file": ("specific_doc.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    upload_res = client_with_mock_rag.post("/api/v1/documents/upload", files=files)
    doc_id = upload_res.json()["document_id"]

    query_payload = {
        "question": "What is discussed on page one?",
        "document_ids": [doc_id],
    }
    response = client_with_mock_rag.post("/api/v1/chat/query", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    for source in data["sources"]:
        assert source["document_id"] == doc_id


def test_chat_query_invalid_document_id_returns_404(client_with_mock_rag: TestClient):
    """Verify supplying a non-existent document ID in document_ids returns 404."""
    response = client_with_mock_rag.post(
        "/api/v1/chat/query",
        json={"question": "Test question", "document_ids": ["00000000-0000-0000-0000-000000000000"]},
    )
    assert response.status_code == 404
    data = response.json()
    assert data["success"] is False
    assert "not found" in data["error"].lower()


def test_chat_query_missing_api_key_error(
    test_document_service: DocumentService,
    test_retrieval_service: RetrievalService,
    sample_pdf_bytes: bytes,
):
    """Verify missing GEMINI_API_KEY returns a clear 500 configuration error when generation is needed."""
    # Index a document so passages are found
    test_document_service.process_and_index_pdf(sample_pdf_bytes, "doc.pdf")

    # Create RAGService with unconfigured GeminiService
    unconfigured_gemini = GeminiService(api_key="")
    rag_service = RAGService(
        retrieval_service=test_retrieval_service,
        gemini_service=unconfigured_gemini,
    )

    app.dependency_overrides[get_rag_service] = lambda: rag_service
    client = TestClient(app)

    response = client.post("/api/v1/chat/query", json={"question": "What is the architecture?"})
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert "GEMINI_API_KEY" in data["error"]

    app.dependency_overrides.clear()


def test_chat_query_rate_limit_handling(
    test_document_service: DocumentService,
    test_retrieval_service: RetrievalService,
    sample_pdf_bytes: bytes,
):
    """Verify Gemini 429 quota exhaustion propagates as a clean 429 status code."""
    test_document_service.process_and_index_pdf(sample_pdf_bytes, "doc.pdf")

    class RateLimitedGemini(GeminiService):
        def generate_answer(self, system_instruction, user_content):
            raise GeminiServiceException("Gemini API quota exceeded.", status_code=429)

    rag_service = RAGService(
        retrieval_service=test_retrieval_service,
        gemini_service=RateLimitedGemini(api_key="key"),
    )

    app.dependency_overrides[get_rag_service] = lambda: rag_service
    client = TestClient(app)

    response = client.post("/api/v1/chat/query", json={"question": "What is the architecture?"})
    assert response.status_code == 429
    data = response.json()
    assert data["success"] is False
    assert "quota" in data["error"].lower() or "rate limit" in data["error"].lower()

    app.dependency_overrides.clear()


def test_chat_query_response_payload_safety(
    client_with_mock_rag: TestClient,
    sample_pdf_bytes: bytes,
):
    """Ensure no system file paths, stack traces, or environment variables leak in query responses."""
    files = {"file": ("safety_check.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    client_with_mock_rag.post("/api/v1/documents/upload", files=files)

    response = client_with_mock_rag.post(
        "/api/v1/chat/query",
        json={"question": "Explain system safety."},
    )
    assert response.status_code == 200
    data = response.json()

    # Verify no private paths or secrets in response
    response_str = str(data)
    assert "C:\\" not in response_str
    assert "/home/" not in response_str
    assert "mock-api-key" not in response_str
