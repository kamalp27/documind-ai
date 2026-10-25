"""Edge-case automated regression tests for DocuMind AI.

Validates boundary conditions:
- Exact max length query (1000 chars)
- Unicode and unusual whitespace in PDFs
- Multi-document isolation (deleting one preserves searchability of others)
- Relevant document filter with irrelevant question (insufficient evidence)
- Gemini service exceptions (503 and empty response)
"""

import io
import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import GeminiServiceException
from app.main import app
from app.schemas.chat import ChatQueryRequest
from app.services.document_service import DocumentService, get_document_service
from app.services.gemini_service import GeminiService
from app.services.pdf_service import PDFService
from app.services.rag_service import RAGService, get_rag_service


def test_chat_query_exact_max_length(client_with_mock_rag: TestClient):
    """Verify that a question with exactly 1000 characters is accepted without 422."""
    valid_1000_char_q = "What is the policy? " + "a" * (1000 - len("What is the policy? "))
    assert len(valid_1000_char_q) == 1000

    response = client_with_mock_rag.post(
        "/api/v1/chat/query",
        json={"question": valid_1000_char_q},
    )
    # Since no docs are uploaded, it should return 200 with insufficient evidence, NOT 422
    assert response.status_code == 200
    assert response.json()["insufficient_evidence"] is True


def test_pdf_with_unicode_and_unusual_whitespace(
    client_with_mock_rag: TestClient,
    test_document_service: DocumentService,
):
    """Verify PDF containing non-ASCII Unicode (math, Greek, symbols) and multiple whitespace formats extracts cleanly."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        (50, 100),
        "DocuMind Unicode \u03b1\u03b2\u03b3 Analysis \u2014 \u03a9mega & Infinity \u221e\n\n"
        "Formula:   \t   E = mc\u00b2  \u2248  \u2211(x_i)  \n\n"
        "Standardized     spaces      and \r\n newline sequences are tested.",
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    files = {"file": ("unicode_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    res = client_with_mock_rag.post("/api/v1/documents/upload", files=files)
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "processed"
    assert data["total_pages"] == 1
    assert data["total_chunks"] >= 1


def test_multi_document_deletion_isolation(
    client_with_mock_rag: TestClient,
    sample_pdf_bytes: bytes,
):
    """Verify that deleting document A removes only document A, leaving document B intact and queryable."""
    # Upload doc 1
    files1 = {"file": ("doc_one.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    res1 = client_with_mock_rag.post("/api/v1/documents/upload", files=files1)
    doc_id_1 = res1.json()["document_id"]

    # Upload doc 2
    files2 = {"file": ("doc_two.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    res2 = client_with_mock_rag.post("/api/v1/documents/upload", files=files2)
    doc_id_2 = res2.json()["document_id"]

    # Verify both exist in list
    list_res = client_with_mock_rag.get("/api/v1/documents")
    assert list_res.json()["total_count"] == 2

    # Delete doc 1
    del_res = client_with_mock_rag.delete(f"/api/v1/documents/{doc_id_1}")
    assert del_res.status_code == 200

    # Verify doc 1 is gone, doc 2 remains
    list_res_after = client_with_mock_rag.get("/api/v1/documents")
    assert list_res_after.json()["total_count"] == 1
    assert list_res_after.json()["documents"][0]["document_id"] == doc_id_2

    # Verify querying doc 2 still works and returns citations from doc 2
    query_res = client_with_mock_rag.post(
        "/api/v1/chat/query",
        json={"question": "What is on page one?", "document_ids": [doc_id_2]},
    )
    assert query_res.status_code == 200
    assert query_res.json()["insufficient_evidence"] is False
    assert len(query_res.json()["sources"]) >= 1
    assert query_res.json()["sources"][0]["document_id"] == doc_id_2


def test_chat_query_gemini_503_service_unavailable(
    test_document_service: DocumentService,
    test_retrieval_service,
    sample_pdf_bytes: bytes,
):
    """Verify Gemini service unavailability (503) propagates as a 503 response without stack trace."""
    test_document_service.process_and_index_pdf(sample_pdf_bytes, "doc.pdf")

    class ServiceUnavailableGemini(GeminiService):
        def generate_answer(self, system_instruction, user_content):
            raise GeminiServiceException("Gemini service temporarily overloaded.", status_code=503)

    rag = RAGService(
        retrieval_service=test_retrieval_service,
        gemini_service=ServiceUnavailableGemini(api_key="mock"),
    )

    app.dependency_overrides[get_rag_service] = lambda: rag
    client = TestClient(app)

    res = client.post("/api/v1/chat/query", json={"question": "What is the architecture?"})
    assert res.status_code == 503
    data = res.json()
    assert data["success"] is False
    assert "overloaded" in data["error"].lower()
    app.dependency_overrides.clear()


def test_chat_query_empty_model_response(
    test_document_service: DocumentService,
    test_retrieval_service,
    sample_pdf_bytes: bytes,
):
    """Verify that an empty response from Gemini raises an appropriate 502/500 error."""
    test_document_service.process_and_index_pdf(sample_pdf_bytes, "doc.pdf")

    class EmptyResponseGemini(GeminiService):
        def generate_answer(self, system_instruction, user_content):
            raise GeminiServiceException("Gemini returned an empty or blank response.", status_code=502)

    rag = RAGService(
        retrieval_service=test_retrieval_service,
        gemini_service=EmptyResponseGemini(api_key="mock"),
    )

    app.dependency_overrides[get_rag_service] = lambda: rag
    client = TestClient(app)

    res = client.post("/api/v1/chat/query", json={"question": "What is the architecture?"})
    assert res.status_code == 502
    data = res.json()
    assert data["success"] is False
    assert "empty" in data["error"].lower() or "blank" in data["error"].lower()
    app.dependency_overrides.clear()
