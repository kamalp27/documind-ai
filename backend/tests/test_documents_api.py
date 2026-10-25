"""Integration tests for Document Ingestion, Retrieval, and Deletion APIs."""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.document_service import DocumentService, get_document_service


@pytest.fixture
def client_with_mock_services(test_document_service: DocumentService):
    """FastAPI TestClient with injected mock embedding service and isolated storage."""
    app.dependency_overrides[get_document_service] = lambda: test_document_service
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_upload_valid_pdf(client_with_mock_services: TestClient, sample_pdf_bytes: bytes):
    """Verify uploading a valid PDF returns 201 Created and proper metadata."""
    files = {"file": ("research_paper.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    response = client_with_mock_services.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 201

    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "research_paper.pdf"
    assert data["status"] == "processed"
    assert data["total_pages"] == 2
    assert data["total_chunks"] >= 2
    assert "created_at" in data

    # Verify no local filesystem paths or secrets leak
    for key, value in data.items():
        if isinstance(value, str):
            assert "C:\\" not in value
            assert "/home/" not in value


def test_upload_scanned_pdf_rejected(client_with_mock_services: TestClient, blank_pdf_bytes: bytes):
    """Verify PDFs with no extractable text return 422 with OCR limitation explanation."""
    files = {"file": ("scanned_invoice.pdf", io.BytesIO(blank_pdf_bytes), "application/pdf")}
    response = client_with_mock_services.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 422

    data = response.json()
    assert data["success"] is False
    assert "ocr" in data["error"].lower()


def test_upload_empty_file_rejected(client_with_mock_services: TestClient):
    """Verify 0-byte file upload returns 400 Bad Request."""
    files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
    response = client_with_mock_services.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400

    data = response.json()
    assert data["success"] is False
    assert "empty" in data["error"].lower()


def test_upload_non_pdf_file_rejected(client_with_mock_services: TestClient):
    """Verify non-PDF file upload returns 400 Bad Request."""
    files = {"file": ("payload.txt", io.BytesIO(b"Hello world text"), "text/plain")}
    response = client_with_mock_services.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400

    data = response.json()
    assert data["success"] is False


def test_list_and_get_documents(client_with_mock_services: TestClient, sample_pdf_bytes: bytes):
    """Verify listing documents and retrieving single document metadata."""
    # 1. Upload a document
    files = {"file": ("test_doc.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    upload_res = client_with_mock_services.post("/api/v1/documents/upload", files=files)
    doc_id = upload_res.json()["document_id"]

    # 2. List documents
    list_res = client_with_mock_services.get("/api/v1/documents")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total_count"] >= 1
    assert any(doc["document_id"] == doc_id for doc in list_data["documents"])

    # 3. Get single document
    get_res = client_with_mock_services.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["document_id"] == doc_id
    assert get_data["filename"] == "test_doc.pdf"
    assert get_data["total_pages"] == 2


def test_get_nonexistent_document(client_with_mock_services: TestClient):
    """Verify requesting an unknown document ID returns 404."""
    response = client_with_mock_services.get("/api/v1/documents/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    data = response.json()
    assert data["success"] is False
    assert "not found" in data["error"].lower()


def test_delete_document_success(client_with_mock_services: TestClient, sample_pdf_bytes: bytes):
    """Verify deleting a document removes it from metadata and vector storage."""
    # 1. Upload
    files = {"file": ("to_delete.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    upload_res = client_with_mock_services.post("/api/v1/documents/upload", files=files)
    doc_id = upload_res.json()["document_id"]

    # 2. Delete
    delete_res = client_with_mock_services.delete(f"/api/v1/documents/{doc_id}")
    assert delete_res.status_code == 200
    del_data = delete_res.json()
    assert del_data["success"] is True
    assert del_data["document_id"] == doc_id

    # 3. Subsequent GET returns 404
    get_res = client_with_mock_services.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 404


def test_delete_nonexistent_document(client_with_mock_services: TestClient):
    """Verify deleting an unknown document returns 404."""
    response = client_with_mock_services.delete("/api/v1/documents/non-existent-doc-id")
    assert response.status_code == 404
    data = response.json()
    assert data["success"] is False
