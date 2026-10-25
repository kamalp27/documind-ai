"""Unit tests for PDFService validation and text extraction."""

import pytest
from app.core.exceptions import (
    DocumentProcessingException,
    FileValidationException,
    ScannedDocumentException,
)
from app.services.pdf_service import PDFService, sanitize_filename


def test_sanitize_filename():
    """Verify filename sanitization prevents path traversal and enforces safety."""
    assert sanitize_filename("my_document.pdf") == "my_document.pdf"
    assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename(r"..\..\Windows\system32\cmd.pdf") == "cmd.pdf"
    assert sanitize_filename("test document name.pdf") == "test document name.pdf"
    assert sanitize_filename("no_extension") == "no_extension.pdf"
    assert sanitize_filename("") == "document.pdf"
    assert sanitize_filename("...") == "document.pdf"


def test_validate_file_valid_pdf(sample_pdf_bytes: bytes):
    """Verify that a valid PDF passes validation."""
    service = PDFService(max_size_bytes=10 * 1024 * 1024)
    filename = service.validate_file(sample_pdf_bytes, "architecture_notes.pdf")
    assert filename == "architecture_notes.pdf"


def test_validate_file_empty_bytes():
    """Verify that 0-byte uploads are rejected with status 400."""
    service = PDFService(max_size_bytes=10 * 1024 * 1024)
    with pytest.raises(FileValidationException) as exc_info:
        service.validate_file(b"", "empty.pdf")
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.message.lower()


def test_validate_file_exceeds_max_size(sample_pdf_bytes: bytes):
    """Verify that files exceeding the configured size limit are rejected with 413."""
    tiny_limit = len(sample_pdf_bytes) - 10
    service = PDFService(max_size_bytes=tiny_limit)
    with pytest.raises(FileValidationException) as exc_info:
        service.validate_file(sample_pdf_bytes, "large.pdf")
    assert exc_info.value.status_code == 413
    assert "exceeds" in exc_info.value.message.lower()


def test_validate_file_invalid_signature():
    """Verify that non-PDF files masquerading as PDFs are rejected."""
    service = PDFService(max_size_bytes=10 * 1024 * 1024)
    fake_pdf = b"Plain text file content without PDF magic header"
    with pytest.raises(FileValidationException) as exc_info:
        service.validate_file(fake_pdf, "fake.pdf")
    assert exc_info.value.status_code == 400
    assert "signature" in exc_info.value.message.lower()


def test_validate_file_invalid_extension():
    """Verify that non-PDF extensions are rejected."""
    service = PDFService(max_size_bytes=10 * 1024 * 1024)
    with pytest.raises(FileValidationException) as exc_info:
        service.validate_file(b"%PDF-1.4 dummy", "script.py")
    # sanitize_filename will append .pdf if omitted, but if filename is already sanitized without .pdf:
    # Let's ensure sanitize_filename works or rejection occurs


def test_extract_pages_success(sample_pdf_bytes: bytes):
    """Verify extracting pages preserves text and 1-indexed page numbering."""
    service = PDFService()
    pages = service.extract_pages(sample_pdf_bytes)
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "DocuMind AI Architecture" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "DocuMind AI Page Two Citations" in pages[1]["text"]


def test_extract_pages_scanned_pdf(blank_pdf_bytes: bytes):
    """Verify that PDFs with no extractable text raise ScannedDocumentException."""
    service = PDFService()
    with pytest.raises(ScannedDocumentException) as exc_info:
        service.extract_pages(blank_pdf_bytes)
    assert exc_info.value.status_code == 422
    assert "scanned" in exc_info.value.message.lower()
    assert "ocr" in exc_info.value.message.lower()


def test_extract_pages_corrupted_pdf():
    """Verify corrupted PDF content raises DocumentProcessingException."""
    service = PDFService()
    corrupt_bytes = b"%PDF-1.4\nCorrupt header and broken body EOF"
    with pytest.raises(DocumentProcessingException) as exc_info:
        service.extract_pages(corrupt_bytes)
    assert exc_info.value.status_code == 422
