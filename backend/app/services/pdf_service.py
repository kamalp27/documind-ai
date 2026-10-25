"""PDF validation and text extraction service using PyMuPDF."""

import os
import re
from typing import Any, Dict, List
import pymupdf as fitz

from app.core.exceptions import (
    DocumentProcessingException,
    FileValidationException,
    ScannedDocumentException,
)

PDF_MAGIC_BYTES = b"%PDF-"


def sanitize_filename(filename: str) -> str:
    """Sanitize user-provided filename to prevent directory traversal and invalid characters."""
    if not filename:
        return "document.pdf"

    # Remove directory paths if provided in client filename
    base_name = os.path.basename(filename.replace("\\", "/"))
    # Remove potentially dangerous characters, keep alphanumeric, dots, hyphens, underscores
    cleaned = re.sub(r"[^\w\s\.-]", "", base_name).strip()
    # Normalize spaces
    cleaned = re.sub(r"\s+", " ", cleaned)

    if not cleaned or cleaned.startswith("."):
        return "document.pdf"

    if "." not in cleaned:
        cleaned += ".pdf"

    return cleaned


class PDFService:
    """Handles PDF file validation and page-by-page text extraction."""

    def __init__(self, max_size_bytes: int = 10 * 1024 * 1024):
        self.max_size_bytes = max_size_bytes

    def validate_file(self, content: bytes, filename: str) -> str:
        """Validate file size, extension, and PDF header signature.

        Returns:
            Sanitized safe display filename.
        """
        # 1. Check non-empty
        if not content or len(content) == 0:
            raise FileValidationException("Uploaded file is empty (0 bytes).", status_code=400)

        # 2. Check maximum size
        if len(content) > self.max_size_bytes:
            max_mb = self.max_size_bytes // (1024 * 1024)
            raise FileValidationException(
                f"File size exceeds the maximum limit of {max_mb} MB.",
                status_code=413,
                details={"max_size_bytes": self.max_size_bytes, "actual_size_bytes": len(content)},
            )

        # 3. Check filename extension
        safe_name = sanitize_filename(filename)
        if not safe_name.lower().endswith(".pdf"):
            raise FileValidationException("Unsupported file type. Only PDF files are accepted.", status_code=400)

        # 4. Check PDF magic bytes signature
        if not content.startswith(PDF_MAGIC_BYTES):
            raise FileValidationException(
                "Invalid PDF file. The file content does not start with a valid PDF signature.",
                status_code=400,
            )

        return safe_name

    def extract_pages(self, content: bytes) -> List[Dict[str, Any]]:
        """Extract text from PDF page by page while preserving 1-indexed page numbers.

        Raises:
            DocumentProcessingException: If the PDF is encrypted, corrupted, or has no pages.
            ScannedDocumentException: If the PDF contains no extractable text.
        """
        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise DocumentProcessingException(
                "Failed to parse PDF document. The file may be corrupt or malformed.",
                details={"error": str(exc)},
            )

        try:
            if doc.is_encrypted:
                raise DocumentProcessingException(
                    "Encrypted or password-protected PDFs are not supported.",
                    details={"is_encrypted": True},
                )

            total_pages = doc.page_count
            if total_pages == 0:
                raise DocumentProcessingException("PDF document contains no pages.")

            extracted_pages: List[Dict[str, Any]] = []
            total_extracted_chars = 0

            for page_idx in range(total_pages):
                page = doc.load_page(page_idx)
                text = page.get_text("text").strip()
                page_num = page_idx + 1  # 1-indexed page number

                extracted_pages.append(
                    {
                        "page_number": page_num,
                        "text": text,
                        "char_count": len(text),
                    }
                )
                total_extracted_chars += len(text)

            # Check if any extractable text was found
            if total_extracted_chars == 0:
                raise ScannedDocumentException(
                    "This document contains no extractable text. Scanned PDFs require OCR, which is not supported in this phase."
                )

            return extracted_pages

        finally:
            doc.close()
