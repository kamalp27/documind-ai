"""Custom application exceptions for DocuMind AI."""

from typing import Any, Dict, Optional


class DocuMindException(Exception):
    """Base exception class for DocuMind AI domain errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class FileValidationException(DocuMindException):
    """Raised when an uploaded file fails validation (size, extension, signature)."""

    def __init__(self, message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=status_code, details=details)


class QueryValidationException(DocuMindException):
    """Raised when a user chat query fails validation."""

    def __init__(self, message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=status_code, details=details)


class DocumentProcessingException(DocuMindException):
    """Raised when document extraction or chunking encounters an unrecoverable error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=422, details=details)


class ScannedDocumentException(DocuMindException):
    """Raised when a PDF contains no extractable text (e.g. scanned image)."""

    def __init__(
        self,
        message: str = "This document appears to be a scanned image with no extractable text. OCR is not supported in this phase.",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, status_code=422, details=details)


class DocumentNotFoundException(DocuMindException):
    """Raised when a requested document is not found."""

    def __init__(self, doc_id: str):
        super().__init__(
            message=f"Document with ID '{doc_id}' not found.",
            status_code=404,
            details={"document_id": doc_id},
        )


class StorageException(DocuMindException):
    """Raised when persistent storage or vector store encounters an error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=500, details=details)


class MissingAPIKeyException(DocuMindException):
    """Raised when Gemini API key is missing or blank."""

    def __init__(
        self,
        message: str = "Gemini API key is not configured. Please set GEMINI_API_KEY in your backend/.env file.",
    ):
        super().__init__(message=message, status_code=500, details={"configuration": "GEMINI_API_KEY"})


class GeminiServiceException(DocuMindException):
    """Raised when Gemini model inference fails, encounters rate limiting, or returns blocked content."""

    def __init__(self, message: str, status_code: int = 502, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=status_code, details=details)


class RAGServiceException(DocuMindException):
    """Raised when retrieval or generation operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=502, details=details)
