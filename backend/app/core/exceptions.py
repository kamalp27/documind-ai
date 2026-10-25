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


class DocumentProcessingException(DocuMindException):
    """Raised when document extraction or chunking encounters an error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=422, details=details)


class DocumentNotFoundException(DocuMindException):
    """Raised when a requested document is not found."""

    def __init__(self, doc_id: str):
        super().__init__(
            message=f"Document with ID '{doc_id}' not found.",
            status_code=404,
            details={"document_id": doc_id},
        )


class RAGServiceException(DocuMindException):
    """Raised when retrieval or generation operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=502, details=details)
