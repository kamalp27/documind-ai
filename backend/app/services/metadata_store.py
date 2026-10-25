"""Local JSON metadata storage for persisted document tracking."""

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional
from app.config import get_settings
from app.core.exceptions import DocumentNotFoundException, StorageException
from app.schemas.document import DocumentMetadata

logger = logging.getLogger("documind.metadata_store")


class DocumentMetadataStore:
    """Thread-safe persistent store for document metadata records."""

    def __init__(self, file_path: Optional[Path] = None):
        settings = get_settings()
        self.file_path = file_path or settings.metadata_file_path
        self._lock = Lock()
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """Ensure parent directory and file exist."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_all(self) -> Dict[str, dict]:
        """Load all metadata records from disk."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_all(self, data: Dict[str, dict]) -> None:
        """Atomically persist records to disk."""
        temp_file = self.file_path.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.file_path)
        except Exception as exc:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            logger.error("Failed to save metadata to '%s': %s", self.file_path, exc, exc_info=True)
            raise StorageException("Failed to persist document metadata.", details={"error": str(exc)})

    def save_document(self, metadata: DocumentMetadata) -> None:
        """Save or update a document metadata entry."""
        with self._lock:
            data = self._load_all()
            data[metadata.document_id] = metadata.model_dump()
            self._save_all(data)
            logger.info("Persisted metadata for document '%s'", metadata.document_id)

    def get_document(self, doc_id: str) -> DocumentMetadata:
        """Retrieve metadata for a specific document by ID.

        Raises:
            DocumentNotFoundException: If the document ID is not found.
        """
        with self._lock:
            data = self._load_all()
            if doc_id not in data:
                raise DocumentNotFoundException(doc_id)
            return DocumentMetadata(**data[doc_id])

    def list_documents(self) -> List[DocumentMetadata]:
        """List all tracked documents ordered by creation date descending."""
        with self._lock:
            data = self._load_all()
            docs = [DocumentMetadata(**item) for item in data.values()]
            docs.sort(key=lambda d: d.created_at, reverse=True)
            return docs

    def delete_document(self, doc_id: str) -> None:
        """Remove a document's metadata entry.

        Raises:
            DocumentNotFoundException: If the document ID does not exist.
        """
        with self._lock:
            data = self._load_all()
            if doc_id not in data:
                raise DocumentNotFoundException(doc_id)
            del data[doc_id]
            self._save_all(data)
            logger.info("Deleted metadata record for document '%s'", doc_id)


_metadata_store_instance: Optional[DocumentMetadataStore] = None


def get_metadata_store() -> DocumentMetadataStore:
    """Retrieve shared metadata store instance."""
    global _metadata_store_instance
    if _metadata_store_instance is None:
        _metadata_store_instance = DocumentMetadataStore()
    return _metadata_store_instance
