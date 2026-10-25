"""Document ingestion and lifecycle orchestration service."""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from app.config import get_settings
from app.core.exceptions import (
    DocuMindException,
    DocumentNotFoundException,
    FileValidationException,
    ScannedDocumentException,
)
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
)
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.metadata_store import DocumentMetadataStore, get_metadata_store
from app.services.pdf_service import PDFService
from app.services.vector_store import ChromaVectorStore, get_vector_store

logger = logging.getLogger("documind.document_service")


class DocumentService:
    """Orchestrates PDF validation, text extraction, chunking, embedding, and vector indexing."""

    def __init__(
        self,
        pdf_service: Optional[PDFService] = None,
        chunking_service: Optional[ChunkingService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        metadata_store: Optional[DocumentMetadataStore] = None,
    ):
        settings = get_settings()
        self.pdf_service = pdf_service or PDFService(max_size_bytes=settings.MAX_UPLOAD_SIZE_BYTES)
        self.chunking_service = chunking_service or ChunkingService(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.metadata_store = metadata_store or get_metadata_store()

    def process_and_index_pdf(self, file_content: bytes, original_filename: str) -> DocumentUploadResponse:
        """Execute end-to-end PDF ingestion pipeline.

        Args:
            file_content: Raw bytes of the uploaded PDF.
            original_filename: Client-supplied filename.

        Returns:
            DocumentUploadResponse with ingestion statistics.
        """
        # Step 1: Validate file signature, extension, and size
        safe_filename = self.pdf_service.validate_file(file_content, original_filename)

        # Step 2: Generate unique document identifier
        doc_id = str(uuid.uuid4())
        logger.info("Processing document '%s' (ID: %s)", safe_filename, doc_id)

        try:
            # Step 3: Extract text per page
            extracted_pages = self.pdf_service.extract_pages(file_content)
            total_pages = len(extracted_pages)

            # Step 4: Chunk text preserving page numbers
            chunks = self.chunking_service.create_chunks(
                pages=extracted_pages,
                doc_id=doc_id,
                filename=safe_filename,
            )

            if not chunks:
                raise ScannedDocumentException(
                    "No valid text chunks could be generated from the document."
                )

            # Step 5: Generate vector embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(chunk_texts)

            # Step 6: Upsert into persistent vector store
            self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

            # Step 7: Persist document metadata
            now_iso = datetime.now(timezone.utc).isoformat()
            metadata = DocumentMetadata(
                document_id=doc_id,
                filename=safe_filename,
                file_size_bytes=len(file_content),
                total_pages=total_pages,
                total_chunks=len(chunks),
                created_at=now_iso,
            )
            self.metadata_store.save_document(metadata)

            return DocumentUploadResponse(
                document_id=doc_id,
                filename=safe_filename,
                status="processed",
                total_pages=total_pages,
                total_chunks=len(chunks),
                created_at=now_iso,
            )

        except Exception as exc:
            # Clean up any partial vector store entries on failure
            logger.warning("Ingestion failed for doc_id '%s'. Rolling back state.", doc_id)
            try:
                self.vector_store.delete_document_chunks(doc_id)
            except Exception:
                pass
            raise

    def get_document(self, doc_id: str) -> DocumentMetadata:
        """Get metadata for a single document."""
        return self.metadata_store.get_document(doc_id)

    def list_documents(self) -> DocumentListResponse:
        """List all indexed documents."""
        documents = self.metadata_store.list_documents()
        return DocumentListResponse(documents=documents, total_count=len(documents))

    def delete_document(self, doc_id: str) -> DocumentDeleteResponse:
        """Purge document metadata and vectors from the system.

        Raises:
            DocumentNotFoundException: If doc_id does not exist.
        """
        # Verify existence in metadata store (will raise DocumentNotFoundException if missing)
        self.metadata_store.get_document(doc_id)

        # Remove from vector store
        self.vector_store.delete_document_chunks(doc_id)

        # Remove from metadata store
        self.metadata_store.delete_document(doc_id)

        return DocumentDeleteResponse(
            success=True,
            message="Document and associated vector chunks deleted successfully.",
            document_id=doc_id,
        )


_document_service_instance: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """Retrieve shared document service instance."""
    global _document_service_instance
    if _document_service_instance is None:
        _document_service_instance = DocumentService()
    return _document_service_instance
