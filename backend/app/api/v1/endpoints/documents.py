"""API router endpoints for PDF document ingestion, inspection, and deletion."""

import logging
from fastapi import APIRouter, Depends, File, UploadFile, status

from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
)
from app.services.document_service import DocumentService, get_document_service

logger = logging.getLogger("documind.api.documents")

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Index PDF Document",
    description="Validates a PDF, extracts text page-by-page, generates vector embeddings, and indexes them in ChromaDB.",
)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """Accept and process a single PDF file upload."""
    content = await file.read()
    filename = file.filename or "document.pdf"
    return document_service.process_and_index_pdf(
        file_content=content,
        original_filename=filename,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List All Indexed Documents",
    description="Returns metadata for all documents currently indexed in the vector database.",
)
async def list_documents(
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List all indexed documents."""
    return document_service.list_documents()


@router.get(
    "/{document_id}",
    response_model=DocumentMetadata,
    status_code=status.HTTP_200_OK,
    summary="Get Document Details",
    description="Retrieve metadata for a specific document by its ID.",
)
async def get_document_details(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentMetadata:
    """Get metadata for a specific document ID."""
    return document_service.get_document(document_id)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Document",
    description="Purges document metadata and all associated vector embeddings from ChromaDB.",
)
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentDeleteResponse:
    """Delete a document and all its chunks from the vector store."""
    return document_service.delete_document(document_id)
