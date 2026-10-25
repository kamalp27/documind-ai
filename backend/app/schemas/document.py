"""Pydantic schemas for document ingestion, metadata, and responses."""

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """Represents a discrete text chunk extracted from a document page."""

    chunk_id: str = Field(description="Unique identifier for the chunk, e.g. docid_p1_c0")
    doc_id: str = Field(description="ID of the parent document")
    filename: str = Field(description="Sanitized original document filename")
    text: str = Field(description="The extracted text content of the chunk")
    page_number: int = Field(description="1-indexed source page number in the original PDF")
    chunk_index: int = Field(description="Zero-indexed sequence number of the chunk within the document")
    char_count: int = Field(description="Number of characters in this chunk")


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful PDF upload and vector indexing."""

    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Safe display filename")
    status: str = Field(default="processed", description="Ingestion processing status")
    total_pages: int = Field(description="Total number of pages extracted")
    total_chunks: int = Field(description="Total number of text chunks created and indexed")
    created_at: str = Field(description="ISO 8601 UTC timestamp of ingestion")


class DocumentMetadata(BaseModel):
    """Detailed metadata for an uploaded document."""

    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Safe display filename")
    file_size_bytes: int = Field(description="File size in bytes")
    total_pages: int = Field(description="Total number of pages")
    total_chunks: int = Field(description="Total number of chunks stored in vector database")
    created_at: str = Field(description="ISO 8601 UTC timestamp of upload")


class DocumentListResponse(BaseModel):
    """Response returned when querying all indexed documents."""

    documents: List[DocumentMetadata] = Field(description="List of indexed document metadata records")
    total_count: int = Field(description="Total count of indexed documents")


class DocumentDeleteResponse(BaseModel):
    """Response returned upon successful deletion of a document and its chunks."""

    success: bool = Field(default=True, description="Whether the deletion succeeded")
    message: str = Field(description="User-friendly status message")
    document_id: str = Field(description="ID of the deleted document")
