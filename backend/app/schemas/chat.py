"""Pydantic schemas for RAG chat query and grounded citation responses."""

import uuid
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ChatQueryRequest(BaseModel):
    """User query request for document question answering."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask about the uploaded documents.",
    )
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document IDs to restrict search to. If omitted, all documents are searched.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional maximum number of chunks to retrieve.",
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be blank or contain only whitespace.")
        return cleaned


class CitationSource(BaseModel):
    """Verifiable source citation mapped from retrieved document chunk metadata."""

    source_id: int = Field(description="1-indexed numeric citation identifier, matching inline tags like [1]")
    document_id: str = Field(description="Unique ID of the source document")
    filename: str = Field(description="Display filename of the document")
    page_numbers: List[int] = Field(description="1-indexed page number(s) in the PDF where the passage appears")
    excerpt: str = Field(description="Relevant text excerpt from the document")
    similarity_score: Optional[float] = Field(
        default=None,
        description="Semantic similarity score between 0.0 and 1.0 (higher indicates greater relevance)",
    )


class ChatQueryResponse(BaseModel):
    """Grounded question answering response with citations."""

    answer: str = Field(description="The synthesized answer or insufficient evidence refusal.")
    sources: List[CitationSource] = Field(
        default_factory=list,
        description="List of verified source citations backing the answer.",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True if uploaded documents did not contain sufficient relevant information to answer.",
    )
    model_used: Optional[str] = Field(
        default=None,
        description="Name of the generative model used, or None if answered by rule.",
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request tracing identifier.",
    )
