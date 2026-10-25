"""Chat API endpoints for grounded document question answering."""

import logging
from fastapi import APIRouter, Depends, status

from app.schemas.chat import ChatQueryRequest, ChatQueryResponse
from app.services.rag_service import RAGService, get_rag_service

logger = logging.getLogger("documind.api.chat")

router = APIRouter()


@router.post(
    "/query",
    response_model=ChatQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Documents with RAG",
    description=(
        "Retrieves the most semantically relevant document passages from ChromaDB, "
        "assembles a grounded prompt with security boundaries, and invokes Google Gemini "
        "to generate a verifiable response with source citations."
    ),
)
async def query_documents(
    request: ChatQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatQueryResponse:
    """Execute a grounded RAG query against indexed documents."""
    return rag_service.answer_query(request)
