"""RAG query orchestration service coordinating retrieval, prompt assembly, and LLM generation."""

import logging
from typing import List, Optional

from app.config import get_settings
from app.schemas.chat import ChatQueryRequest, ChatQueryResponse
from app.services.gemini_service import GeminiService, get_gemini_service
from app.services.prompt_service import PromptService
from app.services.retrieval_service import RetrievalService, get_retrieval_service

logger = logging.getLogger("documind.rag")

NO_EVIDENCE_ANSWER = "The uploaded documents do not contain sufficient information to answer this question."


class RAGService:
    """Orchestrates end-to-end question answering pipeline."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        prompt_service: Optional[PromptService] = None,
        gemini_service: Optional[GeminiService] = None,
    ):
        self.settings = get_settings()
        self.retrieval_service = retrieval_service or get_retrieval_service()
        self.prompt_service = prompt_service or PromptService()
        self.gemini_service = gemini_service or get_gemini_service()

    def answer_query(self, request: ChatQueryRequest) -> ChatQueryResponse:
        """Process user question through retrieval, grounding, and generation.

        Args:
            request: ChatQueryRequest containing question and optional document filters.

        Returns:
            ChatQueryResponse with answer and verified source citations.
        """
        question = request.question.strip()
        logger.info("Processing RAG query: '%s' (doc_filter=%s)", question[:50], request.document_ids)

        # Step 1: Semantic Retrieval
        passages = self.retrieval_service.retrieve(
            question=question,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )

        # Step 2: Early exit if no passages met relevance threshold
        if not passages:
            logger.info("No relevant passages met distance threshold. Returning insufficient evidence response.")
            return ChatQueryResponse(
                answer=NO_EVIDENCE_ANSWER,
                sources=[],
                insufficient_evidence=True,
                model_used=None,
            )

        # Step 3: Grounded Prompt Construction
        system_instruction, user_content, citation_sources = self.prompt_service.build_prompt(
            question=question,
            passages=passages,
        )

        # Step 4: Model Inference
        raw_answer = self.gemini_service.generate_answer(
            system_instruction=system_instruction,
            user_content=user_content,
        )

        # Step 5: Check if model concluded insufficient evidence
        is_insufficient = (
            NO_EVIDENCE_ANSWER.lower() in raw_answer.lower()
            or "do not contain sufficient information" in raw_answer.lower()
        )

        return ChatQueryResponse(
            answer=raw_answer,
            sources=citation_sources,
            insufficient_evidence=is_insufficient,
            model_used=self.gemini_service.model_name,
        )


_rag_service_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Retrieve shared RAG orchestration service."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
