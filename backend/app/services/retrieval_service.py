"""Semantic retrieval service utilizing ChromaDB and Sentence Transformers."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.core.exceptions import DocumentNotFoundException
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.metadata_store import DocumentMetadataStore, get_metadata_store
from app.services.vector_store import ChromaVectorStore, get_vector_store

logger = logging.getLogger("documind.retrieval")


@dataclass
class RetrievedPassage:
    """Represents a passage retrieved from the vector store with distance and citation metadata."""

    chunk_id: str
    doc_id: str
    filename: str
    page_numbers: List[int]
    text: str
    distance: float
    similarity_score: float


class RetrievalService:
    """Performs semantic vector search across indexed document chunks."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        metadata_store: Optional[DocumentMetadataStore] = None,
    ):
        self.settings = get_settings()
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.metadata_store = metadata_store or get_metadata_store()

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedPassage]:
        """Retrieve the most relevant passages for a question.

        Args:
            question: User inquiry string.
            top_k: Number of chunks to retrieve (defaults to RAG_TOP_K setting).
            document_ids: Optional list of document IDs to restrict search to.

        Returns:
            List of filtered, deduplicated RetrievedPassage objects meeting relevance threshold.

        Raises:
            DocumentNotFoundException: If any requested document_id is not found in metadata store.
        """
        # Step 1: Validate document_ids filter if provided
        where_filter: Optional[Dict[str, Any]] = None
        if document_ids:
            for doc_id in document_ids:
                # Will raise DocumentNotFoundException if invalid
                self.metadata_store.get_document(doc_id)

            if len(document_ids) == 1:
                where_filter = {"doc_id": document_ids[0]}
            else:
                where_filter = {"doc_id": {"$in": document_ids}}

        # Step 2: Embed user question using existing embedding service
        query_embeddings = self.embedding_service.embed_texts([question])
        if not query_embeddings:
            return []
        query_vec = query_embeddings[0]

        # Step 3: Query ChromaDB collection
        k = top_k or self.settings.RAG_TOP_K
        results = self.vector_store.query_similar_chunks(
            query_embedding=query_vec,
            n_results=k,
            where=where_filter,
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not ids:
            logger.info("ChromaDB returned 0 chunks for query.")
            return []

        # Step 4: Filter by relevance threshold and deduplicate
        # In ChromaDB cosine space: distance = 1 - cosine_similarity.
        # Smaller distance means higher similarity.
        passages: List[RetrievedPassage] = []
        seen_texts = set()

        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            if text in seen_texts:
                continue

            # Check relevance cutoff
            if dist > self.settings.RAG_RELEVANCE_THRESHOLD:
                logger.debug(
                    "Chunk '%s' discarded (distance %.3f exceeds threshold %.3f)",
                    chunk_id,
                    dist,
                    self.settings.RAG_RELEVANCE_THRESHOLD,
                )
                continue

            seen_texts.add(text)
            similarity = max(0.0, min(1.0, 1.0 - dist))
            page_num = meta.get("page_number", 1)

            passages.append(
                RetrievedPassage(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", "unknown"),
                    filename=meta.get("filename", "document.pdf"),
                    page_numbers=[page_num] if isinstance(page_num, int) else list(page_num),
                    text=text,
                    distance=float(dist),
                    similarity_score=round(similarity, 4),
                )
            )

        logger.info(
            "Retrieved %d relevant passages meeting threshold (<= %.2f).",
            len(passages),
            self.settings.RAG_RELEVANCE_THRESHOLD,
        )
        return passages


_retrieval_service_instance: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """Retrieve shared retrieval service instance."""
    global _retrieval_service_instance
    if _retrieval_service_instance is None:
        _retrieval_service_instance = RetrievalService()
    return _retrieval_service_instance
