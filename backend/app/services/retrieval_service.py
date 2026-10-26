"""Semantic retrieval service utilizing ChromaDB and Sentence Transformers."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.core.exceptions import DocumentNotFoundException
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.metadata_store import DocumentMetadataStore, get_metadata_store
from app.services.vector_store import ChromaVectorStore, get_vector_store

logger = logging.getLogger("documind.retrieval")

# Regular expression patterns conservatively identifying high-level summary/overview queries
SUMMARY_PATTERNS = [
    r"\bsummar(?:y|ies|ize|ise|izing|ising)\b",
    r"\boverview\b",
    r"\bkey\s+(?:takeaways?|points?|highlights?|findings?)\b",
    r"\bmain\s+(?:takeaways?|points?|topics?|ideas?)\b",
    r"\btl;?dr\b",
    r"\bwhat\s+(?:(?:is|are)\s+(?:this|the)\s+document|(?:this|the)\s+document\s+(?:is|are))\s+about\b",
    r"\bwhat\s+does\s+(?:this|the)\s+document\s+(?:cover|discuss|describe)\b",
    r"\b(?:give|provide)\s+(?:me\s+)?(?:a\s+|an\s+)?(?:high[- ]level\s+)?(?:summary|overview|outline)\b",
    r"\bdocument\s+outline\b",
    r"\bsynopsis\b",
]


def is_summary_query(question: str) -> bool:
    """Detect whether a query exhibits a broad summarization or overview intent.

    Avoids matching arbitrary questions just because they contain the word 'document'.
    """
    if not question:
        return False
    q_clean = question.strip().lower()
    return any(re.search(pat, q_clean) for pat in SUMMARY_PATTERNS)


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

        # Step 5: Intent-specific, single-document fallback for broad summary queries
        # If normal semantic retrieval yields 0 passages meeting RAG_RELEVANCE_THRESHOLD,
        # but the query expresses a high-level summary intent and exactly one valid document ID
        # was explicitly specified, retrieve structurally ordered introductory chunks from that document.
        if not passages and document_ids and len(document_ids) == 1 and is_summary_query(question):
            logger.info(
                "Normal retrieval returned 0 qualifying passages for summary query. "
                "Triggering document-scoped summary fallback for doc_id '%s'.",
                document_ids[0],
            )
            passages = self._retrieve_document_summary_fallback(
                doc_id=document_ids[0],
                query_vec=query_vec,
                top_k=k,
                known_distances=dict(zip(ids, distances)),
            )

        logger.info(
            "Retrieved %d relevant passages meeting threshold (<= %.2f).",
            len(passages),
            self.settings.RAG_RELEVANCE_THRESHOLD,
        )
        return passages

    @staticmethod
    def _compute_cosine_distance(vec1: Any, vec2: Any) -> float:
        """Compute cosine distance between two 1D numeric vectors."""
        try:
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 1.0
            cos_sim = dot / (norm1 * norm2)
            return float(max(0.0, 1.0 - cos_sim))
        except Exception:
            return 1.0

    def _retrieve_document_summary_fallback(
        self,
        doc_id: str,
        query_vec: Optional[List[float]] = None,
        top_k: int = 4,
        known_distances: Optional[Dict[str, float]] = None,
    ) -> List[RetrievedPassage]:
        """Fallback retrieval mechanism for broad document-scoped summary queries.

        Retrieves a bounded set of structurally useful, introductory chunks strictly from
        the requested document, ordered by chunk_index, when normal semantic retrieval
        rejects all candidates due to high cosine distance.

        Args:
            doc_id: Validated document ID to fetch chunks from.
            query_vec: Optional query embedding for calculating cosine distance.
            top_k: Maximum number of structural chunks to return.
            known_distances: Mapping of chunk_id to measured distance from initial query.

        Returns:
            List of RetrievedPassage objects ordered structurally by chunk_index.
        """
        raw_chunks = self.vector_store.get_chunks_for_document(doc_id)
        raw_ids = raw_chunks.get("ids") or []
        raw_docs = raw_chunks.get("documents") or []
        raw_metas = raw_chunks.get("metadatas") or []
        raw_embeddings = raw_chunks.get("embeddings")

        if not raw_ids:
            logger.info("Summary fallback: No stored chunks found for doc_id '%s'.", doc_id)
            return []

        known_distances = known_distances or {}

        # Collect candidates belonging strictly to the requested doc_id
        items = []
        for idx, (c_id, doc_text, meta) in enumerate(zip(raw_ids, raw_docs, raw_metas)):
            text_clean = (doc_text or "").strip()
            if not text_clean:
                continue
            if meta.get("doc_id") and meta.get("doc_id") != doc_id:
                continue

            chunk_emb = None
            if raw_embeddings is not None and idx < len(raw_embeddings):
                chunk_emb = raw_embeddings[idx]

            items.append({
                "chunk_id": c_id,
                "text": doc_text,
                "meta": meta,
                "chunk_index": meta.get("chunk_index", 0),
                "embedding": chunk_emb,
            })

        if not items:
            return []

        # Filter out tiny noise chunks if substantial chunks are present
        meaningful_items = [it for it in items if len(it["text"].strip()) >= 20]
        if meaningful_items:
            items = meaningful_items

        # Structurally order chunks: introductory chunks first (chunk_index ascending)
        items.sort(key=lambda x: x["chunk_index"])

        passages: List[RetrievedPassage] = []
        seen_texts = set()

        for it in items:
            text = it["text"]
            if text in seen_texts:
                continue
            seen_texts.add(text)

            c_id = it["chunk_id"]
            meta = it["meta"]

            # Compute or lookup distance
            if c_id in known_distances:
                dist = known_distances[c_id]
            elif query_vec is not None and it["embedding"] is not None:
                dist = self._compute_cosine_distance(query_vec, it["embedding"])
            else:
                dist = self.settings.RAG_RELEVANCE_THRESHOLD

            similarity = max(0.0, min(1.0, 1.0 - dist))
            page_num = meta.get("page_number", 1)

            passages.append(
                RetrievedPassage(
                    chunk_id=c_id,
                    doc_id=doc_id,
                    filename=meta.get("filename", "document.pdf"),
                    page_numbers=[page_num] if isinstance(page_num, int) else list(page_num),
                    text=text,
                    distance=float(dist),
                    similarity_score=round(similarity, 4),
                )
            )

            if len(passages) >= top_k:
                break

        logger.info(
            "Summary fallback activated for doc_id '%s': returned %d structural passages.",
            doc_id,
            len(passages),
        )
        return passages


_retrieval_service_instance: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """Retrieve shared retrieval service instance."""
    global _retrieval_service_instance
    if _retrieval_service_instance is None:
        _retrieval_service_instance = RetrievalService()
    return _retrieval_service_instance
