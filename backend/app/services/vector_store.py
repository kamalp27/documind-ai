"""ChromaDB vector store service for persistent chunk indexing and deletion."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import get_settings
from app.core.exceptions import StorageException
from app.schemas.document import TextChunk

logger = logging.getLogger("documind.vector_store")


class ChromaVectorStore:
    """Manages persistent vector embedding storage and retrieval using ChromaDB."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_dir_path
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self._client = None
        self._collection = None

    def _get_collection(self):
        """Lazily initialize the persistent ChromaDB client and retrieve the collection."""
        if self._collection is None:
            try:
                import chromadb

                logger.info(
                    "Connecting to persistent ChromaDB at '%s' with collection '%s'",
                    self.persist_dir,
                    self.collection_name,
                )
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                logger.error("Failed to initialize ChromaDB collection: %s", exc, exc_info=True)
                raise StorageException(
                    "Failed to connect to local vector storage.",
                    details={"error": str(exc)},
                )
        return self._collection

    def add_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Store chunk vectors, text content, and metadata in ChromaDB.

        Uses upsert to prevent duplicate chunk failures.
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise StorageException(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})."
            )

        collection = self._get_collection()
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            logger.info("Indexed %d chunks into ChromaDB for document '%s'.", len(chunks), chunks[0].doc_id)
        except Exception as exc:
            logger.error("Error upserting chunks into ChromaDB: %s", exc, exc_info=True)
            raise StorageException(
                "Failed to store document embeddings in vector database.",
                details={"error": str(exc)},
            )

    def delete_document_chunks(self, doc_id: str) -> None:
        """Purge all vector embeddings and chunk metadata belonging to a document ID."""
        collection = self._get_collection()
        try:
            collection.delete(where={"doc_id": doc_id})
            logger.info("Successfully purged ChromaDB chunks for doc_id '%s'", doc_id)
        except Exception as exc:
            logger.error("Error deleting chunks for doc_id '%s': %s", doc_id, exc, exc_info=True)
            raise StorageException(
                f"Failed to delete document chunks from vector store.",
                details={"doc_id": doc_id, "error": str(exc)},
            )

    def get_chunk_count_for_document(self, doc_id: str) -> int:
        """Count the number of chunks stored for a specific document."""
        collection = self._get_collection()
        try:
            results = collection.get(where={"doc_id": doc_id})
            return len(results.get("ids", []))
        except Exception as exc:
            logger.warning("Could not count chunks for doc_id '%s': %s", doc_id, exc)
            return 0


_vector_store_instance: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """Retrieve shared vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = ChromaVectorStore()
    return _vector_store_instance
