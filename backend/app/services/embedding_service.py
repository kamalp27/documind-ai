"""Lazy-loading Sentence Transformers embedding service."""

import logging
from typing import List, Optional
from app.config import get_settings
from app.core.exceptions import DocuMindException

logger = logging.getLogger("documind.embeddings")


class EmbeddingService:
    """Provides vector embeddings using Sentence Transformers with lazy loading."""

    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.EMBEDDING_MODEL_NAME
        self._model = None

    def _get_model(self):
        """Lazily load the SentenceTransformer model on first usage."""
        if self._model is None:
            try:
                logger.info("Lazily loading embedding model '%s'...", self.model_name)
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model '%s' loaded successfully.", self.model_name)
            except Exception as exc:
                logger.error("Failed to load embedding model '%s': %s", self.model_name, exc, exc_info=True)
                raise DocuMindException(
                    f"Failed to initialize embedding model '{self.model_name}'.",
                    status_code=500,
                    details={"error": str(exc)},
                )
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of text strings.

        Args:
            texts: List of text passages.

        Returns:
            List of float vector embeddings.
        """
        if not texts:
            return []

        model = self._get_model()
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return [vec.tolist() for vec in embeddings]
        except Exception as exc:
            logger.error("Error generating embeddings: %s", exc, exc_info=True)
            raise DocuMindException(
                "Failed to generate text embeddings.",
                status_code=500,
                details={"error": str(exc)},
            )


_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Retrieve shared singleton embedding service instance."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance
