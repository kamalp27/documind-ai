"""Unit tests for the lightweight ONNX-powered EmbeddingService."""

import pytest
from unittest.mock import MagicMock, patch

from app.core.exceptions import DocuMindException
from app.services.embedding_service import EmbeddingService, get_embedding_service


def test_embedding_service_singleton():
    """Verify get_embedding_service returns a consistent singleton instance."""
    instance1 = get_embedding_service()
    instance2 = get_embedding_service()
    assert instance1 is instance2
    assert isinstance(instance1, EmbeddingService)


def test_embedding_service_lazy_loading():
    """Verify model is not loaded upon initialization."""
    service = EmbeddingService(model_name="test-model")
    assert service._model is None
    assert service.model_name == "test-model"


def test_embedding_service_empty_texts():
    """Verify passing an empty list returns an empty list without loading the model."""
    service = EmbeddingService(model_name="test-model")
    result = service.embed_texts([])
    assert result == []
    # Model should still not be loaded
    assert service._model is None


def test_embedding_service_generates_384d_vectors():
    """Verify real ONNX embedding generation outputs 384-dimensional dense float vectors."""
    service = EmbeddingService()
    texts = [
        "DocuMind AI delivers grounded document retrieval.",
        "High-performance semantic search with vector indexing.",
    ]
    embeddings = service.embed_texts(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
    assert all(isinstance(val, float) for val in embeddings[0])
    assert all(isinstance(val, float) for val in embeddings[1])


def test_embedding_service_handles_encode_interface():
    """Verify compatibility with model objects providing an .encode() method."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 384, [0.2] * 384]

    service = EmbeddingService(model_name="mock-model")
    service._model = mock_model

    result = service.embed_texts(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 384
    assert result[0][0] == 0.1
    mock_model.encode.assert_called_once_with(["hello", "world"], convert_to_numpy=True, show_progress_bar=False)


def test_embedding_service_handles_callable_interface():
    """Verify compatibility with callable model objects (like ONNXMiniLM_L6_V2)."""
    mock_callable = MagicMock(return_value=[[0.5] * 384])
    del mock_callable.encode  # Ensure hasattr(model, 'encode') is False

    service = EmbeddingService(model_name="mock-callable")
    service._model = mock_callable

    result = service.embed_texts(["sample passage"])
    assert len(result) == 1
    assert len(result[0]) == 384
    assert result[0][0] == 0.5
    mock_callable.assert_called_once_with(["sample passage"])


def test_embedding_service_model_init_failure_raises_documind_exception():
    """Verify failure during model initialization raises DocuMindException with status 500."""
    service = EmbeddingService(model_name="faulty-model")

    with patch("chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2", side_effect=RuntimeError("Download failed")):
        with pytest.raises(DocuMindException) as exc_info:
            service.embed_texts(["sample text"])
        assert exc_info.value.status_code == 500
        assert "Failed to initialize embedding model" in exc_info.value.message


def test_embedding_service_inference_failure_raises_documind_exception():
    """Verify failure during embedding inference raises DocuMindException with status 500."""
    mock_failing_model = MagicMock(side_effect=RuntimeError("Inference crashed"))
    del mock_failing_model.encode

    service = EmbeddingService(model_name="failing-model")
    service._model = mock_failing_model

    with pytest.raises(DocuMindException) as exc_info:
        service.embed_texts(["sample text"])
    assert exc_info.value.status_code == 500
    assert "Failed to generate text embeddings" in exc_info.value.message
