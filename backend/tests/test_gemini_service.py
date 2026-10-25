"""Unit tests for GeminiService error handling and inference encapsulation."""

from unittest.mock import MagicMock
import pytest
from app.core.exceptions import GeminiServiceException, MissingAPIKeyException
from app.services.gemini_service import GeminiService


def test_gemini_missing_api_key_raises_exception():
    """Verify calling GeminiService without an API key raises MissingAPIKeyException."""
    service = GeminiService(api_key="")
    with pytest.raises(MissingAPIKeyException) as exc_info:
        service.generate_answer("System instruction", "User content")
    assert exc_info.value.status_code == 500
    assert "GEMINI_API_KEY" in exc_info.value.message


def test_gemini_empty_response_raises_exception():
    """Verify empty or safety-filtered model response raises GeminiServiceException with 502."""
    service = GeminiService(api_key="valid-mock-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = None  # Model returned no text (e.g. safety filter)
    mock_client.models.generate_content.return_value = mock_response

    service._client = mock_client

    with pytest.raises(GeminiServiceException) as exc_info:
        service.generate_answer("System instruction", "User content")
    assert exc_info.value.status_code == 502
    assert "empty" in exc_info.value.message.lower() or "blocked" in exc_info.value.message.lower()


def test_gemini_rate_limit_429():
    """Verify quota or rate limit error raises GeminiServiceException with 429."""
    service = GeminiService(api_key="valid-mock-key")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 ResourceExhausted: Quota exceeded for model")

    service._client = mock_client

    with pytest.raises(GeminiServiceException) as exc_info:
        service.generate_answer("System instruction", "User content")
    assert exc_info.value.status_code == 429
    assert "rate limit" in exc_info.value.message.lower() or "quota" in exc_info.value.message.lower()


def test_gemini_successful_generation():
    """Verify successful model response is returned cleanly."""
    service = GeminiService(api_key="valid-mock-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "  Based on [1], ChromaDB provides persistent vector search.  "
    mock_client.models.generate_content.return_value = mock_response

    service._client = mock_client

    result = service.generate_answer("System instruction", "User content")
    assert result == "Based on [1], ChromaDB provides persistent vector search."
