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


def test_gemini_configures_45_second_timeout_and_low_thinking():
    """Verify GeminiService configures model=gemini-3.6-flash, http_options with 45000ms timeout, and thinking_level=LOW."""
    from google.genai import types

    service = GeminiService(api_key="valid-mock-key")
    assert service.model_name == "gemini-3.6-flash"
    assert service.timeout_seconds == 45
    assert service.timeout_ms == 45000

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Answer"
    mock_client.models.generate_content.return_value = mock_response

    service._client = mock_client
    service.generate_answer("System instruction", "User content")

    # Inspect call args passed to models.generate_content
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs.get("model") == "gemini-3.6-flash"
    config = call_kwargs.get("config")
    assert config is not None
    assert config.http_options is not None
    assert config.http_options.timeout == 45000
    assert config.thinking_config is not None
    assert (
        config.thinking_config.thinking_level == types.ThinkingLevel.LOW
        or config.thinking_config.thinking_level == "LOW"
    )


def test_gemini_timeout_raises_504_after_retry(monkeypatch):
    """Verify that repeated timeout errors raise GeminiServiceException with HTTP 504 Gateway Timeout."""
    # Speed up retry delay during tests
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)

    service = GeminiService(api_key="valid-mock-key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = TimeoutError("The read operation timed out after 45.0s")

    service._client = mock_client

    with pytest.raises(GeminiServiceException) as exc_info:
        service.generate_answer("System instruction", "User content")

    assert exc_info.value.status_code == 504
    assert "timed out after 45 seconds" in exc_info.value.message
    assert exc_info.value.details["timeout_seconds"] == 45
    # Bounded to at most 2 attempts (1 initial + 1 retry)
    assert mock_client.models.generate_content.call_count == 2


def test_gemini_timeout_recovers_on_retry(monkeypatch):
    """Verify that a transient timeout on attempt 1 recovers if attempt 2 succeeds."""
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)

    service = GeminiService(api_key="valid-mock-key")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Grounded response after retry"
    mock_client.models.generate_content.side_effect = [
        TimeoutError("Connection timeout"),
        mock_response,
    ]

    service._client = mock_client
    answer = service.generate_answer("System instruction", "User content")

    assert answer == "Grounded response after retry"
    assert mock_client.models.generate_content.call_count == 2

