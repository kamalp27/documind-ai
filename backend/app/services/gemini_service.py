"""Google Gemini API integration service with lazy client initialization."""

import logging
import random
import re
import time
from typing import Optional
from app.config import get_settings
from app.core.exceptions import GeminiServiceException, MissingAPIKeyException

logger = logging.getLogger("documind.gemini")


class GeminiService:
    """Handles communication with the Google Gemini API for grounded generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else getattr(settings, "GEMINI_TIMEOUT_SECONDS", 20)
        )
        self.timeout_ms = int(self.timeout_seconds * 1000)
        self._client = None

    def _get_client(self):
        """Lazily initialize the google-genai Client."""
        if not self.api_key or not self.api_key.strip():
            logger.error("Attempted Gemini API call without configured GEMINI_API_KEY.")
            raise MissingAPIKeyException()

        if self._client is None:
            try:
                from google import genai
                from google.genai import types

                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(timeout=self.timeout_ms),
                )
                logger.info(
                    "Initialized Google GenAI client for model '%s' (timeout=%ds).",
                    self.model_name,
                    self.timeout_seconds,
                )
            except Exception as exc:
                logger.error("Failed to initialize Google GenAI SDK: %s", exc, exc_info=True)
                raise GeminiServiceException(
                    "Failed to initialize Gemini API client.",
                    status_code=500,
                    details={"error": str(exc)},
                )

        return self._client

    def generate_answer(
        self,
        system_instruction: str,
        user_content: str,
    ) -> str:
        """Generate grounded answer using Google Gemini.

        Args:
            system_instruction: Trusted system directives.
            user_content: Grounded prompt containing <document_context> and user question.

        Returns:
            Generated answer text.

        Raises:
            MissingAPIKeyException: If GEMINI_API_KEY is not configured.
            GeminiServiceException: If API call fails, times out, or output is blocked.
        """
        client = self._get_client()

        max_retries = 3
        max_timeout_retries = 2  # Bounded to at most 1 retry on timeout to keep total request duration <= 42s

        for attempt in range(max_retries):
            try:
                from google.genai import types

                logger.info(
                    "Sending grounded prompt to Gemini (%s) [attempt %d/%d, timeout=%ds]...",
                    self.model_name,
                    attempt + 1,
                    max_retries,
                    self.timeout_seconds,
                )
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1,  # Low temperature minimizes hallucination
                        max_output_tokens=1024,
                        http_options=types.HttpOptions(timeout=self.timeout_ms),
                    ),
                )

                if not response or not response.text:
                    logger.warning("Gemini returned empty or filtered response.")
                    raise GeminiServiceException(
                        "The model returned an empty response or the content was blocked by safety filters.",
                        status_code=502,
                    )

                return response.text.strip()

            except MissingAPIKeyException:
                raise
            except GeminiServiceException:
                raise
            except Exception as exc:
                err_str = str(exc)
                logger.error("Gemini API invocation error on attempt %d: %s", attempt + 1, err_str)

                # Check for client-side or gateway timeout
                is_timeout = (
                    "timeout" in err_str.lower()
                    or "timed out" in err_str.lower()
                    or "deadline" in err_str.lower()
                    or "504" in err_str
                    or isinstance(exc, TimeoutError)
                )

                if is_timeout:
                    if attempt < max_timeout_retries - 1:
                        logger.warning(
                            "Gemini request timed out after %ds; retrying once (attempt %d/%d)...",
                            self.timeout_seconds,
                            attempt + 1,
                            max_timeout_retries,
                        )
                        time.sleep(1.0)
                        continue
                    raise GeminiServiceException(
                        f"The Gemini model request timed out after {self.timeout_seconds} seconds. Please try again.",
                        status_code=504,
                        details={"error_type": type(exc).__name__, "timeout_seconds": self.timeout_seconds},
                    )

                # Bounded retry strictly for transient 503 Service Unavailable / demand spikes
                # Do NOT retry 400 (invalid request), 401/403 (auth/permissions), 404 (model missing), or 429 (quota exhausted)
                is_transient_503 = (
                    ("503" in err_str or "unavailable" in err_str.lower())
                    and not any(code in err_str for code in ["400", "401", "403", "404", "429"])
                    and "quota" not in err_str.lower()
                    and "resourceexhausted" not in err_str.lower()
                )

                if is_transient_503 and attempt < max_retries - 1:
                    # Exponential backoff (1.5s, 3.0s) + random jitter (0.1s - 0.7s)
                    backoff = 1.5 * (2 ** attempt) + random.uniform(0.1, 0.7)

                    # Check for explicit short retryDelay guidance if present
                    retry_match = re.search(r"retryDelay':\s*'(\d+)s", err_str)
                    if retry_match:
                        suggested_delay = float(retry_match.group(1))
                        if 0 < suggested_delay <= 10.0:
                            backoff = max(backoff, suggested_delay + random.uniform(0.1, 0.5))

                    logger.warning("Gemini 503 spike encountered; retrying in %.2fs (attempt %d/%d)...", backoff, attempt + 1, max_retries)
                    time.sleep(backoff)
                    continue

                # Check for rate limiting / quota exhaustion
                if "429" in err_str or "quota" in err_str.lower() or "resourceexhausted" in err_str.lower():
                    raise GeminiServiceException(
                        "Gemini API quota or rate limit exceeded. Please wait a moment before trying again.",
                        status_code=429,
                    )

                # Check for 503 service unavailable after retries exhausted
                if "503" in err_str or "unavailable" in err_str.lower():
                    raise GeminiServiceException(
                        "Gemini model is temporarily unavailable due to high demand. Please try again shortly.",
                        status_code=503,
                    )

                # Generic safe application error without leaking credentials
                raise GeminiServiceException(
                    "An error occurred while communicating with the Gemini API.",
                    status_code=502,
                    details={"error_type": type(exc).__name__},
                )


_gemini_service_instance: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Retrieve shared Gemini service instance."""
    global _gemini_service_instance
    if _gemini_service_instance is None:
        _gemini_service_instance = GeminiService()
    return _gemini_service_instance
