"""Google Gemini API integration service with lazy client initialization."""

import logging
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
    ):
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self._client = None

    def _get_client(self):
        """Lazily initialize the google-genai Client."""
        if not self.api_key or not self.api_key.strip():
            logger.error("Attempted Gemini API call without configured GEMINI_API_KEY.")
            raise MissingAPIKeyException()

        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Google GenAI client for model '%s'.", self.model_name)
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

        try:
            from google.genai import types

            logger.info("Sending grounded prompt to Gemini (%s)...", self.model_name)
            response = client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,  # Low temperature minimizes hallucination
                    max_output_tokens=1024,
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
            logger.error("Gemini API invocation error: %s", err_str)

            # Check for rate limiting / quota exhaustion
            if "429" in err_str or "quota" in err_str.lower() or "resourceexhausted" in err_str.lower():
                raise GeminiServiceException(
                    "Gemini API quota or rate limit exceeded. Please wait a moment before trying again.",
                    status_code=429,
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
