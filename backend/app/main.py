"""FastAPI application entry point for DocuMind AI."""

import logging
from typing import Any, Dict
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.exceptions import DocuMindException

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("documind")

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url=f"{settings.API_V1_STR}/docs" if settings.DEBUG else None,
    redoc_url=f"{settings.API_V1_STR}/redoc" if settings.DEBUG else None,
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.DEBUG else None,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Centralized Exception Handlers
@app.exception_handler(DocuMindException)
async def documind_exception_handler(
    request: Request, exc: DocuMindException
) -> JSONResponse:
    """Handle custom application domain exceptions safely."""
    logger.warning("Domain exception at %s: %s", request.url.path, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected errors without leaking stack traces or secrets."""
    logger.error("Unhandled exception at %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "An unexpected internal server error occurred.",
        },
    )


# Health Check Endpoint
@app.get(
    f"{settings.API_V1_STR}/health",
    tags=["System"],
    summary="Service Health Check",
    response_model=Dict[str, Any],
)
async def health_check() -> Dict[str, Any]:
    """Return application operational health status and metadata."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }
