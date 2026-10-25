"""Aggregation router for API v1 endpoints."""

from fastapi import APIRouter
from app.api.v1.endpoints import chat, documents

api_router = APIRouter()

# Mount document endpoints
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])

# Mount chat / RAG query endpoints
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
