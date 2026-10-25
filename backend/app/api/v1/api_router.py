"""Aggregation router for API v1 endpoints."""

from fastapi import APIRouter
from app.api.v1.endpoints import documents

api_router = APIRouter()

# Mount document endpoints
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
