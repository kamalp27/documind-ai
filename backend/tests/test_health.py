"""Tests for application health check, configuration, and exception handling."""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.core.exceptions import DocuMindException
from app.main import app

client = TestClient(app)


def test_health_check_success():
    """Verify that the health check endpoint returns 200 OK and expected structure."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "DocuMind AI"
    assert "version" in data
    assert "environment" in data


def test_health_check_payload_safety():
    """Ensure no system paths, file locations, or sensitive keys leak in health response."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    # Check that forbidden strings / leaked details do not exist
    for key, value in data.items():
        assert "password" not in key.lower()
        assert "secret" not in key.lower()
        assert "key" not in key.lower()
        if isinstance(value, str):
            assert "C:\\" not in value
            assert "/home/" not in value
            assert "antigravity" not in value.lower()


def test_not_found_endpoint():
    """Verify that requesting an unregistered path returns 404."""
    response = client.get("/api/v1/unregistered-route")
    assert response.status_code == 404


def test_cors_preflight_headers():
    """Verify that CORS preflight requests receive the configured allowed origin."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
    }
    response = client.options("/api/v1/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_settings_properties():
    """Verify configuration model defaults and helper methods."""
    settings = get_settings()
    assert settings.PROJECT_NAME == "DocuMind AI"
    assert settings.API_V1_STR == "/api/v1"
    assert isinstance(settings.cors_origins_list, list)
    assert "http://localhost:5173" in settings.cors_origins_list
    assert settings.RAG_RELEVANCE_THRESHOLD == 0.85


def test_custom_domain_exception_handling():
    """Verify centralized exception handler serializes DocuMindException properly."""
    # Register a dynamic temporary test route on the app
    @app.get("/api/v1/test-exception")
    def trigger_exception():
        raise DocuMindException(
            message="Test domain error message",
            status_code=400,
            details={"field": "test_field"},
        )

    response = client.get("/api/v1/test-exception")
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "Test domain error message"
    assert data["details"] == {"field": "test_field"}
