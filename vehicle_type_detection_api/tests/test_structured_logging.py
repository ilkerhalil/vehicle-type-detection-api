"""
Tests for structured logging and correlation ID middleware.
"""

import uuid

from fastapi.testclient import TestClient
from vehicle_type_detection_api.src.main import app

client = TestClient(app)


def test_correlation_id_in_response():
    """Test that correlation ID is added to response headers."""
    response = client.get("/api/v1/health")
    assert "X-Correlation-ID" in response.headers
    # Verify it's a valid UUID format (36 characters)
    assert len(response.headers["X-Correlation-ID"]) == 36
    # Verify it's a valid UUID
    try:
        uuid.UUID(response.headers["X-Correlation-ID"])
        assert True
    except ValueError:
        assert False, "Correlation ID is not a valid UUID"


def test_correlation_id_preserved_in_header():
    """Test that correlation ID is preserved when passed in request header."""
    cid = "test-correlation-id-123"
    response = client.get("/api/v1/health", headers={"X-Correlation-ID": cid})
    assert response.headers["X-Correlation-ID"] == cid


def test_correlation_id_valid_uuid_format():
    """Test that generated correlation ID is a valid UUID."""
    response = client.get("/api/v1/health")
    correlation_id = response.headers["X-Correlation-ID"]

    # Should be a valid UUID v4
    try:
        parsed_uuid = uuid.UUID(correlation_id)
        # Verify it's version 4 (random UUID)
        assert parsed_uuid.version == 4
    except ValueError:
        assert False, "Correlation ID is not a valid UUID"


def test_correlation_id_different_per_request():
    """Test that each request gets a unique correlation ID."""
    response1 = client.get("/api/v1/health")
    response2 = client.get("/api/v1/health")

    cid1 = response1.headers["X-Correlation-ID"]
    cid2 = response2.headers["X-Correlation-ID"]

    assert cid1 != cid2, "Each request should have a unique correlation ID"


def test_correlation_id_with_post_request():
    """Test correlation ID works with POST requests."""
    response = client.post("/api/v1/batch/detect", json={"engine": "pytorch", "images": []})
    assert "X-Correlation-ID" in response.headers
    # Verify it's a valid UUID
    try:
        uuid.UUID(response.headers["X-Correlation-ID"])
        assert True
    except ValueError:
        assert False, "Correlation ID is not a valid UUID"
