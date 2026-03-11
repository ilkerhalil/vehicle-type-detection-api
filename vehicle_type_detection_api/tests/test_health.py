"""
Tests for the enhanced health check endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from vehicle_type_detection_api.src.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI application."""
    with TestClient(app) as c:
        yield c


def test_health_check_returns_200(client):
    """Test that health check endpoint returns 200 status code."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_check_response_structure(client):
    """Test that health check response contains expected structure."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data
    assert "correlation_id" in data
    assert "components" in data
    assert "metrics" in data


def test_health_check_version_is_1_1_0(client):
    """Test that health check returns version 1.1.0."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["version"] == "1.1.0"


def test_health_check_components_structure(client):
    """Test that health check response has proper components structure."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    components = data["components"]

    # Check required components exist
    assert "api" in components
    assert "pytorch" in components
    assert "openvino" in components
    assert "job_queue" in components

    # Check each component has a status
    for component_name in ["api", "pytorch", "openvino", "job_queue"]:
        assert "status" in components[component_name], f"{component_name} missing status"


def test_health_check_status_is_valid(client):
    """Test that overall status is one of the expected values."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    valid_statuses = ["healthy", "degraded", "unhealthy"]
    assert data["status"] in valid_statuses


def test_health_check_correlation_id_exists(client):
    """Test that correlation_id is present and is a valid UUID."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "correlation_id" in data
    correlation_id = data["correlation_id"]
    # Check it's a valid UUID format (36 characters with dashes)
    assert len(correlation_id) == 36
    assert correlation_id.count("-") == 4


def test_health_check_timestamp_is_valid_isoformat(client):
    """Test that timestamp is a valid ISO format string."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "timestamp" in data
    timestamp = data["timestamp"]
    # Should contain T and Z for UTC ISO format, or at least be parseable
    assert isinstance(timestamp, str)
    assert "T" in timestamp


def test_health_check_job_queue_has_counts(client):
    """Test that job queue component has job counts if available."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    job_queue = data["components"]["job_queue"]

    # If job queue is healthy, it should have counts
    if job_queue["status"] == "healthy":
        assert "pending_jobs" in job_queue
        assert "processing_jobs" in job_queue
        assert isinstance(job_queue["pending_jobs"], int)
        assert isinstance(job_queue["processing_jobs"], int)


def test_health_check_metrics_is_dict(client):
    """Test that metrics is a dictionary."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["metrics"], dict)


if __name__ == "__main__":
    pytest.main([__file__])
