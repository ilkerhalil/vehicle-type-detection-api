import pytest
from fastapi.testclient import TestClient

from vehicle_type_detection_api.src.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI application."""
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Vehicle Type Detection API" in response.text


def test_detect_endpoint_valid_image(client):
    """Test the detect endpoint with a valid image."""
    # Create a dummy test image (for testing purposes)
    with open("samples/27.jpg", "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        response = client.post("/api/v1/pytorch/detect", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "vehicles" in data


def test_detect_endpoint_no_image(client):
    """Test the detect endpoint without an image file."""
    response = client.post("/detect")
    assert response.status_code == 422  # Validation error for missing file


if __name__ == "__main__":
    pytest.main([__file__])
