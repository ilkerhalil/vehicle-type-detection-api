"""
Tests for Batch Processing Router.
Tests sync batch, async batch, and job management endpoints.
"""

import base64
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add the src directory to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "vehicle_type_detection_api" / "src"))

from vehicle_type_detection_api.src.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI application."""
    with TestClient(app) as c:
        yield c


def create_test_image_data(filename="test.jpg", data=None):
    """Create a test image data dictionary with base64 encoded data."""
    if data is None:
        # Create a simple base64 encoded "image"
        data = base64.b64encode(b"fake_image_data").decode()
    return {"filename": filename, "data": data}


class TestBatchDetectSync:
    """Test synchronous batch detection endpoint."""

    def test_batch_detect_sync_success_pytorch(self, client):
        """Test successful sync batch detection with PyTorch engine."""
        # Note: This test requires the PyTorch adapter to be available
        # Create a batch with 2 images
        images = [
            create_test_image_data("car1.jpg"),
            create_test_image_data("car2.jpg"),
        ]

        request_data = {
            "engine": "pytorch",
            "images": images,
            "confidence_threshold": 0.5,
            "max_concurrent": 2,
        }

        response = client.post("/api/v1/batch/detect", json=request_data)

        # Should return 200 on success, or 503 if adapter not available
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "batch_id" in data
            assert data["engine"] == "pytorch"
            assert data["total_images"] == 2
            assert "results" in data
            assert "summary" in data
            assert data["summary"]["successful"] <= 2
        else:
            # Verify error response format
            data = response.json()
            assert "detail" in data

    def test_batch_detect_sync_success_openvino(self, client):
        """Test successful sync batch detection with OpenVINO engine."""
        images = [
            create_test_image_data("car1.jpg"),
            create_test_image_data("car2.jpg"),
        ]

        request_data = {
            "engine": "openvino",
            "images": images,
            "confidence_threshold": 0.5,
            "max_concurrent": 2,
        }

        response = client.post("/api/v1/batch/detect", json=request_data)

        # Should return 200 on success, or 503 if adapter not available
        assert response.status_code in [200, 503]

    def test_batch_detect_sync_batch_size_exceeded(self, client):
        """Test batch size exceeded validation."""
        # Create more than 10 images to trigger validation
        images = [create_test_image_data(f"car{i}.jpg") for i in range(15)]

        request_data = {
            "engine": "pytorch",
            "images": images,
            "confidence_threshold": 0.5,
            "max_concurrent": 3,
        }

        response = client.post("/api/v1/batch/detect", json=request_data)

        # Pydantic validator returns 422 for validation errors
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_batch_detect_sync_invalid_image_format(self, client):
        """Test invalid base64 image format - returns 400."""
        images = [
            {"filename": "bad.jpg", "data": "not_valid_base64!!!"},
        ]

        request_data = {
            "engine": "pytorch",
            "images": images,
            "confidence_threshold": 0.5,
            "max_concurrent": 1,
        }

        response = client.post("/api/v1/batch/detect", json=request_data)

        # Router validates base64 and returns 400 for invalid data
        assert response.status_code == 400

    def test_batch_detect_sync_empty_images(self, client):
        """Test batch with empty images list."""
        request_data = {
            "engine": "pytorch",
            "images": [],
            "confidence_threshold": 0.5,
            "max_concurrent": 3,
        }

        response = client.post("/api/v1/batch/detect", json=request_data)

        # Should accept empty batch (processed by service)
        assert response.status_code in [200, 503]

    def test_batch_detect_sync_invalid_engine(self, client):
        """Test invalid engine selection."""
        images = [create_test_image_data("test.jpg")]

        request_data = {
            "engine": "invalid_engine",
            "images": images,
            "confidence_threshold": 0.5,
            "max_concurrent": 1,
        }

        response = client.post("/api/v1/batch/detect", json=request_data)

        # Should return 400 for invalid engine
        assert response.status_code == 422  # Pydantic validation error

    def test_batch_detect_sync_invalid_confidence_threshold(self, client):
        """Test invalid confidence threshold."""
        images = [create_test_image_data("test.jpg")]

        request_data = {
            "engine": "pytorch",
            "images": images,
            "confidence_threshold": 1.5,  # Invalid: > 1.0
            "max_concurrent": 1,
        }

        response = client.post("/api/v1/batch/detect", json=request_data)

        # Should return 422 for validation error
        assert response.status_code == 422

    def test_batch_detect_sync_single_image(self, client):
        """Test batch with single image."""
        images = [create_test_image_data("single.jpg")]

        request_data = {
            "engine": "openvino",
            "images": images,
            "confidence_threshold": 0.5,
            "max_concurrent": 1,
        }

        response = client.post("/api/v1/batch/detect", json=request_data)

        # Should return 200 on success, or 503 if adapter not available
        assert response.status_code in [200, 503]


class TestBatchDetectAsync:
    """Test asynchronous batch detection endpoint."""

    def test_batch_detect_async_success(self, client):
        """Test successful async batch job creation."""
        images = [
            create_test_image_data("car1.jpg"),
            create_test_image_data("car2.jpg"),
        ]

        request_data = {
            "engine": "pytorch",
            "images": images,
            "confidence_threshold": 0.5,
            "webhook_url": "https://example.com/webhook",
        }

        response = client.post("/api/v1/jobs/batch", json=request_data)

        # Should return 202 (Accepted) on success, or 503 if adapter not available
        assert response.status_code in [202, 503]

        if response.status_code == 202:
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "queued"
            assert data["engine"] == "pytorch"
            assert data["total_images"] == 2
            assert "results_url" in data
            assert data["results_url"].startswith("/api/v1/jobs/")
            assert "created_at" in data
            assert "estimated_completion" in data
        else:
            # Verify error response format
            data = response.json()
            assert "detail" in data

    def test_batch_detect_async_batch_size_exceeded(self, client):
        """Test async batch size exceeded validation - Pydantic returns 422."""
        # Create more than 100 images to trigger validation
        images = [create_test_image_data(f"car{i}.jpg") for i in range(110)]

        request_data = {
            "engine": "pytorch",
            "images": images,
            "confidence_threshold": 0.5,
        }

        response = client.post("/api/v1/jobs/batch", json=request_data)

        # Pydantic validation returns 422 for validation errors
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_batch_detect_async_invalid_image_format(self, client):
        """Test invalid base64 image format in async request."""
        images = [
            {"filename": "bad.jpg", "data": "not_valid_base64!!!"},
        ]

        request_data = {
            "engine": "pytorch",
            "images": images,
            "confidence_threshold": 0.5,
        }

        response = client.post("/api/v1/jobs/batch", json=request_data)

        # Should return 400 with INVALID_IMAGE_FORMAT error
        assert response.status_code in [400, 503]

    def test_batch_detect_async_without_webhook(self, client):
        """Test async batch without webhook URL."""
        images = [
            create_test_image_data("car1.jpg"),
        ]

        request_data = {
            "engine": "openvino",
            "images": images,
            "confidence_threshold": 0.5,
        }

        response = client.post("/api/v1/jobs/batch", json=request_data)

        # Should return 202 or 503
        assert response.status_code in [202, 503]


class TestJobsList:
    """Test job listing endpoint."""

    def test_list_jobs_empty(self, client):
        """Test listing jobs with no jobs in queue."""
        response = client.get("/api/v1/jobs")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert isinstance(data["jobs"], list)

    def test_list_jobs_with_status_filter(self, client):
        """Test listing jobs with status filter."""
        response = client.get("/api/v1/jobs?status=queued")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data

    def test_list_jobs_with_limit(self, client):
        """Test listing jobs with limit parameter."""
        response = client.get("/api/v1/jobs?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10

    def test_list_jobs_with_offset(self, client):
        """Test listing jobs with offset parameter."""
        response = client.get("/api/v1/jobs?offset=5")

        assert response.status_code == 200
        data = response.json()
        assert data["offset"] == 5


class TestJobGet:
    """Test job retrieval endpoint."""

    def test_get_job_not_found(self, client):
        """Test getting a non-existent job."""
        response = client.get("/api/v1/jobs/non-existent-job-id")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        detail = data["detail"]
        assert "error" in detail
        assert detail["error"]["code"] == "JOB_NOT_FOUND"

    def test_get_job_created_then_retrieved(self, client):
        """Test creating a job and then retrieving it."""
        # First create a job
        images = [create_test_image_data("test.jpg")]

        request_data = {
            "engine": "pytorch",
            "images": images,
            "confidence_threshold": 0.5,
        }

        create_response = client.post("/api/v1/jobs/batch", json=request_data)

        # If job creation succeeds
        if create_response.status_code == 202:
            job_data = create_response.json()
            job_id = job_data["job_id"]

            # Now retrieve the job
            get_response = client.get(f"/api/v1/jobs/{job_id}")

            assert get_response.status_code == 200
            data = get_response.json()
            assert data["job_id"] == job_id
            assert "status" in data
            assert "engine" in data
            assert "timestamps" in data


class TestJobDelete:
    """Test job deletion endpoint."""

    def test_delete_job_not_found(self, client):
        """Test deleting a non-existent job."""
        response = client.delete("/api/v1/jobs/non-existent-job-id")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        detail = data["detail"]
        assert "error" in detail
        assert detail["error"]["code"] == "JOB_NOT_FOUND"

    def test_delete_job_success(self, client):
        """Test creating and then deleting a job."""
        # First create a job
        images = [create_test_image_data("test.jpg")]

        request_data = {
            "engine": "pytorch",
            "images": images,
            "confidence_threshold": 0.5,
        }

        create_response = client.post("/api/v1/jobs/batch", json=request_data)

        # If job creation succeeds
        if create_response.status_code == 202:
            job_data = create_response.json()
            job_id = job_data["job_id"]

            # Delete the job
            delete_response = client.delete(f"/api/v1/jobs/{job_id}")

            assert delete_response.status_code == 204

            # Verify job is gone
            get_response = client.get(f"/api/v1/jobs/{job_id}")
            assert get_response.status_code == 404


class TestBatchRouterModels:
    """Test Pydantic models validation."""

    def test_image_input_model(self):
        """Test ImageInput model validation."""
        from vehicle_type_detection_api.src.routers.batch import ImageInput

        # Valid input
        img = ImageInput(filename="test.jpg", data="base64data")
        assert img.filename == "test.jpg"
        assert img.data == "base64data"

    def test_batch_detect_request_model(self):
        """Test BatchDetectRequest model."""
        from vehicle_type_detection_api.src.routers.batch import (
            BatchDetectRequest,
            ImageInput,
        )

        images = [ImageInput(filename="test.jpg", data="base64data")]

        # Valid request
        request = BatchDetectRequest(
            engine="pytorch",
            images=images,
            confidence_threshold=0.5,
            max_concurrent=3,
        )
        assert request.engine == "pytorch"
        assert len(request.images) == 1

    def test_async_batch_request_model(self):
        """Test AsyncBatchRequest model."""
        from vehicle_type_detection_api.src.routers.batch import (
            AsyncBatchRequest,
            ImageInput,
        )

        images = [ImageInput(filename="test.jpg", data="base64data")]

        # Valid request with webhook
        request = AsyncBatchRequest(
            engine="openvino",
            images=images,
            confidence_threshold=0.5,
            webhook_url="https://example.com/webhook",
        )
        assert request.engine == "openvino"
        assert request.webhook_url == "https://example.com/webhook"

    def test_confidence_threshold_validation(self):
        """Test confidence threshold range validation."""
        from vehicle_type_detection_api.src.routers.batch import (
            BatchDetectRequest,
            ImageInput,
        )

        images = [ImageInput(filename="test.jpg", data="base64data")]

        # Valid thresholds
        BatchDetectRequest(engine="pytorch", images=images, confidence_threshold=0.0)
        BatchDetectRequest(engine="pytorch", images=images, confidence_threshold=1.0)
        BatchDetectRequest(engine="pytorch", images=images, confidence_threshold=0.5)

    def test_max_concurrent_validation(self):
        """Test max_concurrent range validation."""
        from vehicle_type_detection_api.src.routers.batch import (
            BatchDetectRequest,
            ImageInput,
        )

        images = [ImageInput(filename="test.jpg", data="base64data")]

        # Valid values
        BatchDetectRequest(engine="pytorch", images=images, max_concurrent=1)
        BatchDetectRequest(engine="pytorch", images=images, max_concurrent=3)
        BatchDetectRequest(engine="pytorch", images=images, max_concurrent=5)


class TestHelperFunctions:
    """Test helper functions."""

    def test_decode_base64_image_valid(self):
        """Test decoding valid base64 image data."""
        from vehicle_type_detection_api.src.routers.batch import decode_base64_image

        # Valid base64
        valid_data = base64.b64encode(b"test_data").decode()
        result, error = decode_base64_image(valid_data, "test.jpg")

        assert error is None
        assert result == b"test_data"

    def test_decode_base64_image_with_prefix(self):
        """Test decoding base64 with data URL prefix."""
        from vehicle_type_detection_api.src.routers.batch import decode_base64_image

        # Base64 with data URL prefix
        encoded = base64.b64encode(b"test_data").decode()
        data_with_prefix = f"data:image/jpeg;base64,{encoded}"

        result, error = decode_base64_image(data_with_prefix, "test.jpg")

        assert error is None
        assert result == b"test_data"

    def test_decode_base64_image_invalid(self):
        """Test decoding invalid base64 data."""
        from vehicle_type_detection_api.src.routers.batch import decode_base64_image

        # Invalid base64
        result, error = decode_base64_image("invalid_base64!!!", "test.jpg")

        assert error is not None
        assert "Invalid base64" in error
        assert result == b""

    def test_get_job_storage_singleton(self):
        """Test job storage singleton behavior."""
        from vehicle_type_detection_api.src.routers.batch import get_job_storage

        # Get storage twice
        storage1 = get_job_storage()
        storage2 = get_job_storage()

        # Should be the same instance
        assert storage1 is storage2


if __name__ == "__main__":
    pytest.main([__file__])
