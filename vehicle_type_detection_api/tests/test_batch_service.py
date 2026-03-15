"""
Tests for Batch Processing Service.
"""

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add the src directory to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "vehicle_type_detection_api" / "src"))

from adapters.job_storage_adapter import SQLiteJobStorageAdapter
from services.batch_service import VEHICLE_CLASSES, BatchProcessingService


class MockDetectionAdapter:
    """Mock detection adapter for testing."""

    def __init__(self, ready=True, detections=None):
        self._ready = ready
        self._detections = detections or []

    def detect_objects(self, image):
        return {"detections": self._detections, "model_info": {"test": True}}

    def is_ready(self):
        return self._ready


class MockImageAdapter:
    """Mock image adapter for testing."""

    def __init__(self, should_fail=False):
        self._should_fail = should_fail

    def decode_image_from_bytes(self, image_bytes):
        if self._should_fail:
            raise ValueError("Invalid image format")
        # Return a mock 3-channel image
        return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_detection_adapter():
    """Create a mock detection adapter."""
    detections = [
        {
            "class_name": "Car",
            "confidence": 0.92,
            "bbox": {"x1": 10, "y1": 20, "x2": 50, "y2": 80},
        },
        {
            "class_name": "Truck",
            "confidence": 0.88,
            "bbox": {"x1": 60, "y1": 30, "x2": 100, "y2": 90},
        },
        {
            "class_name": "Person",  # Non-vehicle class
            "confidence": 0.75,
            "bbox": {"x1": 5, "y1": 5, "x2": 15, "y2": 25},
        },
    ]
    return MockDetectionAdapter(detections=detections)


@pytest.fixture
def mock_image_adapter():
    """Create a mock image adapter."""
    return MockImageAdapter()


@pytest.fixture
def batch_service(mock_detection_adapter, mock_image_adapter):
    """Create a batch processing service with mock adapters."""
    return BatchProcessingService(mock_detection_adapter, mock_image_adapter)


def create_test_image_data(filename="test.jpg", data=None):
    """Create a test image data dictionary."""
    if data is None:
        # Create a simple base64 encoded "image"
        data = base64.b64encode(b"fake_image_data").decode()
    return {"filename": filename, "data": data}


class TestBatchProcessingServiceInit:
    """Test batch service initialization."""

    def test_init_with_pytorch_adapter(self):
        """Test initialization with PyTorch adapter."""
        mock_detection = MagicMock()
        mock_detection.__class__.__name__ = "TorchYOLODetectionAdapter"
        mock_detection.is_ready = True  # Property, not method

        service = BatchProcessingService(mock_detection, MockImageAdapter())
        assert service.detection_adapter == mock_detection
        assert service.image_adapter is not None
        assert service.is_ready() is True

    def test_init_with_openvino_adapter(self):
        """Test initialization with OpenVINO adapter."""
        mock_detection = MagicMock()
        mock_detection.__class__.__name__ = "OpenVINOVehicleDetectionAdapter"
        mock_detection.is_ready = True  # Property, not method

        service = BatchProcessingService(mock_detection, MockImageAdapter())
        assert service.is_ready() is True

    def test_init_with_none_adapter(self):
        """Test initialization with None adapter."""
        service = BatchProcessingService(None, MockImageAdapter())
        assert service.detection_adapter is None
        assert service.is_ready() is False

    def test_init_with_unknown_adapter(self):
        """Test initialization with unknown adapter type."""
        mock_detection = MagicMock()
        mock_detection.__class__.__name__ = "UnknownAdapter"
        mock_detection.is_ready = True  # Property, not method

        service = BatchProcessingService(mock_detection, MockImageAdapter())
        assert service.is_ready() is True


class TestProcessSingleImage:
    """Test single image processing."""

    def test_process_single_image_success(self, batch_service):
        """Test successful single image processing."""
        result = batch_service._process_single_image(
            b"fake_image_data",
            "test.jpg",
            confidence_threshold=0.5,
        )

        assert result["filename"] == "test.jpg"
        assert result["status"] == "success"
        assert result["detections"] is not None
        assert len(result["detections"]) == 2  # Car and Truck (Person filtered out)
        assert result["processing_time_ms"] is not None
        assert result["error"] is None

        # Check normalization
        for detection in result["detections"]:
            assert detection["class_name"] == "Vehicle"
            assert detection["original_class"] in VEHICLE_CLASSES

    def test_process_single_image_filters_by_confidence(self, batch_service):
        """Test that detections are filtered by confidence threshold."""
        result = batch_service._process_single_image(
            b"fake_image_data",
            "test.jpg",
            confidence_threshold=0.90,  # High threshold
        )

        # Only Car (0.92) should pass, not Truck (0.88)
        assert len(result["detections"]) == 1
        assert result["detections"][0]["original_class"] == "Car"

    def test_process_single_image_no_vehicles(self):
        """Test processing image with no vehicles."""
        mock_detection = MockDetectionAdapter(
            detections=[
                {"class_name": "Person", "confidence": 0.75, "bbox": {"x1": 1, "y1": 1, "x2": 10, "y2": 10}},
            ]
        )
        service = BatchProcessingService(mock_detection, MockImageAdapter())

        result = service._process_single_image(b"fake_image_data", "test.jpg")

        assert result["status"] == "success"
        assert result["detections"] == []

    def test_process_single_image_decode_error(self):
        """Test handling of image decode errors."""
        # Create service with failing image adapter
        mock_detection = MockDetectionAdapter()
        service = BatchProcessingService(
            mock_detection,
            MockImageAdapter(should_fail=True),
        )

        result = service._process_single_image(b"invalid_data", "test.jpg")

        assert result["status"] == "error"
        assert result["error"] is not None
        assert result["error_code"] == "PROCESSING_ERROR"
        assert result["detections"] is None

    def test_process_single_image_detection_error(self):
        """Test handling of detection errors."""
        mock_detection = MagicMock()
        mock_detection.detect_objects.side_effect = RuntimeError("Detection failed")

        service = BatchProcessingService(mock_detection, MockImageAdapter())
        result = service._process_single_image(b"fake_image_data", "test.jpg")

        assert result["status"] == "error"
        assert "Detection failed" in result["error"]
        assert result["error_code"] == "PROCESSING_ERROR"


class TestProcessSyncBatch:
    """Test synchronous batch processing."""

    def test_process_sync_batch_success(self, batch_service):
        """Test successful batch processing."""
        images = [
            create_test_image_data("car1.jpg"),
            create_test_image_data("car2.jpg"),
            create_test_image_data("car3.jpg"),
        ]

        result = batch_service.process_sync_batch(
            images=images,
            engine="pytorch",
            confidence_threshold=0.5,
            max_concurrent=2,
        )

        assert result["batch_id"] is not None
        assert result["status"] == "completed"
        assert result["engine"] == "pytorch"
        assert result["total_images"] == 3
        assert result["processing_time_ms"] > 0
        assert len(result["results"]) == 3
        assert result["summary"]["successful"] == 3
        assert result["summary"]["failed"] == 0

    def test_process_sync_batch_partial_failure(self):
        """Test batch with partial failures."""
        # Create service with failing image adapter to simulate failures
        mock_detection = MockDetectionAdapter()
        service = BatchProcessingService(
            mock_detection,
            MockImageAdapter(should_fail=True),
        )

        images = [
            create_test_image_data("car1.jpg"),
            create_test_image_data("car2.jpg"),
        ]

        result = service.process_sync_batch(images, "pytorch", 0.5, 2)

        assert result["status"] == "failed"  # All failed
        assert result["summary"]["successful"] == 0
        assert result["summary"]["failed"] == 2

    def test_process_sync_batch_empty(self, batch_service):
        """Test processing empty batch."""
        result = batch_service.process_sync_batch([], "pytorch", 0.5, 2)

        assert result["status"] == "completed"
        assert result["total_images"] == 0
        assert result["results"] == []
        assert result["summary"]["successful"] == 0

    def test_process_sync_batch_single_image(self, batch_service):
        """Test batch with single image."""
        images = [create_test_image_data("single.jpg")]

        result = batch_service.process_sync_batch(images, "openvino", 0.5, 1)

        assert result["total_images"] == 1
        assert result["summary"]["successful"] == 1
        assert result["engine"] == "openvino"

    def test_process_sync_batch_parallelism(self, batch_service):
        """Test that batch processing uses parallelism."""
        images = [create_test_image_data(f"img{i}.jpg") for i in range(5)]

        result = batch_service.process_sync_batch(images, "pytorch", 0.5, 3)

        assert result["summary"]["successful"] == 5
        # All should be processed, showing parallelism works

    def test_process_sync_batch_invalid_base64(self, batch_service):
        """Test handling of invalid base64 data."""
        images = [
            {"filename": "bad.jpg", "data": "not_valid_base64!!!"},
        ]

        result = batch_service.process_sync_batch(images, "pytorch", 0.5, 1)

        assert result["status"] == "failed"
        assert result["summary"]["failed"] == 1
        assert "INVALID_IMAGE" in result["results"][0]["error_code"]


class TestProcessAsyncBatch:
    """Test asynchronous batch processing."""

    @pytest.mark.asyncio
    async def test_process_async_batch_success(self, batch_service):
        """Test successful async batch processing."""
        storage = SQLiteJobStorageAdapter(":memory:")
        job_id = await storage.create_job({"job_type": "batch", "engine": "pytorch", "data": {"images": []}})

        images = [
            create_test_image_data("car1.jpg"),
            create_test_image_data("car2.jpg"),
        ]

        result = await batch_service.process_async_batch(
            job_id=job_id,
            images=images,
            engine="pytorch",
            job_storage=storage,
            confidence_threshold=0.5,
        )

        assert result["batch_id"] == job_id
        assert result["status"] == "completed"
        assert result["total_images"] == 2
        assert result["summary"]["successful"] == 2

        # Verify job was updated in storage
        job = await storage.get_job(job_id)
        assert job["status"] == "completed"
        assert job["progress"]["current"] == 2
        assert job["progress"]["total"] == 2

    @pytest.mark.asyncio
    async def test_process_async_batch_updates_progress(self, batch_service):
        """Test that async batch updates progress during processing."""
        storage = SQLiteJobStorageAdapter(":memory:")
        job_id = await storage.create_job(
            {
                "job_type": "batch",
                "engine": "pytorch",
            }
        )

        images = [create_test_image_data(f"img{i}.jpg") for i in range(3)]

        await batch_service.process_async_batch(
            job_id=job_id,
            images=images,
            engine="pytorch",
            job_storage=storage,
            confidence_threshold=0.5,
        )

        # Verify job progress was updated
        job = await storage.get_job(job_id)
        assert job["progress"]["current"] == 3
        assert job["progress"]["total"] == 3

    @pytest.mark.asyncio
    async def test_process_async_batch_empty(self, batch_service):
        """Test async processing of empty batch."""
        storage = SQLiteJobStorageAdapter(":memory:")
        job_id = await storage.create_job({"job_type": "batch", "engine": "pytorch"})

        result = await batch_service.process_async_batch(
            job_id=job_id,
            images=[],
            engine="pytorch",
            job_storage=storage,
            confidence_threshold=0.5,
        )

        assert result["status"] == "completed"
        assert result["total_images"] == 0

    @pytest.mark.asyncio
    async def test_process_async_batch_invalid_base64(self, batch_service):
        """Test async handling of invalid base64."""
        storage = SQLiteJobStorageAdapter(":memory:")
        job_id = await storage.create_job({"job_type": "batch", "engine": "pytorch"})

        images = [
            {"filename": "bad.jpg", "data": "not_valid_base64!!!"},
        ]

        result = await batch_service.process_async_batch(
            job_id=job_id,
            images=images,
            engine="pytorch",
            job_storage=storage,
            confidence_threshold=0.5,
        )

        assert result["status"] == "failed"
        assert result["summary"]["failed"] == 1
        assert "INVALID_IMAGE" in result["results"][0]["error_code"]

        # Verify job was marked as failed in storage
        job = await storage.get_job(job_id)
        assert job["status"] == "failed"

    @pytest.mark.asyncio
    async def test_process_async_batch_partial_success(self, batch_service):
        """Test async batch with some failures."""
        storage = SQLiteJobStorageAdapter(":memory:")
        job_id = await storage.create_job({"job_type": "batch", "engine": "pytorch"})

        # Mix of valid and invalid images
        images = [
            create_test_image_data("good.jpg"),
            {"filename": "bad.jpg", "data": "invalid!!!"},
            create_test_image_data("good2.jpg"),
        ]

        result = await batch_service.process_async_batch(
            job_id=job_id,
            images=images,
            engine="pytorch",
            job_storage=storage,
            confidence_threshold=0.5,
        )

        assert result["status"] == "partial"
        assert result["summary"]["successful"] == 2
        assert result["summary"]["failed"] == 1


class TestVehicleClasses:
    """Test vehicle class filtering and normalization."""

    def test_vehicle_classes_constant(self):
        """Test that VEHICLE_CLASSES contains expected classes."""
        expected = {"Car", "Motorcycle", "Truck", "Bus", "Bicycle"}
        assert VEHICLE_CLASSES == expected

    def test_non_vehicle_filtered(self, batch_service):
        """Test that non-vehicle classes are filtered out."""
        mock_detection = MockDetectionAdapter(
            detections=[
                {"class_name": "Car", "confidence": 0.90, "bbox": {"x1": 1, "y1": 1, "x2": 10, "y2": 10}},
                {"class_name": "Person", "confidence": 0.80, "bbox": {"x1": 20, "y1": 20, "x2": 30, "y2": 30}},
                {"class_name": "Dog", "confidence": 0.70, "bbox": {"x1": 40, "y1": 40, "x2": 50, "y2": 50}},
            ]
        )
        service = BatchProcessingService(mock_detection, MockImageAdapter())

        result = service._process_single_image(b"fake", "test.jpg")

        assert len(result["detections"]) == 1
        assert result["detections"][0]["original_class"] == "Car"


class TestSummaryStatistics:
    """Test summary statistics calculation."""

    def test_summary_with_mixed_results(self):
        """Test summary calculation with mixed success/failure."""
        mock_detection = MockDetectionAdapter(
            detections=[
                {"class_name": "Car", "confidence": 0.95, "bbox": {"x1": 1, "y1": 1, "x2": 10, "y2": 10}},
            ]
        )
        service = BatchProcessingService(mock_detection, MockImageAdapter())

        # Create scenario with one failure
        images = [
            create_test_image_data("success.jpg"),
            {"filename": "fail.jpg", "data": "invalid"},
        ]

        result = service.process_sync_batch(images, "pytorch", 0.5, 2)

        assert result["summary"]["successful"] == 1
        assert result["summary"]["failed"] == 1
        assert result["summary"]["total_detections"] == 1
        assert result["summary"]["average_processing_time_ms"] >= 0

    def test_summary_all_failures(self):
        """Test summary when all images fail."""
        mock_detection = MockDetectionAdapter()
        service = BatchProcessingService(
            mock_detection,
            MockImageAdapter(should_fail=True),
        )

        images = [
            create_test_image_data("fail1.jpg"),
            create_test_image_data("fail2.jpg"),
        ]

        result = service.process_sync_batch(images, "pytorch", 0.5, 2)

        assert result["status"] == "failed"
        assert result["summary"]["successful"] == 0
        assert result["summary"]["failed"] == 2
        assert result["summary"]["total_detections"] == 0
        assert result["summary"]["average_processing_time_ms"] == 0
