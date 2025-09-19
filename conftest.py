"""
Pytest configuration and fixtures for Vehicle Type Detection API
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
import pytest
import tempfile
from unittest.mock import MagicMock

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "vehicle-type-detection" / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Import after path setup
from fastapi.testclient import TestClient
import httpx
import numpy as np
from PIL import Image


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def project_root_path():
    """Get the project root path."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def models_path(project_root_path):
    """Get the models directory path."""
    return project_root_path / "models"


@pytest.fixture(scope="session")
def samples_path(project_root_path):
    """Get the samples directory path."""
    return project_root_path / "samples"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_image_array():
    """Create a sample image as numpy array."""
    # Create a simple test image (3 channels, 640x640)
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_image_pil():
    """Create a sample PIL image."""
    # Create a simple test image
    image = Image.new('RGB', (640, 640), color='red')
    return image


@pytest.fixture
def sample_image_bytes(sample_image_pil):
    """Create sample image as bytes."""
    import io
    buffer = io.BytesIO()
    sample_image_pil.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def mock_detection_result():
    """Mock detection result."""
    return [
        {
            "class_id": 0,
            "class_name": "Car",
            "confidence": 0.95,
            "bbox": {
                "x1": 100.0,
                "y1": 100.0,
                "x2": 200.0,
                "y2": 200.0
            }
        },
        {
            "class_id": 1,
            "class_name": "Truck",
            "confidence": 0.87,
            "bbox": {
                "x1": 300.0,
                "y1": 150.0,
                "x2": 450.0,
                "y2": 300.0
            }
        }
    ]


@pytest.fixture
def mock_model_info():
    """Mock model information."""
    return {
        "model_type": "Test",
        "model_path": "/fake/path/model.pt",
        "labels_path": "/fake/path/labels.txt",
        "input_shape": [1, 3, 640, 640],
        "num_classes": 5,
        "class_names": ["Car", "Truck", "Bus", "Motorcycle", "Bicycle"],
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45
    }


@pytest.fixture
def mock_detection_adapter():
    """Mock detection adapter."""
    adapter = MagicMock()
    adapter.detect_vehicles.return_value = [
        {
            "class_id": 0,
            "class_name": "Car",
            "confidence": 0.95,
            "bbox": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 200.0}
        }
    ]
    adapter.get_model_info.return_value = {
        "model_type": "Mock",
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45
    }
    adapter.set_confidence_threshold = MagicMock()
    adapter.set_iou_threshold = MagicMock()
    return adapter


@pytest.fixture
def mock_image_adapter():
    """Mock image adapter."""
    adapter = MagicMock()
    adapter.load_image_from_bytes.return_value = np.random.randint(
        0, 255, (640, 640, 3), dtype=np.uint8
    )
    adapter.annotate_image.return_value = np.random.randint(
        0, 255, (640, 640, 3), dtype=np.uint8
    )
    adapter.image_to_bytes.return_value = b"fake_image_bytes"
    return adapter


@pytest.fixture
def mock_detection_service(mock_detection_adapter, mock_image_adapter):
    """Mock detection service."""
    service = MagicMock()
    service.detection_adapter = mock_detection_adapter
    service.image_adapter = mock_image_adapter
    service.is_ready.return_value = True
    service.get_supported_classes.return_value = ["Car", "Truck", "Bus", "Motorcycle", "Bicycle"]

    async def mock_detect_from_bytes(image_bytes):
        return {
            "success": True,
            "detections": mock_detection_adapter.detect_vehicles.return_value,
            "metadata": {"image_size": [640, 640], "processing_time": 0.1}
        }

    async def mock_detect_and_annotate_from_bytes(image_bytes):
        return b"fake_annotated_image_bytes"

    service.detect_vehicles_from_bytes = mock_detect_from_bytes
    service.detect_and_annotate_vehicles_from_bytes = mock_detect_and_annotate_from_bytes

    return service


@pytest.fixture
def app_with_mocks(mock_detection_service):
    """Create FastAPI app with mocked dependencies."""
    # This would be implemented based on your actual app structure
    # For now, return a mock
    return MagicMock()


@pytest.fixture
def client(app_with_mocks):
    """Create test client."""
    # This would create a real TestClient with your mocked app
    # For now, return a mock
    return MagicMock()


@pytest.fixture
async def async_client():
    """Create async test client."""
    # This would create a real async client
    # For now, return a mock
    return MagicMock()


# Markers
pytest.mark.asyncio = pytest.mark.asyncio
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.api = pytest.mark.api
pytest.mark.slow = pytest.mark.slow


# Test environment setup
def pytest_configure(config):
    """Configure pytest environment."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["ENVIRONMENT"] = "test"


def pytest_unconfigure(config):
    """Clean up after tests."""
    # Clean up environment variables
    os.environ.pop("TESTING", None)


# Custom test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test paths
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "api" in str(item.fspath):
            item.add_marker(pytest.mark.api)


# Skip tests based on availability
def pytest_runtest_setup(item):
    """Setup for individual tests."""
    # Skip tests based on markers and availability
    if item.get_closest_marker("pytorch"):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

    if item.get_closest_marker("onnx"):
        try:
            import onnxruntime
        except ImportError:
            pytest.skip("ONNX Runtime not available")

    if item.get_closest_marker("openvino"):
        try:
            import openvino
        except ImportError:
            pytest.skip("OpenVINO not available")

    if item.get_closest_marker("gpu"):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available for GPU test")