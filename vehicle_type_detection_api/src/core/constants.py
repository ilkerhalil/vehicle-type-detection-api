"""
Shared constants for Vehicle Type Detection API
"""

from enum import Enum


class DetectionEngine(str, Enum):
    """Supported detection engines"""

    PYTORCH = "pytorch"
    OPENVINO = "openvino"


class JobType(str, Enum):
    """Job types for queue processing"""

    BATCH = "batch"
    VIDEO = "video"


class JobStatus(str, Enum):
    """Job status values"""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Vehicle classes supported by the model
VEHICLE_CLASSES = frozenset({"Car", "Motorcycle", "Truck", "Bus", "Bicycle"})

VEHICLE_CLASS_LIST = list(VEHICLE_CLASSES)

# Class ID to name mapping
CLASS_ID_TO_NAME = {
    0: "Car",
    1: "Motorcycle",
    2: "Truck",
    3: "Bus",
    4: "Bicycle",
}

# Class name to ID mapping
CLASS_NAME_TO_ID = {name: id_ for id_, name in CLASS_ID_TO_NAME.items()}


# Supported video formats
SUPPORTED_VIDEO_FORMATS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"})

# Supported image formats
SUPPORTED_IMAGE_FORMATS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})