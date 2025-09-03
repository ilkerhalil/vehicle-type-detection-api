"""
FastAPI Dependency Providers
Provides dependency injection for adapters and services using FastAPI's Depends system
"""

from functools import lru_cache

from ..core.config import get_settings
from ..core.logger import setup_logger
from .image_adapter import OpenCVImageProcessingAdapter

# Try to import OpenVINO adapter
try:
    from .openvino_adapter import OpenVINOVehicleDetectionAdapter

    OPENVINO_AVAILABLE = True
except ImportError as e:
    print(f"OpenVINO adapter not available due to import error: {e}")
    OpenVINOVehicleDetectionAdapter = None
    OPENVINO_AVAILABLE = False

# Try to import PyTorch adapter
try:
    from .torch_yolo_adapter import TorchYOLODetectionAdapter

    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch adapter not available due to import error: {e}")
    TorchYOLODetectionAdapter = None
    TORCH_AVAILABLE = False

from ..services.detection_service import VehicleObjectDetectionService

logger = setup_logger(__name__)


@lru_cache()
def get_image_processing_adapter() -> OpenCVImageProcessingAdapter:
    """
    Get OpenCV image processing adapter

    Returns:
        OpenCVImageProcessingAdapter: Image processing adapter instance
    """
    logger.debug("Providing image processing adapter dependency")
    return OpenCVImageProcessingAdapter()


@lru_cache()
def get_openvino_detection_adapter():
    """
    Get singleton OpenVINO vehicle detection adapter

    Returns:
        OpenVINOVehicleDetectionAdapter: Singleton instance
    """
    if not OPENVINO_AVAILABLE:
        raise RuntimeError("OpenVINO adapter is not available due to import issues.")

    settings = get_settings()
    logger.debug("Providing OpenVINO detection adapter dependency")

    adapter = OpenVINOVehicleDetectionAdapter(
        model_path=str(settings.PROJECT_ROOT / "models" / "best_openvino_model"),
        labels_path=str(settings.PROJECT_ROOT / "models" / "labels.txt"),
    )

    logger.info(f"Created OpenVINO adapter instance: {type(adapter).__name__}")
    return adapter


@lru_cache()
def get_torch_yolo_detection_adapter():
    """
    Get singleton PyTorch YOLO vehicle detection adapter

    Returns:
        TorchYOLODetectionAdapter: Singleton instance
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch adapter is not available due to import issues.")

    settings = get_settings()
    logger.debug("Providing PyTorch YOLO detection adapter dependency")

    # Vehicle classes from MaryamBoneh Vehicle-Detection dataset
    vehicle_classes = ["Car", "Motorcycle", "Truck", "Bus", "Bicycle"]

    return TorchYOLODetectionAdapter(
        model_path=str(settings.PROJECT_ROOT / "models" / "best.pt"), class_names=vehicle_classes
    )


def get_openvino_vehicle_object_detection_service(detection_adapter=None, image_adapter=None):
    """
    Get vehicle object detection service with OpenVINO adapter

    Args:
        detection_adapter: OpenVINO vehicle detection adapter
        image_adapter: Image processing adapter

    Returns:
        VehicleObjectDetectionService: Service with injected dependencies
    """
    if detection_adapter is None:
        try:
            detection_adapter = get_openvino_detection_adapter()
            logger.info(f"Successfully got OpenVINO adapter: {type(detection_adapter).__name__}")
        except Exception as e:
            logger.error(f"Failed to get OpenVINO adapter: {e}")
            detection_adapter = None

    if image_adapter is None:
        image_adapter = get_image_processing_adapter()

    logger.debug("Providing OpenVINO vehicle object detection service dependency")
    logger.info(f"OpenVINO service will use adapter: {type(detection_adapter).__name__}")

    return VehicleObjectDetectionService(detection_adapter=detection_adapter, image_adapter=image_adapter)


def get_torch_vehicle_object_detection_service(detection_adapter=None, image_adapter=None):
    """
    Get vehicle object detection service with PyTorch adapter

    Args:
        detection_adapter: PyTorch YOLO vehicle detection adapter
        image_adapter: Image processing adapter

    Returns:
        VehicleObjectDetectionService: Service with injected dependencies
    """
    if detection_adapter is None:
        detection_adapter = get_torch_yolo_detection_adapter()
    if image_adapter is None:
        image_adapter = get_image_processing_adapter()

    logger.debug("Providing PyTorch vehicle object detection service dependency")
    return VehicleObjectDetectionService(detection_adapter=detection_adapter, image_adapter=image_adapter)
