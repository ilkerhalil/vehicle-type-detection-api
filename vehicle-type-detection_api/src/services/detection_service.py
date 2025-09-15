"""
Vehicle Detection Service for Object Detection
Orchestrates vehicle detection using YOLO model through dependency injection
"""

from typing import Any, Dict, Union

# Try to import PyTorch adapter
try:
    from ..adapters.torch_yolo_adapter import TorchYOLODetectionAdapter

    TORCH_ADAPTER_AVAILABLE = True
except ImportError:
    TorchYOLODetectionAdapter = None
    TORCH_ADAPTER_AVAILABLE = False

from ..adapters.image_adapter import OpenCVImageProcessingAdapter
from ..core.logger import setup_logger

logger = setup_logger(__name__)


class VehicleObjectDetectionService:
    """
    Service for vehicle object detection using YOLO model
    Implements the business logic for vehicle detection with dependency injection
    """

    def __init__(
        self,
        detection_adapter: Union[Any, None],  # Any YOLO adapter
        image_adapter: OpenCVImageProcessingAdapter,
    ):
        """
        Initialize vehicle detection service with injected dependencies

        Args:
            detection_adapter: YOLO vehicle detection adapter (PyTorch, OpenVINO)
            image_adapter: Image processing adapter
        """
        self.detection_adapter = detection_adapter
        self.image_adapter = image_adapter

        # Determine adapter type for logging
        adapter_type = "Unknown"
        if detection_adapter:
            adapter_class_name = detection_adapter.__class__.__name__
            if "PyTorch" in adapter_class_name or "Torch" in adapter_class_name:
                adapter_type = "PyTorch"
            elif "OpenVINO" in adapter_class_name:
                adapter_type = "OpenVINO"

        logger.info(f"Vehicle object detection service initialized with {adapter_type} adapter")

        if not detection_adapter:
            logger.warning("Detection service initialized with None adapter!")

    def detect_vehicles_in_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Detect vehicles in uploaded image

        Args:
            image_bytes: Raw image bytes

        Returns:
            Dictionary containing detection results and metadata
        """
        try:
            # Decode image from bytes
            image = self.image_adapter.decode_image_from_bytes(image_bytes)
            logger.debug(f"Image decoded: shape {image.shape}")

            # Perform detection (no preprocessing needed, YOLO adapter handles it)
            detection_result = self.detection_adapter.detect_objects(image)

            # Filter for vehicle classes and normalize to generic "Vehicle"
            vehicle_classes = {"Car", "Motorcycle", "Truck", "Bus", "Bicycle"}
            vehicle_detections = []

            for detection in detection_result["detections"]:
                if detection["class_name"] in vehicle_classes:
                    # Normalize all vehicle types to generic "Vehicle" due to model accuracy issues
                    normalized_detection = detection.copy()
                    normalized_detection["class_name"] = "Vehicle"
                    normalized_detection["original_class"] = detection["class_name"]
                    vehicle_detections.append(normalized_detection)

            # Create response
            result = {
                "total_detections": len(vehicle_detections),
                "vehicle_detections": vehicle_detections,
                "all_detections": detection_result["detections"],
                "image_info": {"width": image.shape[1], "height": image.shape[0], "channels": image.shape[2]},
                "model_info": detection_result["model_info"],
            }

            logger.info(
                f"Vehicle detection completed: {len(vehicle_detections)} vehicles found out of "
                f"{len(detection_result['detections'])} total objects"
            )
            return result

        except Exception as e:
            logger.error(f"Error during vehicle detection: {e}")
            raise

    async def detect_vehicles_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Async wrapper for detect_vehicles_in_image

        Args:
            image_bytes: Raw image bytes

        Returns:
            Dictionary containing detection results and metadata
        """
        return self.detect_vehicles_in_image(image_bytes)

    async def detect_and_annotate_vehicles_from_bytes(self, image_bytes: bytes) -> bytes:
        """
        Detect vehicles and return annotated image with bounding boxes

        Args:
            image_bytes: Raw image bytes

        Returns:
            Annotated image as bytes (JPEG format)
        """
        try:
            # Decode image from bytes
            image = self.image_adapter.decode_image_from_bytes(image_bytes)
            logger.debug(f"Image decoded for annotation: shape {image.shape}")

            # Perform detection
            detection_result = self.detection_adapter.detect_objects(image)

            # Filter for vehicle classes
            vehicle_classes = {"Car", "Motorcycle", "Truck", "Bus", "Bicycle"}
            vehicle_detections = []

            for detection in detection_result["detections"]:
                if detection["class_name"] in vehicle_classes:
                    vehicle_detections.append(detection)

            # Annotate image with bounding boxes
            annotated_image = self.image_adapter.draw_bounding_boxes(image, vehicle_detections)

            # Encode annotated image to bytes
            annotated_bytes = self.image_adapter.encode_image_to_bytes(annotated_image, format="JPEG")

            logger.info(f"Image annotation completed: {len(vehicle_detections)} vehicles annotated")
            return annotated_bytes

        except Exception as e:
            logger.error(f"Error during vehicle annotation: {e}")
            raise

    def get_supported_classes(self) -> list[str]:
        """
        Get list of supported detection classes

        Returns:
            List of class names that can be detected
        """
        if self.detection_adapter is None:
            return []
        return self.detection_adapter.get_supported_classes()

    def is_ready(self) -> bool:
        """
        Check if detection service is ready

        Returns:
            True if service is ready for detection
        """
        if self.detection_adapter is None:
            return False
        return self.detection_adapter.is_ready()
