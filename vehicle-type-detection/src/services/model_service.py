"""
Vehicle Detection Service
Orchestrates image processing and vehicle detection using the Hexagonal Architecture
"""

from typing import Any, Dict, List

import numpy as np

from adapters.ports import ImageProcessingPort, VehicleDetectionPort
from core.logger import setup_logger

logger = setup_logger(__name__)


class VehicleDetectionService:
    """
    Service that orchestrates vehicle detection operations
    Uses dependency injection for adapters following Hexagonal Architecture
    """

    def __init__(self, detection_adapter: VehicleDetectionPort, image_adapter: ImageProcessingPort):
        """
        Initialize service with injected adapters

        Args:
            detection_adapter: Vehicle detection port implementation
            image_adapter: Image processing port implementation
        """
        self.detection_adapter = detection_adapter
        self.image_adapter = image_adapter

        logger.info("Vehicle detection service initialized with dependency injection")

    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict vehicle type from image bytes

        Args:
            image_bytes: Image data as bytes (JPEG, PNG, etc.)

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Decode image from bytes
            image = self.image_adapter.decode_image_from_bytes(image_bytes)

            # Preprocess for model
            preprocessed_image = self.image_adapter.preprocess_image(
                image, target_size=(224, 224)  # ResNet18 standard size
            )

            # Run prediction
            result = self.detection_adapter.predict(preprocessed_image)

            logger.info(f"Prediction completed: {result['predicted_class']} (confidence: {result['confidence']:.4f})")
            return result

        except Exception as e:
            logger.error(f"Error predicting from bytes: {e}")
            raise

    def predict_from_array(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict vehicle type from numpy array

        Args:
            image: Image as numpy array (HWC format, BGR)

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess for model
            preprocessed_image = self.image_adapter.preprocess_image(
                image, target_size=(224, 224)  # ResNet18 standard size
            )

            # Run prediction
            result = self.detection_adapter.predict(preprocessed_image)

            logger.info(f"Prediction completed: {result['predicted_class']} (confidence: {result['confidence']:.4f})")
            return result

        except Exception as e:
            logger.error(f"Error predicting from array: {e}")
            raise

    def get_supported_classes(self) -> List[str]:
        """
        Get list of supported vehicle classes

        Returns:
            List of supported vehicle class names
        """
        return self.detection_adapter.get_supported_classes()

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the service is ready for predictions

        Returns:
            Dictionary containing health status
        """
        detection_ready = self.detection_adapter.is_ready()

        return {
            "status": "healthy" if detection_ready else "unhealthy",
            "detection_adapter_ready": detection_ready,
            "supported_classes": self.get_supported_classes() if detection_ready else [],
            "num_classes": len(self.get_supported_classes()) if detection_ready else 0,
        }
