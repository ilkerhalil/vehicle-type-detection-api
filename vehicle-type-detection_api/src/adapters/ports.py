"""
Port definitions for vehicle type detection
Following Hexagonal Architecture patterns
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VehicleDetectionPort(ABC):
    """
    Port (interface) for vehicle type detection
    This defines the contract for any vehicle detection implementation
    """

    @abstractmethod
    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """
        Predict vehicle type from preprocessed image

        Args:
            image: Preprocessed image as numpy array (NCHW format)

        Returns:
            Dictionary containing prediction results with keys:
            - predicted_class: str
            - confidence: float
            - all_probabilities: dict[str, float]
        """
        pass

    @abstractmethod
    def get_supported_classes(self) -> list[str]:
        """
        Get list of supported vehicle classes

        Returns:
            List of supported vehicle class names
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the detection service is ready to make predictions

        Returns:
            True if ready, False otherwise
        """
        pass


class ImageProcessingPort(ABC):
    """
    Port for image processing operations
    """

    @abstractmethod
    def preprocess_image(self, image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """
        Preprocess image for model inference

        Args:
            image: Input image as numpy array (HWC format, BGR)
            target_size: Target size as (width, height)

        Returns:
            Preprocessed image ready for model inference (NCHW format)
        """
        pass

    @abstractmethod
    def decode_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Decode image from bytes

        Args:
            image_bytes: Image data as bytes

        Returns:
            Decoded image as numpy array (HWC format, BGR)
        """
        pass
