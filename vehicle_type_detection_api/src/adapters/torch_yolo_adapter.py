"""
PyTorch YOLO Detection Adapter
Handles vehicle detection using PyTorch .pt models with ultralytics
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class TorchYOLODetectionAdapter:
    """PyTorch YOLO model adapter for vehicle detection using ultralytics"""

    def __init__(self, model_path: str, class_names: Optional[List[str]] = None):
        """
        Initialize PyTorch YOLO detection adapter

        Args:
            model_path: Path to the .pt model file
            class_names: List of class names for detection
        """
        self.model_path = model_path
        self.class_names = class_names or [
            "Car",
            "Motorcycle",
            "Truck",
            "Bus",
            "Bicycle",  # MaryamBoneh Vehicle-Detection classes
        ]
        self.input_size = (640, 640)  # YOLO standard input size
        self.confidence_threshold = 0.3  # Balanced threshold
        self.iou_threshold = 0.3  # Aggressive NMS for overlapping detections
        self.device = torch.device("cpu")
        self.model = None
        self._is_ready = False

        logger.info(f"Creating PyTorch YOLO adapter with model: {model_path}")
        logger.info(f"Using device: {self.device}")

        # Initialize model
        try:
            self._load_model()
            self._is_ready = True
            logger.info(f"PyTorch YOLO model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error initializing PyTorch YOLO adapter: {e}")
            self._is_ready = False
            raise

    def _load_model(self):
        """Load PyTorch YOLO model using ultralytics"""
        try:
            logger.info(f"Loading PyTorch model from {self.model_path}")

            # Load model using ultralytics YOLO
            try:
                from ultralytics import YOLO

                self.model = YOLO(self.model_path)
                # Force model to CPU
                self.model.to(self.device)
                logger.info(f"Model loaded using ultralytics YOLO on device: {self.device}")
                return
            except Exception as ultralytics_error:
                logger.warning(f"ultralytics YOLO loading failed: {ultralytics_error}")
                # Try alternative loading methods...
                raise ultralytics_error

        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            raise

    @property
    def is_ready(self) -> bool:
        """Check if adapter is ready for inference"""
        return self._is_ready and self.model is not None

    def detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect vehicles in the image using PyTorch YOLO model

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Dictionary containing detection results
        """
        if not self.is_ready:
            raise RuntimeError("PyTorch YOLO adapter is not ready for inference")

        try:
            logger.debug(f"Processing image with shape: {image.shape}")

            # Convert BGR to RGB for ultralytics models
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run inference with confidence and IoU thresholds
            results = self.model.predict(
                image_rgb, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False
            )

            # Process results
            detections = self._process_results(results, image.shape)

            logger.debug(f"Found {len(detections)} detections")

            return {
                "detections": detections,
                "model_info": {
                    "input_size": list(self.input_size),
                    "confidence_threshold": self.confidence_threshold,
                    "iou_threshold": self.iou_threshold,
                    "device": str(self.device),
                },
            }

        except Exception as e:
            logger.error(f"Error during PyTorch YOLO inference: {e}")
            raise

    def _process_results(self, results, original_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """
        Process ultralytics YOLO results into standardized format

        Args:
            results: Raw results from ultralytics YOLO model
            original_shape: Original image shape (height, width, channels)

        Returns:
            List of detection dictionaries
        """
        detections = []
        original_height, original_width = original_shape[:2]

        try:
            for result in results:
                if result.boxes is not None:
                    detections.extend(self._process_single_result(result, original_width, original_height))
        except Exception as e:
            logger.error(f"Error processing results: {e}")

        return detections

    def _process_single_result(self, result, original_width: int, original_height: int) -> List[Dict[str, Any]]:
        """Process single YOLO result"""
        detections = []
        boxes = result.boxes.cpu()

        for i in range(len(boxes)):
            detection = self._process_single_detection(boxes, i, original_width, original_height)
            if detection:
                detections.append(detection)

        return detections

    def _process_single_detection(
        self, boxes, index: int, original_width: int, original_height: int
    ) -> Optional[Dict[str, Any]]:
        """Process single detection box"""
        # Get box coordinates (xyxy format)
        x1, y1, x2, y2 = boxes.xyxy[index].numpy()
        confidence = float(boxes.conf[index].numpy())
        class_id = int(boxes.cls[index].numpy())

        # Filter by confidence threshold
        if confidence < self.confidence_threshold:
            return None

        # Get class name
        class_name = self._get_class_name(class_id)

        # Normalize coordinates
        x1, y1, x2, y2 = self._normalize_coordinates(x1, y1, x2, y2, original_width, original_height)

        # Calculate dimensions and filter small detections
        width, height, area = x2 - x1, y2 - y1, (x2 - x1) * (y2 - y1)
        if not self._is_detection_valid(area, original_width, original_height):
            return None

        detection_dict = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": width, "height": height},
        }

        logger.debug(
            f"Added detection: {class_name} ({confidence:.3f}) at " f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )
        return detection_dict

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        return self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

    def _normalize_coordinates(
        self, x1: float, y1: float, x2: float, y2: float, original_width: int, original_height: int
    ) -> Tuple[float, float, float, float]:
        """Normalize coordinates to image boundaries"""
        x1 = max(0, min(float(x1), original_width))
        y1 = max(0, min(float(y1), original_height))
        x2 = max(0, min(float(x2), original_width))
        y2 = max(0, min(float(y2), original_height))
        return x1, y1, x2, y2

    def _is_detection_valid(self, area: float, original_width: int, original_height: int) -> bool:
        """Check if detection is valid based on area"""
        min_area = (original_width * original_height) * 0.01
        if area < min_area:
            logger.debug(f"Filtering small detection: area={area:.0f} < min_area={min_area:.0f}")
            return False
        return True

    def detect_and_annotate(self, image: np.ndarray) -> np.ndarray:
        """
        Detect objects and return annotated image

        Args:
            image: Input image as numpy array

        Returns:
            Annotated image as numpy array
        """
        if not self.is_ready:
            raise RuntimeError("Model not ready or not loaded")

        try:
            # Convert BGR to RGB for ultralytics models
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform prediction
            results = self.model(image_rgb, conf=self.confidence_threshold, iou=self.iou_threshold)

            # Get annotated image
            annotated_img = results[0].plot()

            # Convert back to BGR for consistency
            annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

            return annotated_img_bgr

        except Exception as e:
            logger.error(f"Detection and annotation failed: {e}")
            raise

    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold set to: {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    def set_iou_threshold(self, threshold: float):
        """Set IoU threshold for NMS"""
        if 0.0 <= threshold <= 1.0:
            self.iou_threshold = threshold
            logger.info(f"IoU threshold set to: {threshold}")
        else:
            raise ValueError("IoU threshold must be between 0.0 and 1.0")

    def get_supported_classes(self) -> List[str]:
        """Get list of supported class names"""
        return self.class_names

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_ready:
            return {"status": "not_ready"}

        try:
            model_info = {
                "model_path": self.model_path,
                "model_type": "PyTorch YOLO (ultralytics)",
                "device": str(self.device),
                "input_size": self.input_size,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "class_names": self.class_names,
                "num_classes": len(self.class_names),
                "status": "ready",
            }

            # Add model-specific info if available
            if hasattr(self.model, "names"):
                model_info["model_classes"] = self.model.names

            return model_info

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"status": "error", "error": str(e)}

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
