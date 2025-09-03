"""
OpenVINO Model Adapter for vehicle type detection
Implements the VehicleDetectionPort using OpenVINO Runtime with Singleton pattern
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import openvino as ov
except ImportError:
    raise ImportError("OpenVINO not installed. Please install with: pip install openvino")

from adapters.ports import VehicleDetectionPort
from core.logger import setup_logger

logger = setup_logger(__name__)


class OpenVINOVehicleDetectionAdapter(VehicleDetectionPort):
    """
    Singleton adapter that implements vehicle detection using OpenVINO Runtime
    """

    _instance: Optional["OpenVINOVehicleDetectionAdapter"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, model_path: str, labels_path: str):
        """
        Singleton pattern implementation with thread safety
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info(f"Creating new OpenVINO adapter instance with model: {model_path}")
                    cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self, model_path: str, labels_path: str):
        """
        Initialize the OpenVINO vehicle detection adapter (only once due to singleton)
        """
        # Prevent reinitialization in singleton
        if self._initialized:
            return

        logger.info(f"Initializing OpenVINO adapter with model: {model_path}")

        self.model_path = model_path
        self.labels_path = labels_path
        self.core = ov.Core()
        self.model = None
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
        self.class_names = []
        self.input_shape = None
        self.confidence_threshold = 0.25  # Lower threshold for better detection
        self.iou_threshold = 0.45

        try:
            self._load_model()
            self._load_labels()
            self._initialized = True
            logger.info("OpenVINO adapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenVINO adapter: {e}")
            raise

    def _load_model(self):
        """Load and compile the OpenVINO model"""
        try:
            # Load the model
            model_path = Path(self.model_path)
            if model_path.is_dir():
                # If it's a directory, look for .xml file
                xml_files = list(model_path.glob("*.xml"))
                if not xml_files:
                    raise FileNotFoundError(f"No .xml file found in {model_path}")
                xml_file = xml_files[0]
            else:
                # If it's a file path
                xml_file = model_path

            logger.info(f"Loading OpenVINO model from: {xml_file}")
            self.model = self.core.read_model(str(xml_file))

            # Compile model for CPU (you can change device as needed)
            self.compiled_model = self.core.compile_model(self.model, "CPU")

            # Get input and output layers
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)

            # Get input shape
            self.input_shape = self.input_layer.shape
            logger.info(f"Model input shape: {self.input_shape}")
            logger.info(f"Model output shape: {self.output_layer.shape}")

        except Exception as e:
            logger.error(f"Error loading OpenVINO model: {e}")
            raise

    def _load_labels(self):
        """Load class labels from labels file"""
        try:
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.class_names)} class labels: {self.class_names}")
        except Exception as e:
            logger.error(f"Error loading labels from {self.labels_path}: {e}")
            # Fallback to default labels
            self.class_names = ["Car", "Motorcycle", "Truck", "Bus", "Bicycle"]
            logger.warning(f"Using default labels: {self.class_names}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OpenVINO inference

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Preprocessed image tensor
        """
        try:
            # Get input dimensions
            _, _, input_height, input_width = self.input_shape

            # Resize image
            resized = cv2.resize(image, (input_width, input_height))

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1]
            normalized = rgb_image.astype(np.float32) / 255.0

            # Transpose to CHW format
            chw_image = np.transpose(normalized, (2, 0, 1))

            # Add batch dimension
            batch_image = np.expand_dims(chw_image, axis=0)

            return batch_image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def _postprocess_detections(self, outputs: np.ndarray, original_shape: tuple) -> List[Dict[str, Any]]:
        """
        Post-process OpenVINO model outputs to extract detections

        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (height, width)

        Returns:
            List of detection dictionaries
        """
        try:
            predictions = self._prepare_predictions(outputs)
            if predictions.shape[0] == 0:
                logger.warning("No predictions in model output")
                return []

            scale_factors = self._calculate_scale_factors(original_shape)
            detections = self._extract_detections(predictions, original_shape, scale_factors)

            # Apply NMS (Non-Maximum Suppression)
            detections = self._apply_nms(detections)

            logger.info(f"OpenVINO detection completed: {len(detections)} objects detected")
            return detections

        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _prepare_predictions(self, outputs: np.ndarray) -> np.ndarray:
        """Prepare and reshape model predictions"""
        # YOLOv8 OpenVINO output shape is typically (1, 9, num_boxes) for our model
        if len(outputs.shape) == 3:
            batch_size, features, num_boxes = outputs.shape
            if features < num_boxes:
                # Shape is (1, 9, 14196) - transpose to (1, 14196, 9)
                outputs = outputs.transpose(0, 2, 1)
            predictions = outputs[0]  # Remove batch dimension: (14196, 9)
        else:
            predictions = outputs
        return predictions

    def _calculate_scale_factors(self, original_shape: tuple) -> tuple:
        """Calculate scaling factors for coordinate conversion"""
        original_height, original_width = original_shape[:2]
        input_height, input_width = self.input_shape[2], self.input_shape[3]
        scale_x = original_width / input_width
        scale_y = original_height / input_height
        return scale_x, scale_y

    def _extract_detections(
        self, predictions: np.ndarray, original_shape: tuple, scale_factors: tuple
    ) -> List[Dict[str, Any]]:
        """Extract valid detections from predictions"""
        detections = []
        original_height, original_width = original_shape[:2]
        scale_x, scale_y = scale_factors

        valid_detections = 0
        for i, detection in enumerate(predictions):
            detection_dict = self._process_single_openvino_detection(
                detection, original_width, original_height, scale_x, scale_y
            )
            if detection_dict:
                detections.append(detection_dict)
                valid_detections += 1

        return detections

    def _process_single_openvino_detection(
        self, detection: np.ndarray, original_width: int, original_height: int, scale_x: float, scale_y: float
    ) -> Optional[Dict[str, Any]]:
        """Process single OpenVINO detection"""
        # YOLOv8 format: [x_center, y_center, width, height, class_score_0, class_score_1, ...]
        if len(detection) < 4 + len(self.class_names):
            return None

        x_center, y_center, width, height = detection[:4]

        # Get class scores (indices 4-8 for 5 classes)
        class_scores = detection[4:9]  # Fixed: Always take 5 class scores
        if len(class_scores) == 0:
            return None

        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence < self.confidence_threshold:
            return None

        # Convert to original image coordinates and format
        coords = self._convert_openvino_coordinates(
            x_center, y_center, width, height, scale_x, scale_y, original_width, original_height
        )

        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

        return {
            "class_id": int(class_id),
            "class_name": class_name,
            "confidence": float(confidence),
            "bbox": {"x1": float(coords[0]), "y1": float(coords[1]), "x2": float(coords[2]), "y2": float(coords[3])},
        }

    def _convert_openvino_coordinates(
        self,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        scale_x: float,
        scale_y: float,
        original_width: int,
        original_height: int,
    ) -> tuple:
        """Convert OpenVINO coordinates to original image coordinates"""
        # Convert to original image coordinates
        x_center *= scale_x
        y_center *= scale_y
        width *= scale_x
        height *= scale_y

        # Convert center format to corner format
        x1 = max(0, x_center - width / 2)
        y1 = max(0, y_center - height / 2)
        x2 = min(original_width, x_center + width / 2)
        y2 = min(original_height, y_center + height / 2)

        return x1, y1, x2, y2

    def _apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return detections

        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = []
        confidences = []

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            width = x2 - x1
            height = y2 - y1
            boxes.append([x1, y1, width, height])
            confidences.append(det["confidence"])

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.iou_threshold)

        if len(indices) > 0:
            filtered_detections = [detections[i] for i in indices.flatten()]
            return filtered_detections

        return []

    def detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect vehicles in the image using OpenVINO model

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Dictionary containing detection results
        """
        try:
            # Use the detect_vehicles method for object detection
            detections = self.detect_vehicles(image)

            # Format results to match expected structure
            result = {
                "detections": detections,
                "num_detections": len(detections),
                "image_shape": image.shape,
                "model_info": {
                    "engine": "OpenVINO",
                    "confidence_threshold": self.confidence_threshold,
                    "iou_threshold": self.iou_threshold,
                },
            }

            logger.info(f"OpenVINO object detection completed: {len(detections)} objects found")
            return result

        except Exception as e:
            logger.error(f"Error in OpenVINO object detection: {e}")
            return {
                "detections": [],
                "num_detections": 0,
                "image_shape": image.shape if image is not None else None,
                "error": str(e),
            }

    def detect_vehicles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the given image using OpenVINO model

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of detection dictionaries with keys: class_id, class_name, confidence, bbox
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty or invalid image provided")
                return []

            original_shape = image.shape

            # Preprocess image
            preprocessed_image = self._preprocess_image(image)

            # Run inference
            result = self.compiled_model([preprocessed_image])
            outputs = result[self.output_layer]

            # Post-process results
            detections = self._postprocess_detections(outputs, original_shape)

            logger.info(f"OpenVINO detection completed. Found {len(detections)} vehicles")
            return detections

        except Exception as e:
            logger.error(f"Error during OpenVINO vehicle detection: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": "OpenVINO",
            "model_path": self.model_path,
            "labels_path": self.labels_path,
            "input_shape": self.input_shape,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
        }

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (useful for testing or reloading)"""
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.info("OpenVINO adapter singleton instance reset")

    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for detections"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to: {self.confidence_threshold}")

    def set_iou_threshold(self, threshold: float):
        """Set IoU threshold for NMS"""
        self.iou_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"IoU threshold set to: {self.iou_threshold}")

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """
        Predict vehicle types from preprocessed image (Port interface implementation)

        Args:
            image: Preprocessed image as numpy array

        Returns:
            Dictionary with prediction results
        """
        try:
            detections = self.detect_vehicles(image)

            if not detections:
                return {
                    "predicted_class": "No vehicle detected",
                    "confidence": 0.0,
                    "all_probabilities": {},
                    "detections": [],
                }

            # Get the detection with highest confidence
            best_detection = max(detections, key=lambda x: x["confidence"])

            # Create probabilities dict from all detections
            all_probabilities = {}
            for detection in detections:
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                if class_name in all_probabilities:
                    all_probabilities[class_name] = max(all_probabilities[class_name], confidence)
                else:
                    all_probabilities[class_name] = confidence

            return {
                "predicted_class": best_detection["class_name"],
                "confidence": best_detection["confidence"],
                "all_probabilities": all_probabilities,
                "detections": detections,
            }

        except Exception as e:
            logger.error(f"Error in OpenVINO predict method: {e}")
            return {"predicted_class": "Error", "confidence": 0.0, "all_probabilities": {}, "detections": []}

    def get_supported_classes(self) -> list[str]:
        """
        Get list of supported vehicle classes (Port interface implementation)

        Returns:
            List of supported vehicle class names
        """
        return self.class_names.copy()

    def is_ready(self) -> bool:
        """
        Check if the OpenVINO detection service is ready (Port interface implementation)

        Returns:
            True if ready, False otherwise
        """
        return (
            self.compiled_model is not None
            and self.input_layer is not None
            and self.output_layer is not None
            and len(self.class_names) > 0
        )
