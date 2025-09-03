"""
OpenCV Image Processing Adapter
Implements ImageProcessingPort using OpenCV
"""

import cv2
import numpy as np

from ..core.logger import setup_logger
from .ports import ImageProcessingPort

logger = setup_logger(__name__)


class OpenCVImageProcessingAdapter(ImageProcessingPort):
    """
    Adapter that implements image processing using OpenCV
    """

    def preprocess_image(self, image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """
        Preprocess image for model inference using OpenCV

        Args:
            image: Input image as numpy array (HWC format, BGR)
            target_size: Target size as (width, height)

        Returns:
            Preprocessed image ready for model inference (NCHW format)
        """
        try:
            # Validate input
            if len(image.shape) != 3:
                raise ValueError("Input image must be a color image (3 channels)")

            logger.debug(f"Original image shape: {image.shape}, dtype: {image.dtype}")
            logger.debug(f"Original image min/max: {image.min()}/{image.max()}")

            # Resize image to target size
            resized_image = cv2.resize(image, target_size)
            logger.debug(f"Resized image shape: {resized_image.shape}")

            # Keep BGR format (model expects BGR, not RGB!)
            # Model documentation specifies BGR format
            bgr_image = resized_image  # OpenCV loads as BGR, which is what we need
            logger.debug(f"BGR image min/max: {bgr_image.min()}/{bgr_image.max()}")

            # Convert to float32 for processing
            float_image = bgr_image.astype(np.float32)
            logger.debug(
                f"Float image min/max/mean: {float_image.min():.3f}/{float_image.max():.3f}/{float_image.mean():.3f}"
            )

            # Apply model-specific mean subtraction (BGR format)
            # Model documentation: Mean subtraction: [103.939, 116.779, 123.68] (BGR order)
            mean_bgr = np.array([103.939, 116.779, 123.68])  # B, G, R
            logger.debug(f"Using BGR mean subtraction: {mean_bgr}")

            preprocessed_image = float_image - mean_bgr
            logger.debug(
                f"After mean subtraction min/max/mean: "
                f"{preprocessed_image.min():.3f}/{preprocessed_image.max():.3f}/{preprocessed_image.mean():.3f}"
            )

            # Convert to NCHW format (batch, channels, height, width)
            nchw_image = np.transpose(preprocessed_image, (2, 0, 1))
            batched_image = np.expand_dims(nchw_image, axis=0)

            logger.debug(f"Image preprocessed: {image.shape} -> {batched_image.shape}")
            logger.debug(
                f"Final processed image stats: min={batched_image.min():.3f}, "
                f"max={batched_image.max():.3f}, mean={batched_image.mean():.3f}"
            )

            return batched_image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def decode_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Decode image from bytes using OpenCV

        Args:
            image_bytes: Image data as bytes

        Returns:
            Decoded image as numpy array (HWC format, BGR)
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)

            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Could not decode image from bytes")

            logger.debug(f"Image decoded from bytes: shape {image.shape}")
            return image

        except Exception as e:
            logger.error(f"Error decoding image from bytes: {e}")
            raise

    def draw_bounding_boxes(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes on image

        Args:
            image: Input image as numpy array (HWC format, BGR)
            detections: List of detection dictionaries with bbox and class info

        Returns:
            Annotated image with bounding boxes
        """
        try:
            # Create a copy to avoid modifying original image
            annotated_image = image.copy()

            # Define colors for different classes (BGR format)
            colors = {
                "Car": (0, 255, 0),  # Green
                "Motorcycle": (255, 0, 0),  # Blue
                "Truck": (0, 0, 255),  # Red
                "Bus": (255, 255, 0),  # Cyan
                "Bicycle": (0, 255, 255),  # Yellow
                "Vehicle": (0, 255, 0),  # Green (default)
            }

            for detection in detections:
                # Get bounding box coordinates
                bbox = detection.get("bbox", {})
                x1 = int(bbox.get("x1", 0))
                y1 = int(bbox.get("y1", 0))
                x2 = int(bbox.get("x2", 0))
                y2 = int(bbox.get("y2", 0))

                # Get class information
                class_name = detection.get("class_name", "Unknown")
                confidence = detection.get("confidence", 0.0)

                # Get color for this class
                color = colors.get(class_name, (0, 255, 0))

                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # Create label text
                label = f"{class_name}: {confidence:.2f}"

                # Get text size for background rectangle
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Draw background rectangle for text
                cv2.rectangle(annotated_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)

                # Draw text
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # White text
                    1,
                )

            logger.debug(f"Drew {len(detections)} bounding boxes on image")
            return annotated_image

        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {e}")
            raise

    def encode_image_to_bytes(self, image: np.ndarray, format: str = "JPEG") -> bytes:
        """
        Encode image to bytes

        Args:
            image: Input image as numpy array (HWC format, BGR)
            format: Output format (JPEG, PNG, etc.)

        Returns:
            Encoded image as bytes
        """
        try:
            # Set encoding parameters based on format
            if format.upper() == "JPEG":
                ext = ".jpg"
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
            elif format.upper() == "PNG":
                ext = ".png"
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
            else:
                ext = ".jpg"
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]

            # Encode image
            success, encoded_image = cv2.imencode(ext, image, encode_params)

            if not success:
                raise ValueError(f"Could not encode image to {format} format")

            # Convert to bytes
            image_bytes = encoded_image.tobytes()

            logger.debug(f"Image encoded to {format}: {len(image_bytes)} bytes")
            return image_bytes

        except Exception as e:
            logger.error(f"Error encoding image to bytes: {e}")
            raise
