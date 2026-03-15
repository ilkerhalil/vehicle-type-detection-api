"""
Batch Processing Service for Vehicle Detection
Handles synchronous and asynchronous batch image processing
"""

import base64
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Union

from vehicle_type_detection_api.src.adapters.ports import (
    ImageProcessingPort,
    JobStoragePort,
)
from vehicle_type_detection_api.src.core.logger import setup_logger

logger = setup_logger(__name__)

# Vehicle classes to filter and normalize
VEHICLE_CLASSES = {"Car", "Motorcycle", "Truck", "Bus", "Bicycle"}


class BatchProcessingService:
    """
    Service for batch processing images with vehicle detection.
    Supports both synchronous and asynchronous batch processing.
    """

    def __init__(
        self,
        detection_adapter: Union[Any, None],
        image_adapter: ImageProcessingPort,
    ):
        """
        Initialize batch processing service with injected dependencies.

        Args:
            detection_adapter: Vehicle detection adapter (PyTorch, OpenVINO, etc.)
            image_adapter: Image processing adapter for decoding/encoding
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

        logger.info(f"Batch processing service initialized with {adapter_type} adapter")

        if not detection_adapter:
            logger.warning("Batch service initialized with None detection adapter!")

    def _process_single_image(
        self,
        image_bytes: bytes,
        filename: str,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Process a single image with error handling.

        Args:
            image_bytes: Raw image bytes
            filename: Name of the image file
            confidence_threshold: Minimum confidence for detections

        Returns:
            Dictionary containing processing result with status and detections
        """
        start_time = time.time()
        result = {
            "filename": filename,
            "status": "pending",
            "detections": None,
            "processing_time_ms": None,
            "error": None,
            "error_code": None,
        }

        try:
            # Decode image from bytes
            image = self.image_adapter.decode_image_from_bytes(image_bytes)

            # Perform detection
            detection_result = self.detection_adapter.detect_objects(image)

            # Filter vehicle classes and normalize to "Vehicle"
            vehicle_detections = []
            for detection in detection_result.get("detections", []):
                if detection["class_name"] in VEHICLE_CLASSES:
                    # Normalize class name to "Vehicle"
                    normalized_detection = detection.copy()
                    normalized_detection["original_class"] = detection["class_name"]
                    normalized_detection["class_name"] = "Vehicle"
                    # Filter by confidence threshold
                    if normalized_detection.get("confidence", 0) >= confidence_threshold:
                        vehicle_detections.append(normalized_detection)

            result["status"] = "success"
            result["detections"] = vehicle_detections
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)

            logger.debug(f"Processed {filename}: {len(vehicle_detections)} vehicles detected")

        except Exception as e:
            logger.error(f"Error processing image {filename}: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            result["error_code"] = "PROCESSING_ERROR"
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)

        return result

    def process_sync_batch(
        self,
        images: List[Dict[str, str]],
        engine: str,
        confidence_threshold: float = 0.5,
        max_concurrent: int = 3,
    ) -> Dict[str, Any]:
        """
        Process a batch of images synchronously with parallel processing.

        Args:
            images: List of image dictionaries with 'filename' and 'data' (base64) keys
            engine: Detection engine used (pytorch, openvino)
            confidence_threshold: Minimum confidence for detections
            max_concurrent: Maximum number of concurrent processing threads

        Returns:
            Dictionary containing batch results with status and summary
        """
        batch_start_time = time.time()
        batch_id = str(uuid.uuid4())

        logger.info(f"Starting sync batch {batch_id} with {len(images)} images")

        results = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all tasks
            future_to_image = {}
            for img_data in images:
                try:
                    # Decode base64 image data
                    image_bytes = base64.b64decode(img_data["data"])
                    future = executor.submit(
                        self._process_single_image,
                        image_bytes,
                        img_data["filename"],
                        confidence_threshold,
                    )
                    future_to_image[future] = img_data["filename"]
                except Exception as e:
                    logger.error(f"Error decoding image {img_data.get('filename', 'unknown')}: {e}")
                    results.append(
                        {
                            "filename": img_data.get("filename", "unknown"),
                            "status": "error",
                            "error": f"Invalid base64 data: {str(e)}",
                            "error_code": "INVALID_IMAGE",
                            "detections": None,
                            "processing_time_ms": 0,
                        }
                    )

            # Collect results as they complete
            for future in as_completed(future_to_image):
                filename = future_to_image[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Unexpected error processing {filename}: {e}")
                    results.append(
                        {
                            "filename": filename,
                            "status": "error",
                            "error": str(e),
                            "error_code": "UNEXPECTED_ERROR",
                            "detections": None,
                            "processing_time_ms": 0,
                        }
                    )

        # Calculate summary statistics
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        total_detections = sum(len(r["detections"]) for r in results if r["status"] == "success" and r["detections"])

        processing_times = [r["processing_time_ms"] for r in results if r["processing_time_ms"] is not None]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        total_time_ms = int((time.time() - batch_start_time) * 1000)

        # Determine overall status
        if failed == 0:
            status = "completed"
        elif successful > 0:
            status = "partial"
        else:
            status = "failed"

        batch_result = {
            "batch_id": batch_id,
            "status": status,
            "engine": engine,
            "total_images": len(images),
            "processing_time_ms": total_time_ms,
            "results": results,
            "summary": {
                "successful": successful,
                "failed": failed,
                "total_detections": total_detections,
                "average_processing_time_ms": round(avg_processing_time, 1),
            },
        }

        logger.info(
            f"Sync batch {batch_id} completed: {successful}/{len(images)} successful, "
            f"{total_detections} total detections, {total_time_ms}ms total time"
        )

        return batch_result

    async def process_async_batch(
        self,
        job_id: str,
        images: List[Dict[str, str]],
        engine: str,
        job_storage: JobStoragePort,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Process a batch of images asynchronously with progress updates.

        Args:
            job_id: Unique job identifier
            images: List of image dictionaries with 'filename' and 'data' (base64) keys
            engine: Detection engine used (pytorch, openvino)
            job_storage: Job storage adapter for progress updates
            confidence_threshold: Minimum confidence for detections

        Returns:
            Dictionary containing batch results with status and summary
        """
        batch_start_time = time.time()
        total_images = len(images)

        logger.info(f"Starting async batch job {job_id} with {total_images} images")

        # Update job status to processing
        await job_storage.update_job(
            job_id,
            {
                "status": "processing",
                "started_at": time.time(),
                "progress_current": 0,
                "progress_total": total_images,
            },
        )

        results = []
        current_filename = None

        try:
            for idx, img_data in enumerate(images):
                current_filename = img_data.get("filename", f"image_{idx}")

                # Update progress before processing
                await job_storage.update_job(
                    job_id,
                    {
                        "progress_current": idx,
                        "progress_total": total_images,
                    },
                )

                try:
                    # Decode base64 image data
                    image_bytes = base64.b64decode(img_data["data"])

                    # Process image
                    result = self._process_single_image(
                        image_bytes,
                        current_filename,
                        confidence_threshold,
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(f"Error processing image {current_filename}: {e}")
                    results.append(
                        {
                            "filename": current_filename,
                            "status": "error",
                            "error": f"Invalid base64 data: {str(e)}",
                            "error_code": "INVALID_IMAGE",
                            "detections": None,
                            "processing_time_ms": 0,
                        }
                    )

                # Update progress after processing
                await job_storage.update_job(
                    job_id,
                    {
                        "progress_current": idx + 1,
                        "progress_total": total_images,
                    },
                )

            # Calculate summary statistics
            successful = sum(1 for r in results if r["status"] == "success")
            failed = sum(1 for r in results if r["status"] == "error")
            total_detections = sum(
                len(r["detections"]) for r in results if r["status"] == "success" and r["detections"]
            )

            processing_times = [r["processing_time_ms"] for r in results if r["processing_time_ms"] is not None]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

            total_time_ms = int((time.time() - batch_start_time) * 1000)

            # Determine overall status
            if failed == 0:
                status = "completed"
            elif successful > 0:
                status = "partial"
            else:
                status = "failed"

            batch_result = {
                "batch_id": job_id,
                "status": status,
                "engine": engine,
                "total_images": total_images,
                "processing_time_ms": total_time_ms,
                "results": results,
                "summary": {
                    "successful": successful,
                    "failed": failed,
                    "total_detections": total_detections,
                    "average_processing_time_ms": round(avg_processing_time, 1),
                },
            }

            # Update job with results
            await job_storage.update_job(
                job_id,
                {
                    "status": status,
                    "completed_at": time.time(),
                    "results": batch_result,
                    "progress_current": total_images,
                    "progress_total": total_images,
                },
            )

            logger.info(
                f"Async batch {job_id} completed: {successful}/{total_images} successful, "
                f"{total_detections} total detections, {total_time_ms}ms total time"
            )

            return batch_result

        except Exception as e:
            logger.error(f"Async batch {job_id} failed: {e}")
            await job_storage.update_job(
                job_id,
                {
                    "status": "failed",
                    "completed_at": time.time(),
                    "error": str(e),
                },
            )
            raise

    def is_ready(self) -> bool:
        """
        Check if batch processing service is ready.

        Returns:
            True if service is ready for processing
        """
        if self.detection_adapter is None:
            return False
        # is_ready is a property, not a method
        return self.detection_adapter.is_ready
