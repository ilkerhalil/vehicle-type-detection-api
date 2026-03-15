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


class JobStoragePort(ABC):
    """
    Port (interface) for job storage and queue management
    This defines the contract for any job storage implementation
    """

    @abstractmethod
    async def create_job(self, job_data: dict) -> str:
        """
        Create a new job in the queue

        Args:
            job_data: Dictionary containing job configuration with keys:
                - job_type: str (batch, video)
                - engine: str (pytorch, openvino)
                - data: dict (job-specific data)
                - webhook_url: str | None
                - progress_total: int

        Returns:
            Job ID (UUID string) for the created job
        """
        pass

    @abstractmethod
    async def get_job(self, job_id: str) -> dict | None:
        """
        Get job by ID

        Args:
            job_id: Unique job identifier

        Returns:
            Job dictionary or None if not found
        """
        pass

    @abstractmethod
    async def update_job(self, job_id: str, updates: dict) -> bool:
        """
        Update job fields

        Args:
            job_id: Unique job identifier
            updates: Dictionary of fields to update

        Returns:
            True if update successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_next_pending_job(self) -> dict | None:
        """
        Get the next pending job from the queue (FIFO order)

        Returns:
            Job dictionary or None if no pending jobs
        """
        pass

    @abstractmethod
    async def list_jobs(
        self, status: str | None = None, job_type: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """
        List jobs with optional filtering

        Args:
            status: Filter by status (queued, processing, completed, failed, cancelled)
            job_type: Filter by job type (batch, video)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of job dictionaries
        """
        pass

    @abstractmethod
    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a job by ID

        Args:
            job_id: Unique job identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        Delete jobs older than specified days

        Args:
            days: Number of days to keep jobs

        Returns:
            Number of jobs deleted
        """
        pass


class VideoProcessingPort(ABC):
    """
    Port (interface) for video processing operations
    This defines the contract for any video processing implementation
    """

    @abstractmethod
    def extract_frames(self, video_path: str, interval_seconds: float) -> list[np.ndarray]:
        """
        Extract frames from video at specified interval

        Args:
            video_path: Path to video file
            interval_seconds: Time interval between frames

        Returns:
            List of frame images as numpy arrays
        """
        pass

    @abstractmethod
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with keys:
                - duration_seconds: float
                - fps: float
                - resolution: str (e.g., "1920x1080")
                - codec: str
                - total_frames: int
        """
        pass

    @abstractmethod
    def create_annotated_video(self, original_path: str, output_path: str, frame_detections: list[dict]) -> None:
        """
        Create annotated video with bounding boxes

        Args:
            original_path: Path to original video file
            output_path: Path to save annotated video
            frame_detections: List of detection results for each frame
        """
        pass

    @abstractmethod
    def get_frame_timestamp(self, frame_number: int, fps: float) -> float:
        """
        Calculate timestamp for a given frame

        Args:
            frame_number: Frame index (0-based)
            fps: Frames per second

        Returns:
            Timestamp in seconds
        """
        pass
