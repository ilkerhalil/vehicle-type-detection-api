"""
Video Processing Service
Orchestrates video processing with vehicle detection
"""

from typing import Any

from ..adapters.dependencies import (
    get_openvino_detection_adapter,
    get_torch_yolo_detection_adapter,
)
from ..adapters.ports import VehicleDetectionPort
from ..adapters.video_adapter import OpenCVVideoAdapter
from ..core.config import get_settings
from ..core.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


class VideoProcessingService:
    """
    Service for processing videos with vehicle detection
    """

    def __init__(
        self,
        detection_adapter: VehicleDetectionPort,
        video_adapter: OpenCVVideoAdapter | None = None,
    ):
        """
        Initialize video processing service

        Args:
            detection_adapter: Vehicle detection adapter (PyTorch or OpenVINO)
            video_adapter: Video processing adapter (optional, defaults to OpenCV)
        """
        self._detection_adapter = detection_adapter
        self._video_adapter = video_adapter or OpenCVVideoAdapter()

    def process_video(
        self,
        video_path: str,
        engine: str = "pytorch",
        confidence_threshold: float = 0.5,
        frame_interval: float = 1.0,
    ) -> dict[str, Any]:
        """
        Process video and detect vehicles in each frame

        Args:
            video_path: Path to video file
            engine: Detection engine (pytorch or openvino)
            confidence_threshold: Minimum confidence for detections
            frame_interval: Interval between frames to process (seconds)

        Returns:
            Dictionary with processing results
        """
        # Get video info
        video_info = self._video_adapter.get_video_info(video_path)
        logger.info(f"Processing video: {video_path}, duration: {video_info['duration_seconds']}s")

        # Extract frames
        frames = self._video_adapter.extract_frames(video_path, frame_interval)
        logger.info(f"Extracted {len(frames)} frames for processing")

        # Process each frame
        results = []
        for idx, frame in enumerate(frames):
            # Detect vehicles
            detections = self._detection_adapter.predict(frame)

            # Filter by confidence
            filtered_detections = [
                det for det in detections.get("detections", []) if det.get("confidence", 0) >= confidence_threshold
            ]

            timestamp = self._video_adapter.get_frame_timestamp(idx, video_info["fps"] * frame_interval)

            results.append(
                {
                    "frame_number": idx,
                    "timestamp": timestamp,
                    "detections": filtered_detections,
                    "detection_count": len(filtered_detections),
                }
            )

        # Calculate summary
        total_detections = sum(r["detection_count"] for r in results)
        avg_detections = total_detections / len(results) if results else 0

        return {
            "video_info": video_info,
            "frame_interval": frame_interval,
            "total_frames_processed": len(frames),
            "results": results,
            "summary": {
                "total_detections": total_detections,
                "average_detections_per_frame": avg_detections,
                "frames_with_detections": sum(1 for r in results if r["detection_count"] > 0),
            },
        }

    def process_video_async(
        self,
        video_path: str,
        engine: str = "pytorch",
        confidence_threshold: float = 0.5,
        frame_interval: float = 1.0,
    ) -> str:
        """
        Start async video processing and return job ID

        Args:
            video_path: Path to video file
            engine: Detection engine
            confidence_threshold: Minimum confidence
            frame_interval: Frame interval in seconds

        Returns:
            Job ID for tracking
        """
        # This would integrate with job storage
        # For now, return a placeholder
        import uuid

        job_id = str(uuid.uuid4())
        logger.info(f"Created async video processing job: {job_id}")
        return job_id

    def get_supported_video_formats(self) -> list[str]:
        """Get list of supported video formats"""
        return self._video_adapter._supported_formats


def get_video_processing_service(engine: str = "pytorch") -> VideoProcessingService:
    """
    Factory function to create video processing service

    Args:
        engine: Detection engine (pytorch or openvino)

    Returns:
        VideoProcessingService instance
    """
    if engine == "openvino":
        detection_adapter = get_openvino_detection_adapter()
    else:
        detection_adapter = get_torch_yolo_detection_adapter()

    return VideoProcessingService(detection_adapter=detection_adapter)
