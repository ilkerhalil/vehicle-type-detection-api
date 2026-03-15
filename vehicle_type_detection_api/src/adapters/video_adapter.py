"""
Video Processing Adapter
OpenCV-based implementation of VideoProcessingPort
"""

from pathlib import Path

import cv2
import numpy as np

from ..adapters.ports import VideoProcessingPort
from ..core.logger import setup_logger

logger = setup_logger(__name__)


class OpenCVVideoAdapter(VideoProcessingPort):
    """
    OpenCV-based video processing adapter
    Implements VideoProcessingPort for video frame extraction and annotation
    """

    def __init__(self):
        """Initialize the video adapter"""
        self._supported_formats = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]

    def extract_frames(self, video_path: str, interval_seconds: float) -> list[np.ndarray]:
        """
        Extract frames from video at specified interval

        Args:
            video_path: Path to video file
            interval_seconds: Time interval between frames

        Returns:
            List of frame images as numpy arrays
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default fallback

        frame_interval = int(fps * interval_seconds)
        frames = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frames.append(frame)

                frame_count += 1

                # Safety limit
                if frame_count > 10000:
                    break
        finally:
            cap.release()

        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec = int(cap.get(cv2.CAP_PROP_FOURCC))

            duration_seconds = frame_count / fps if fps > 0 else 0

            return {
                "duration_seconds": duration_seconds,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "codec": self._fourcc_to_str(codec),
                "total_frames": frame_count,
            }
        finally:
            cap.release()

    def create_annotated_video(
        self, original_path: str, output_path: str, frame_detections: list[dict], fps: float = 30.0
    ) -> None:
        """
        Create annotated video with bounding boxes

        Args:
            original_path: Path to original video file
            output_path: Path to save annotated video
            frame_detections: List of detection results for each frame
        """
        if not Path(original_path).exists():
            raise FileNotFoundError(f"Video file not found: {original_path}")

        cap = cv2.VideoCapture(original_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {original_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, original_fps or 30.0, (width, height))

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Annotate frame if detections exist for this frame
                if frame_idx < len(frame_detections):
                    frame = self._annotate_frame(frame, frame_detections[frame_idx])

                out.write(frame)
                frame_idx += 1

            out.release()
            logger.info(f"Created annotated video: {output_path}")
        finally:
            cap.release()

    def get_frame_timestamp(self, frame_number: int, fps: float) -> float:
        """
        Calculate timestamp for a given frame

        Args:
            frame_number: Frame index (0-based)
            fps: Frames per second

        Returns:
            Timestamp in seconds
        """
        return frame_number / fps if fps > 0 else 0

    def _annotate_frame(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """Annotate frame with bounding boxes"""
        if "detections" not in detections:
            return frame

        annotated = frame.copy()
        color_map = {
            "Car": (0, 255, 0),
            "Motorcycle": (255, 0, 0),
            "Truck": (0, 0, 255),
            "Bus": (255, 255, 0),
            "Bicycle": (0, 255, 255),
        }

        for det in detections["detections"]:
            class_name = det.get("class_name", "Vehicle")
            bbox = det.get("bbox", {})
            confidence = det.get("confidence", 0)

            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 0))
            y2 = int(bbox.get("y2", 0))

            color = color_map.get(class_name, (0, 255, 0))

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return annotated

    @staticmethod
    def _fourcc_to_str(fourcc: int) -> str:
        """Convert fourcc integer to string"""
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    def is_supported_format(self, filename: str) -> bool:
        """Check if video format is supported"""
        return Path(filename).suffix.lower() in self._supported_formats
