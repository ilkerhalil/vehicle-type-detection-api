"""
Video Processing Router for Vehicle Detection API
Provides endpoints for video-based vehicle detection
"""

import tempfile
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from ..core.config import get_settings
from ..core.logger import setup_logger
from ..services.video_service import get_video_processing_service

logger = setup_logger(__name__)
router = APIRouter(prefix="/api/v1/video", tags=["Video Processing"])

settings = get_settings()


# =============================================================================
# Pydantic Models
# =============================================================================


class VideoProcessRequest(BaseModel):
    """Video processing request"""

    engine: Literal["pytorch", "openvino"] = "pytorch"
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    frame_interval: float = Field(default=1.0, ge=0.1, le=10.0, description="Seconds between frames")


class VideoInfoResponse(BaseModel):
    """Video information response"""

    duration_seconds: float
    fps: float
    resolution: str
    codec: str
    total_frames: int


class VideoProcessResponse(BaseModel):
    """Video processing response"""

    job_id: str
    message: str
    status: str = "processing"


class VideoResultFrame(BaseModel):
    """Single frame detection result"""

    frame_number: int
    timestamp: float
    detections: list[dict[str, Any]]
    detection_count: int


class VideoResultResponse(BaseModel):
    """Video processing result response"""

    video_info: dict[str, Any]
    frame_interval: float
    total_frames_processed: int
    results: list[VideoResultFrame]
    summary: dict[str, Any]


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/health")
async def video_health_check() -> dict[str, str]:
    """
    Health check for video processing service

    Returns:
        Health status
    """
    return {"status": "healthy", "service": "video-processing"}


@router.post("/detect", response_model=VideoProcessResponse)
async def process_video(
    file: UploadFile = File(...),
    engine: Literal["pytorch", "openvino"] = "pytorch",
    confidence_threshold: float = 0.5,
    frame_interval: float = 1.0,
) -> VideoProcessResponse:
    """
    Process video and detect vehicles

    Args:
        file: Video file upload
        engine: Detection engine (pytorch or openvino)
        confidence_threshold: Minimum confidence for detections
        frame_interval: Interval between frames to process

    Returns:
        Job ID for tracking the processing
    """
    # Validate file format
    allowed_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    file_ext = Path(file.filename or "").suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format. Allowed: {allowed_extensions}",
        )

    # Check file size
    max_size = settings.VIDEO_MAX_FILE_SIZE_MB * 1024 * 1024
    contents = await file.read()

    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video file too large. Max size: {settings.VIDEO_MAX_FILE_SIZE_MB}MB",
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Get video processing service
        service = get_video_processing_service(engine=engine)

        # Submit async job (placeholder - would integrate with job queue)
        job_id = service.process_video_async(
            video_path=tmp_path,
            engine=engine,
            confidence_threshold=confidence_threshold,
            frame_interval=frame_interval,
        )

        return VideoProcessResponse(job_id=job_id, message="Video processing started", status="processing")

    except Exception as e:
        logger.error(f"Video processing error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    finally:
        # Cleanup temp file
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/detect/sync", response_model=VideoResultResponse)
async def process_video_sync(
    file: UploadFile = File(...),
    engine: Literal["pytorch", "openvino"] = "pytorch",
    confidence_threshold: float = 0.5,
    frame_interval: float = 1.0,
) -> VideoResultResponse:
    """
    Process video synchronously (for shorter videos)

    Args:
        file: Video file upload
        engine: Detection engine
        confidence_threshold: Minimum confidence
        frame_interval: Seconds between frames

    Returns:
        Detection results
    """
    # Validate file format
    allowed_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    file_ext = Path(file.filename or "").suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format. Allowed: {allowed_extensions}",
        )

    # Save to temp file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        service = get_video_processing_service(engine=engine)
        result = service.process_video(
            video_path=tmp_path,
            engine=engine,
            confidence_threshold=confidence_threshold,
            frame_interval=frame_interval,
        )

        return VideoResultResponse(**result)

    except Exception as e:
        logger.error(f"Video processing error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/info")
async def get_video_info(
    file: UploadFile = File(...),
) -> VideoInfoResponse:
    """
    Get video metadata without processing

    Args:
        file: Video file upload

    Returns:
        Video metadata
    """
    from ..adapters.video_adapter import OpenCVVideoAdapter

    contents = await file.read()
    file_ext = Path(file.filename or "").suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        adapter = OpenCVVideoAdapter()
        info = adapter.get_video_info(tmp_path)
        return VideoInfoResponse(**info)

    except Exception as e:
        logger.error(f"Video info error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/formats")
async def get_supported_formats() -> dict[str, list[str]]:
    """Get supported video formats"""
    service = get_video_processing_service()
    return {"formats": service.get_supported_video_formats()}
