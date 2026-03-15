"""
Batch Processing Router for Vehicle Detection API
Provides sync and async batch processing endpoints
"""

import base64
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..adapters.dependencies import (
    OPENVINO_AVAILABLE,
    TORCH_AVAILABLE,
    get_image_processing_adapter,
    get_openvino_detection_adapter,
    get_torch_yolo_detection_adapter,
)
from ..adapters.job_storage_adapter import SQLiteJobStorageAdapter
from ..core.config import get_settings
from ..core.logger import setup_logger
from ..services.batch_service import BatchProcessingService

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Batch Processing"])

settings = get_settings()

# =============================================================================
# Pydantic Models
# =============================================================================


class ImageInput(BaseModel):
    """Single image input for batch processing"""

    filename: str
    data: str  # base64 encoded


class BatchDetectRequest(BaseModel):
    """Synchronous batch detection request"""

    engine: Literal["pytorch", "openvino"]
    images: List[ImageInput]
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_concurrent: int = Field(default=3, ge=1, le=5)

    @field_validator("images")
    @classmethod
    def validate_batch_size(cls, images: List[ImageInput]) -> List[ImageInput]:
        """Validate batch size doesn't exceed sync limit"""
        max_images = settings.BATCH_SYNC_MAX_IMAGES
        if len(images) > max_images:
            raise ValueError(f"Batch size exceeds maximum of {max_images} images for sync processing")
        return images


class AsyncBatchRequest(BaseModel):
    """Asynchronous batch detection request"""

    engine: Literal["pytorch", "openvino"]
    images: List[ImageInput]
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    webhook_url: Optional[str] = None

    @field_validator("images")
    @classmethod
    def validate_batch_size(cls, images: List[ImageInput]) -> List[ImageInput]:
        """Validate batch size doesn't exceed async limit"""
        max_images = settings.BATCH_ASYNC_MAX_IMAGES
        if len(images) > max_images:
            raise ValueError(f"Batch size exceeds maximum of {max_images} images for async processing")
        return images


class BoundingBox(BaseModel):
    """Bounding box coordinates"""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Vehicle detection result"""

    class_name: str
    original_class: str
    confidence: float
    bbox: BoundingBox


class ImageResult(BaseModel):
    """Result for a single image in batch"""

    filename: str
    status: Literal["success", "error"]
    detections: Optional[List[Detection]] = None
    processing_time_ms: Optional[int] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


class BatchSummary(BaseModel):
    """Summary of batch processing results"""

    successful: int
    failed: int
    total_detections: int
    average_processing_time_ms: float


class BatchDetectResponse(BaseModel):
    """Synchronous batch detection response"""

    batch_id: str
    status: Literal["completed", "partial", "failed"]
    engine: str
    total_images: int
    processing_time_ms: int
    results: List[ImageResult]
    summary: BatchSummary


class AsyncBatchResponse(BaseModel):
    """Asynchronous batch job creation response"""

    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    engine: str
    total_images: int
    position_in_queue: Optional[int] = None
    estimated_completion: Optional[str] = None
    results_url: str
    created_at: str


class JobProgress(BaseModel):
    """Job progress information"""

    processed: int
    total: int
    percentage: int
    current_file: Optional[str] = None


class JobTimestamps(BaseModel):
    """Job timing information"""

    created: str
    started: Optional[str] = None
    completed: Optional[str] = None
    estimated_completion: Optional[str] = None


class JobResponse(BaseModel):
    """Job status and results response"""

    job_id: str
    status: Literal["queued", "processing", "completed", "failed", "cancelled"]
    engine: str
    total_images: int
    progress: Optional[JobProgress] = None
    timestamps: JobTimestamps
    processing_time_seconds: Optional[int] = None
    results: Optional[BatchDetectResponse] = None
    error: Optional[Dict[str, Any]] = None


class JobListResponse(BaseModel):
    """List of jobs response"""

    jobs: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    """Error response model"""

    error: Dict[str, Any]


# =============================================================================
# Helper Functions
# =============================================================================


def get_detection_adapter(engine: str) -> Any:
    """
    Get the appropriate detection adapter based on engine.

    Args:
        engine: Detection engine ("pytorch" or "openvino")

    Returns:
        Detection adapter instance

    Raises:
        HTTPException: If adapter is not available
    """
    if engine == "pytorch":
        if not TORCH_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "ENGINE_NOT_AVAILABLE",
                        "message": "PyTorch adapter is not available",
                        "details": {"engine": "pytorch"},
                    }
                },
            )
        try:
            return get_torch_yolo_detection_adapter()
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch adapter: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "ENGINE_NOT_AVAILABLE",
                        "message": f"Failed to initialize PyTorch adapter: {str(e)}",
                        "details": {"engine": "pytorch"},
                    }
                },
            )
    elif engine == "openvino":
        if not OPENVINO_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "ENGINE_NOT_AVAILABLE",
                        "message": "OpenVINO adapter is not available",
                        "details": {"engine": "openvino"},
                    }
                },
            )
        try:
            return get_openvino_detection_adapter()
        except Exception as e:
            logger.error(f"Failed to initialize OpenVINO adapter: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "ENGINE_NOT_AVAILABLE",
                        "message": f"Failed to initialize OpenVINO adapter: {str(e)}",
                        "details": {"engine": "openvino"},
                    }
                },
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_ENGINE",
                    "message": f"Invalid engine: {engine}",
                    "details": {"valid_engines": ["pytorch", "openvino"]},
                }
            },
        )


_job_storage: Optional[SQLiteJobStorageAdapter] = None


def get_job_storage() -> SQLiteJobStorageAdapter:
    """
    Get or create singleton job storage adapter.

    Returns:
        SQLiteJobStorageAdapter instance
    """
    global _job_storage
    if _job_storage is None:
        db_path = settings.JOB_QUEUE_SQLITE_PATH
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _job_storage = SQLiteJobStorageAdapter(db_path)
        logger.info(f"Initialized job storage at {db_path}")
    return _job_storage


def decode_base64_image(data: str, filename: str) -> tuple[bytes, Optional[str]]:
    """
    Decode base64 image data.

    Args:
        data: Base64 encoded image data
        filename: Filename for error reporting

    Returns:
        Tuple of (image_bytes, error_message)
    """
    try:
        # Remove data URL prefix if present
        if "," in data:
            data = data.split(",")[1]
        image_bytes = base64.b64decode(data)
        return image_bytes, None
    except Exception as e:
        error_msg = f"Invalid base64 data for {filename}: {str(e)}"
        logger.warning(error_msg)
        return b"", error_msg


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/batch/detect",
    response_model=BatchDetectResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Engine not available"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Synchronous batch detection",
    description="Process up to 10 images synchronously with parallel processing",
)
async def batch_detect_sync(request: BatchDetectRequest) -> Dict[str, Any]:
    """
    Process a batch of images synchronously.

    - **engine**: Detection engine (pytorch or openvino)
    - **images**: List of images (max 10) with filename and base64 data
    - **confidence_threshold**: Minimum confidence for detections (0.0-1.0)
    - **max_concurrent**: Maximum parallel processing threads (1-5)

    Returns complete results for all images.
    """
    start_time = time.time()
    logger.info(f"Received sync batch request: {len(request.images)} images, engine={request.engine}")

    # Validate batch size
    if len(request.images) > settings.BATCH_SYNC_MAX_IMAGES:
        logger.warning(f"Batch size {len(request.images)} exceeds limit {settings.BATCH_SYNC_MAX_IMAGES}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "BATCH_SIZE_EXCEEDED",
                    "message": f"Batch size of {len(request.images)} exceeds maximum of {settings.BATCH_SYNC_MAX_IMAGES} for sync processing",
                    "details": {
                        "provided": len(request.images),
                        "maximum": settings.BATCH_SYNC_MAX_IMAGES,
                        "suggestion": "Use async batch endpoint for large batches",
                    },
                }
            },
        )

    # Get detection adapter
    detection_adapter = get_detection_adapter(request.engine)

    # Get image adapter
    image_adapter = get_image_processing_adapter()

    # Create batch service
    batch_service = BatchProcessingService(
        detection_adapter=detection_adapter,
        image_adapter=image_adapter,
    )

    # Check if service is ready
    if not batch_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "code": "ENGINE_NOT_AVAILABLE",
                    "message": f"{request.engine} adapter is not ready",
                    "details": {"engine": request.engine},
                }
            },
        )

    # Decode and validate images
    images_data = []
    for img in request.images:
        image_bytes, error = decode_base64_image(img.data, img.filename)
        if error:
            # Return early with error for invalid image
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "INVALID_IMAGE_FORMAT",
                        "message": error,
                        "details": {"filename": img.filename},
                    }
                },
            )
        images_data.append({"filename": img.filename, "data": image_bytes})

    try:
        # Process batch
        result = batch_service.process_sync_batch(
            images=images_data,
            engine=request.engine,
            confidence_threshold=request.confidence_threshold,
            max_concurrent=request.max_concurrent,
        )

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(
            f"Sync batch completed: {result['summary']['successful']}/{len(request.images)} successful, "
            f"{result['summary']['total_detections']} detections, {processing_time}ms total"
        )

        return result

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "PROCESSING_ERROR",
                    "message": f"Batch processing failed: {str(e)}",
                    "details": {},
                }
            },
        )


@router.post(
    "/jobs/batch",
    response_model=AsyncBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Engine not available"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Asynchronous batch detection",
    description="Submit up to 100 images for async processing via job queue",
)
async def batch_detect_async(request: AsyncBatchRequest) -> Dict[str, Any]:
    """
    Submit a batch of images for asynchronous processing.

    - **engine**: Detection engine (pytorch or openvino)
    - **images**: List of images (max 100) with filename and base64 data
    - **confidence_threshold**: Minimum confidence for detections (0.0-1.0)
    - **webhook_url**: Optional webhook URL for completion notification

    Returns job ID for tracking progress.
    """
    start_time = time.time()
    logger.info(f"Received async batch request: {len(request.images)} images, engine={request.engine}")

    # Validate batch size
    if len(request.images) > settings.BATCH_ASYNC_MAX_IMAGES:
        logger.warning(f"Batch size {len(request.images)} exceeds limit {settings.BATCH_ASYNC_MAX_IMAGES}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "BATCH_ASYNC_SIZE_EXCEEDED",
                    "message": f"Batch size of {len(request.images)} exceeds maximum of {settings.BATCH_ASYNC_MAX_IMAGES} for async processing",
                    "details": {
                        "provided": len(request.images),
                        "maximum": settings.BATCH_ASYNC_MAX_IMAGES,
                    },
                }
            },
        )

    # Validate engine availability
    if request.engine == "pytorch" and not TORCH_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "code": "ENGINE_NOT_AVAILABLE",
                    "message": "PyTorch adapter is not available",
                    "details": {"engine": "pytorch"},
                }
            },
        )
    elif request.engine == "openvino" and not OPENVINO_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "code": "ENGINE_NOT_AVAILABLE",
                    "message": "OpenVINO adapter is not available",
                    "details": {"engine": "openvino"},
                }
            },
        )

    # Validate images (decode check)
    for img in request.images:
        _, error = decode_base64_image(img.data, img.filename)
        if error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "INVALID_IMAGE_FORMAT",
                        "message": error,
                        "details": {"filename": img.filename},
                    }
                },
            )

    try:
        # Create job in storage
        job_storage = get_job_storage()
        job_data = {
            "job_type": "batch",
            "engine": request.engine,
            "data": {
                "images": [{"filename": img.filename, "data": img.data} for img in request.images],
                "confidence_threshold": request.confidence_threshold,
                "total_images": len(request.images),
            },
            "webhook_url": request.webhook_url,
            "progress_total": len(request.images),
        }
        job_id = await job_storage.create_job(job_data)

        # Get queue position
        pending_jobs = await job_storage.list_jobs(status="queued")
        position = len(pending_jobs)

        # Estimate completion (rough estimate: 500ms per image)
        estimated_seconds = len(request.images) * 0.5
        estimated_completion = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + estimated_seconds))

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Async batch job created: {job_id}, position={position}, time={processing_time}ms")

        return {
            "job_id": job_id,
            "status": "queued",
            "engine": request.engine,
            "total_images": len(request.images),
            "position_in_queue": position,
            "estimated_completion": estimated_completion,
            "results_url": f"/api/v1/jobs/{job_id}",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create async batch job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "JOB_CREATION_FAILED",
                    "message": f"Failed to create job: {str(e)}",
                    "details": {},
                }
            },
        )


@router.get(
    "/jobs",
    response_model=JobListResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="List all jobs",
    description="List all batch and video processing jobs with optional filtering",
)
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    List all processing jobs.

    - **status**: Filter by status (queued, processing, completed, failed, cancelled)
    - **limit**: Maximum number of results (default 100)
    - **offset**: Number of results to skip (default 0)

    Returns list of jobs with their status.
    """
    try:
        job_storage = get_job_storage()
        jobs = await job_storage.list_jobs(status=status, limit=limit, offset=offset)

        # Convert job data for response
        job_list = []
        for job in jobs:
            job_list.append(
                {
                    "job_id": job["job_id"],
                    "status": job["status"],
                    "engine": job["engine"],
                    "job_type": job["job_type"],
                    "created_at": job["created_at"],
                    "updated_at": job["updated_at"],
                    "progress": job.get("progress"),
                }
            )

        return {
            "jobs": job_list,
            "total": len(job_list),
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "LIST_JOBS_FAILED",
                    "message": f"Failed to list jobs: {str(e)}",
                    "details": {},
                }
            },
        )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Get job status",
    description="Get detailed status and results for a specific job",
)
async def get_job(job_id: str) -> Dict[str, Any]:
    """
    Get job status and results.

    - **job_id**: Unique job identifier

    Returns job details including progress and results if completed.
    """
    try:
        job_storage = get_job_storage()
        job = await job_storage.get_job(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "JOB_NOT_FOUND",
                        "message": f"Job {job_id} not found",
                        "details": {"job_id": job_id},
                    }
                },
            )

        # Calculate percentage
        progress_data = job.get("progress", {})
        current = progress_data.get("current", 0)
        total = progress_data.get("total", 0)
        percentage = int((current / total) * 100) if total > 0 else 0

        # Build response
        response: Dict[str, Any] = {
            "job_id": job["job_id"],
            "status": job["status"],
            "engine": job["engine"],
            "total_images": job["data"].get("total_images", 0) if job.get("data") else 0,
            "progress": (
                {
                    "processed": current,
                    "total": total,
                    "percentage": percentage,
                }
                if total > 0
                else None
            ),
            "timestamps": {
                "created": job["created_at"],
                "started": job.get("started_at"),
                "completed": job.get("completed_at"),
            },
        }

        # Add results if completed
        if job["status"] == "completed" and job.get("results"):
            response["results"] = job["results"]
            if job.get("started_at") and job.get("completed_at"):
                try:
                    started = time.mktime(time.strptime(job["started_at"], "%Y-%m-%d %H:%M:%S"))
                    completed = time.mktime(time.strptime(job["completed_at"], "%Y-%m-%d %H:%M:%S"))
                    response["processing_time_seconds"] = int(completed - started)
                except (ValueError, TypeError):
                    pass

        # Add error if failed
        if job["status"] == "failed" and job.get("error"):
            response["error"] = {
                "code": "JOB_FAILED",
                "message": job["error"],
                "timestamp": job.get("updated_at"),
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "GET_JOB_FAILED",
                    "message": f"Failed to get job: {str(e)}",
                    "details": {"job_id": job_id},
                }
            },
        )


@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Delete job",
    description="Delete a job and its associated data",
)
async def delete_job(job_id: str) -> None:
    """
    Delete a processing job.

    - **job_id**: Unique job identifier to delete

    Returns no content on success.
    """
    try:
        job_storage = get_job_storage()

        # Check if job exists
        job = await job_storage.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "JOB_NOT_FOUND",
                        "message": f"Job {job_id} not found",
                        "details": {"job_id": job_id},
                    }
                },
            )

        # Delete the job
        deleted = await job_storage.delete_job(job_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "code": "DELETE_FAILED",
                        "message": f"Failed to delete job {job_id}",
                        "details": {"job_id": job_id},
                    }
                },
            )

        logger.info(f"Deleted job {job_id}")
        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "DELETE_FAILED",
                    "message": f"Failed to delete job: {str(e)}",
                    "details": {"job_id": job_id},
                }
            },
        )
