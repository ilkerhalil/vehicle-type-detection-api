"""
Job Management Router for Vehicle Detection API
Provides endpoints for job queue management
"""

from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from ..core.config import get_settings
from ..core.logger import setup_logger
from ..services.job_queue_service import get_job_queue_service

logger = setup_logger(__name__)
router = APIRouter(prefix="/api/v1/jobs", tags=["Job Management"])

settings = get_settings()


# =============================================================================
# Pydantic Models
# =============================================================================


class JobCreateRequest(BaseModel):
    """Request to create a new job"""

    job_type: Literal["batch", "video"]
    engine: Literal["pytorch", "openvino"]
    data: dict[str, Any]
    webhook_url: Optional[str] = None


class JobCreateResponse(BaseModel):
    """Job creation response"""

    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Job status response"""

    job_id: str
    status: str
    job_type: str
    engine: str
    progress_current: int
    progress_total: int
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class JobListResponse(BaseModel):
    """Job list response"""

    jobs: list[JobStatusResponse]
    total: int


class JobUpdateRequest(BaseModel):
    """Job update request"""

    status: Optional[str] = None
    progress_current: Optional[int] = None
    result: Optional[dict[str, Any]] = None


class JobCancelResponse(BaseModel):
    """Job cancellation response"""

    job_id: str
    status: str
    message: str


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/health")
async def jobs_health_check() -> dict[str, str]:
    """Health check for job management service"""
    return {"status": "healthy", "service": "job-management"}


@router.post("/", response_model=JobCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_job(request: JobCreateRequest) -> JobCreateResponse:
    """
    Create a new job

    Args:
        request: Job creation request

    Returns:
        Job ID and status
    """
    try:
        service = get_job_queue_service()

        if request.job_type == "batch":
            job_id = await service.submit_batch_job(
                engine=request.engine,
                data=request.data,
                webhook_url=request.webhook_url,
            )
        elif request.job_type == "video":
            job_id = await service.submit_video_job(
                engine=request.engine,
                video_path=request.data.get("video_path", ""),
                webhook_url=request.webhook_url,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job type: {request.job_type}",
            )

        return JobCreateResponse(job_id=job_id, status="queued", message="Job created successfully")

    except Exception as e:
        logger.error(f"Job creation error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    """
    Get job status

    Args:
        job_id: Job ID

    Returns:
        Job status
    """
    try:
        service = get_job_queue_service()
        job = await service.get_job_status(job_id)

        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}")

        return JobStatusResponse(
            job_id=job_id,
            status=job.get("status", "unknown"),
            job_type=job.get("job_type", "unknown"),
            engine=job.get("engine", "unknown"),
            progress_current=job.get("progress_current", 0),
            progress_total=job.get("progress_total", 0),
            result=job.get("result"),
            error=job.get("error"),
            created_at=job.get("created_at"),
            updated_at=job.get("updated_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get job error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/", response_model=JobListResponse)
async def list_jobs(
    status_filter: Optional[str] = Query(None, alias="status"),
    job_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> JobListResponse:
    """
    List jobs with optional filtering

    Args:
        status_filter: Filter by status
        job_type: Filter by job type
        limit: Max results
        offset: Results to skip

    Returns:
        List of jobs
    """
    try:
        service = get_job_queue_service()
        jobs = await service.list_jobs(
            status=status_filter,
            job_type=job_type,
            limit=limit,
            offset=offset,
        )

        job_responses = [
            JobStatusResponse(
                job_id=job.get("job_id", ""),
                status=job.get("status", "unknown"),
                job_type=job.get("job_type", "unknown"),
                engine=job.get("engine", "unknown"),
                progress_current=job.get("progress_current", 0),
                progress_total=job.get("progress_total", 0),
                result=job.get("result"),
                error=job.get("error"),
                created_at=job.get("created_at"),
                updated_at=job.get("updated_at"),
            )
            for job in jobs
        ]

        return JobListResponse(jobs=job_responses, total=len(job_responses))

    except Exception as e:
        logger.error(f"List jobs error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.patch("/{job_id}", response_model=JobStatusResponse)
async def update_job(job_id: str, request: JobUpdateRequest) -> JobStatusResponse:
    """
    Update job

    Args:
        job_id: Job ID
        request: Update data

    Returns:
        Updated job status
    """
    try:
        service = get_job_queue_service()

        # Build update dict
        updates = {}
        if request.status:
            updates["status"] = request.status
        if request.progress_current is not None:
            updates["progress_current"] = request.progress_current
        if request.result:
            updates["result"] = request.result

        if not updates:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No updates provided")

        success = await service._storage.update_job(job_id, updates)

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}")

        # Get updated job
        job = await service.get_job_status(job_id)

        return JobStatusResponse(
            job_id=job_id,
            status=job.get("status", "unknown"),
            job_type=job.get("job_type", "unknown"),
            engine=job.get("engine", "unknown"),
            progress_current=job.get("progress_current", 0),
            progress_total=job.get("progress_total", 0),
            result=job.get("result"),
            error=job.get("error"),
            created_at=job.get("created_at"),
            updated_at=job.get("updated_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update job error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/{job_id}", response_model=JobCancelResponse)
async def delete_job(job_id: str) -> JobCancelResponse:
    """
    Delete a job

    Args:
        job_id: Job ID

    Returns:
        Deletion confirmation
    """
    try:
        service = get_job_queue_service()
        success = await service.delete_job(job_id)

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}")

        return JobCancelResponse(job_id=job_id, status="deleted", message="Job deleted successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete job error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/{job_id}/cancel", response_model=JobCancelResponse)
async def cancel_job(job_id: str) -> JobCancelResponse:
    """
    Cancel a job

    Args:
        job_id: Job ID

    Returns:
        Cancellation confirmation
    """
    try:
        service = get_job_queue_service()
        success = await service.cancel_job(job_id)

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}")

        return JobCancelResponse(job_id=job_id, status="cancelled", message="Job cancelled successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel job error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/cleanup")
async def cleanup_old_jobs(days: int = Query(7, ge=1, le=90)) -> dict[str, Any]:
    """
    Clean up old completed jobs

    Args:
        days: Number of days to keep jobs

    Returns:
        Cleanup result
    """
    try:
        service = get_job_queue_service()
        deleted_count = await service.cleanup_old_jobs(days)

        return {
            "message": f"Cleaned up {deleted_count} old jobs",
            "deleted_count": deleted_count,
            "days_kept": days,
        }

    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
