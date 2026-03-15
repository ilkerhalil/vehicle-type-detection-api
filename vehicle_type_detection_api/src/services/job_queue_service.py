"""
Job Queue Service
Manages job queue operations using JobStoragePort
"""

from typing import Any

from ..adapters.job_storage_adapter import SQLiteJobStorageAdapter
from ..core.config import get_settings
from ..core.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


class JobQueueService:
    """
    Service for managing job queue operations
    Provides high-level interface for job management
    """

    def __init__(self, storage: SQLiteJobStorageAdapter):
        """
        Initialize job queue service

        Args:
            storage: Job storage adapter
        """
        self._storage = storage

    async def submit_batch_job(
        self,
        engine: str,
        data: dict[str, Any],
        webhook_url: str | None = None,
    ) -> str:
        """
        Submit a batch processing job

        Args:
            engine: Detection engine (pytorch or openvino)
            data: Job data (images, parameters)
            webhook_url: Optional webhook URL for notifications

        Returns:
            Job ID
        """
        job_data = {
            "job_type": "batch",
            "engine": engine,
            "data": data,
            "webhook_url": webhook_url,
            "progress_total": len(data.get("images", [])),
            "status": "queued",
        }

        job_id = await self._storage.create_job(job_data)
        logger.info(f"Submitted batch job: {job_id}")
        return job_id

    async def submit_video_job(
        self,
        engine: str,
        video_path: str,
        webhook_url: str | None = None,
    ) -> str:
        """
        Submit a video processing job

        Args:
            engine: Detection engine
            video_path: Path to video file
            webhook_url: Optional webhook URL

        Returns:
            Job ID
        """
        job_data = {
            "job_type": "video",
            "engine": engine,
            "data": {"video_path": video_path},
            "webhook_url": webhook_url,
            "progress_total": 1,
            "status": "queued",
        }

        job_id = await self._storage.create_job(job_data)
        logger.info(f"Submitted video job: {job_id}")
        return job_id

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """
        Get job status

        Args:
            job_id: Job ID

        Returns:
            Job data or None
        """
        return await self._storage.get_job(job_id)

    async def update_job_progress(self, job_id: str, progress: int, status: str = "processing") -> bool:
        """
        Update job progress

        Args:
            job_id: Job ID
            progress: Current progress value
            status: New status

        Returns:
            True if updated
        """
        return await self._storage.update_job(
            job_id,
            {"progress_current": progress, "status": status},
        )

    async def complete_job(self, job_id: str, result: dict[str, Any], status: str = "completed") -> bool:
        """
        Mark job as completed

        Args:
            job_id: Job ID
            result: Job result data
            status: Final status (completed or failed)

        Returns:
            True if updated
        """
        return await self._storage.update_job(
            job_id,
            {
                "status": status,
                "result": result,
                "progress_current": result.get("progress_total", 0),
            },
        )

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job

        Args:
            job_id: Job ID

        Returns:
            True if cancelled
        """
        return await self._storage.update_job(job_id, {"status": "cancelled"})

    async def list_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        List jobs with filtering

        Args:
            status: Filter by status
            job_type: Filter by job type
            limit: Max results
            offset: Results to skip

        Returns:
            List of jobs
        """
        return await self._storage.list_jobs(status=status, job_type=job_type, limit=limit, offset=offset)

    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a job

        Args:
            job_id: Job ID

        Returns:
            True if deleted
        """
        return await self._storage.delete_job(job_id)

    async def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        Clean up old jobs

        Args:
            days: Number of days to keep

        Returns:
            Number of deleted jobs
        """
        return await self._storage.cleanup_old_jobs(days)


# Singleton instance
_job_queue_service: JobQueueService | None = None


def get_job_queue_service() -> JobQueueService:
    """Get or create job queue service singleton"""
    global _job_queue_service
    if _job_queue_service is None:
        from ..core.config import get_settings

        settings = get_settings()
        db_path = settings.JOB_QUEUE_SQLITE_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = SQLiteJobStorageAdapter(db_path)
        _job_queue_service = JobQueueService(storage)
    return _job_queue_service
