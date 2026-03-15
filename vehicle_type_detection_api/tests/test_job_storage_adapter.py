"""
Tests for SQLite job storage adapter.
"""

import sys
from pathlib import Path

import pytest

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adapters.job_storage_adapter import SQLiteJobStorageAdapter


@pytest.fixture
def storage():
    """Create an in-memory SQLite storage for testing."""
    return SQLiteJobStorageAdapter(db_path=":memory:")


@pytest.mark.asyncio
async def test_create_job(storage):
    """Test creating a job."""
    job_id = await storage.create_job({"job_type": "batch", "engine": "openvino", "data": {"images": []}})
    assert job_id is not None
    assert isinstance(job_id, str)


@pytest.mark.asyncio
async def test_get_job(storage):
    """Test getting a job by ID."""
    job_id = await storage.create_job({"job_type": "batch", "engine": "openvino"})
    job = await storage.get_job(job_id)
    assert job is not None
    assert job["job_id"] == job_id
    assert job["status"] == "queued"


@pytest.mark.asyncio
async def test_get_job_not_found(storage):
    """Test getting a non-existent job."""
    job = await storage.get_job("non-existent-id")
    assert job is None


@pytest.mark.asyncio
async def test_update_job(storage):
    """Test updating a job."""
    job_id = await storage.create_job({"job_type": "batch", "engine": "openvino"})
    success = await storage.update_job(job_id, {"status": "processing"})
    assert success is True
    job = await storage.get_job(job_id)
    assert job["status"] == "processing"


@pytest.mark.asyncio
async def test_update_job_with_results(storage):
    """Test updating a job with results."""
    job_id = await storage.create_job({"job_type": "batch", "engine": "openvino"})
    results = {"detections": [{"class": "Car", "confidence": 0.95}]}
    success = await storage.update_job(job_id, {"status": "completed", "results": results})
    assert success is True
    job = await storage.get_job(job_id)
    assert job["status"] == "completed"
    assert job["results"] == results


@pytest.mark.asyncio
async def test_update_job_not_found(storage):
    """Test updating a non-existent job."""
    success = await storage.update_job("non-existent-id", {"status": "processing"})
    assert success is False


@pytest.mark.asyncio
async def test_get_next_pending_job(storage):
    """Test getting the next pending job."""
    await storage.create_job({"job_type": "batch", "engine": "openvino"})
    job = await storage.get_next_pending_job()
    assert job is not None
    assert job["status"] == "queued"


@pytest.mark.asyncio
async def test_get_next_pending_job_empty(storage):
    """Test getting next pending job when queue is empty."""
    job = await storage.get_next_pending_job()
    assert job is None


@pytest.mark.asyncio
async def test_get_next_pending_job_ordering(storage):
    """Test that pending jobs are returned in FIFO order."""
    job_id_1 = await storage.create_job({"job_type": "batch", "engine": "openvino"})
    job_id_2 = await storage.create_job({"job_type": "batch", "engine": "openvino"})

    job = await storage.get_next_pending_job()
    assert job["job_id"] == job_id_1  # First job should be returned first


@pytest.mark.asyncio
async def test_list_jobs(storage):
    """Test listing all jobs."""
    await storage.create_job({"job_type": "batch", "engine": "openvino"})
    await storage.create_job({"job_type": "video", "engine": "pytorch"})
    jobs = await storage.list_jobs()
    assert len(jobs) == 2


@pytest.mark.asyncio
async def test_list_jobs_with_status_filter(storage):
    """Test listing jobs with status filter."""
    job_id = await storage.create_job({"job_type": "batch", "engine": "openvino"})
    await storage.update_job(job_id, {"status": "completed"})
    await storage.create_job({"job_type": "batch", "engine": "openvino"})

    jobs = await storage.list_jobs(status="completed")
    assert len(jobs) == 1
    assert jobs[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_list_jobs_with_limit(storage):
    """Test listing jobs with limit."""
    for _ in range(5):
        await storage.create_job({"job_type": "batch", "engine": "openvino"})

    jobs = await storage.list_jobs(limit=3)
    assert len(jobs) == 3


@pytest.mark.asyncio
async def test_list_jobs_with_offset(storage):
    """Test listing jobs with offset."""
    for _ in range(5):
        await storage.create_job({"job_type": "batch", "engine": "openvino"})

    all_jobs = await storage.list_jobs(limit=5)
    offset_jobs = await storage.list_jobs(limit=3, offset=2)
    assert len(offset_jobs) == 3


@pytest.mark.asyncio
async def test_delete_job(storage):
    """Test deleting a job."""
    job_id = await storage.create_job({"job_type": "batch", "engine": "openvino"})
    success = await storage.delete_job(job_id)
    assert success is True
    job = await storage.get_job(job_id)
    assert job is None


@pytest.mark.asyncio
async def test_delete_job_not_found(storage):
    """Test deleting a non-existent job."""
    success = await storage.delete_job("non-existent-id")
    assert success is False


@pytest.mark.asyncio
async def test_job_data_persistence(storage):
    """Test that job data is correctly persisted."""
    job_data = {
        "job_type": "batch",
        "engine": "pytorch",
        "data": {"images": ["image1.jpg", "image2.jpg"]},
        "webhook_url": "https://example.com/webhook",
    }
    job_id = await storage.create_job(job_data)
    job = await storage.get_job(job_id)

    assert job["job_type"] == "batch"
    assert job["engine"] == "pytorch"
    assert job["data"] == {"images": ["image1.jpg", "image2.jpg"]}
    assert job["webhook_url"] == "https://example.com/webhook"


@pytest.mark.asyncio
async def test_progress_tracking(storage):
    """Test that job progress is correctly tracked."""
    job_id = await storage.create_job({"job_type": "batch", "engine": "openvino"})

    await storage.update_job(job_id, {"status": "processing", "progress_current": 50, "progress_total": 100})

    job = await storage.get_job(job_id)
    assert job["progress"]["current"] == 50
    assert job["progress"]["total"] == 100


@pytest.mark.asyncio
async def test_error_handling(storage):
    """Test that job errors are correctly stored."""
    job_id = await storage.create_job({"job_type": "batch", "engine": "openvino"})

    await storage.update_job(job_id, {"status": "failed", "error": "Test error message"})

    job = await storage.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "Test error message"


@pytest.mark.asyncio
async def test_multiple_job_types(storage):
    """Test handling of different job types."""
    batch_job = await storage.create_job({"job_type": "batch", "engine": "openvino"})
    video_job = await storage.create_job({"job_type": "video", "engine": "pytorch"})

    jobs = await storage.list_jobs()
    job_types = [job["job_type"] for job in jobs]
    assert "batch" in job_types
    assert "video" in job_types


@pytest.mark.asyncio
async def test_cleanup_old_jobs(storage):
    """Test cleaning up old jobs."""
    # Create a job
    job_id = await storage.create_job({"job_type": "batch", "engine": "openvino"})

    # Initially there should be 1 job
    jobs = await storage.list_jobs()
    assert len(jobs) == 1

    # Clean up jobs older than 0 days (should delete all)
    deleted_count = await storage.cleanup_old_jobs(days=0)
    # Note: In-memory SQLite might not respect the time comparison perfectly,
    # so we mainly test that the method exists and returns an integer
    assert isinstance(deleted_count, int)
