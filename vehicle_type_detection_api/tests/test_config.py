import pytest
from vehicle_type_detection_api.src.core.config import get_settings, Settings

def test_get_settings_returns_settings():
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_batch_settings():
    settings = get_settings()
    assert settings.BATCH_SYNC_MAX_IMAGES == 10
    assert settings.BATCH_ASYNC_MAX_IMAGES == 100
    assert settings.BATCH_SYNC_TIMEOUT_SECONDS == 30


def test_video_settings():
    settings = get_settings()
    assert settings.VIDEO_MAX_DURATION_SECONDS == 600
    assert settings.VIDEO_MAX_FILE_SIZE_MB == 500
    assert settings.VIDEO_FRAME_INTERVAL_DEFAULT == 1.0


def test_job_queue_settings():
    settings = get_settings()
    assert settings.JOB_QUEUE_BACKEND in ["sqlite", "redis"]
    assert settings.JOB_MAX_CONCURRENT == 4


def test_monitoring_settings():
    settings = get_settings()
    assert settings.ENABLE_METRICS is True
    assert settings.METRICS_ENDPOINT == "/metrics"


def test_logging_settings():
    settings = get_settings()
    assert settings.LOG_STRUCTURED_FORMAT in ["json", "text"]
    assert settings.ENABLE_CORRELATION_IDS is True
