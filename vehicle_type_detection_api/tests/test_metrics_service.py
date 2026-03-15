import pytest
from vehicle_type_detection_api.src.services.metrics_service import MetricsService


@pytest.fixture
def metrics_service():
    return MetricsService()


def test_increment_request_count(metrics_service):
    metrics_service.increment_request_count("pytorch", "success")
    metrics_service.increment_request_count("pytorch", "success")
    count = metrics_service.get_request_count("pytorch", "success")
    assert count == 2


def test_record_latency(metrics_service):
    metrics_service.record_latency("openvino", 0.15)
    metrics_service.record_latency("openvino", 0.25)
    # Should be in bucket 0.25
    assert metrics_service.get_latency_bucket_count("openvino", 0.25) == 2


def test_get_prometheus_format(metrics_service):
    metrics_service.increment_request_count("pytorch", "success")
    output = metrics_service.get_prometheus_format()
    assert "vehicle_detection_requests_total" in output


def test_record_detection(metrics_service):
    metrics_service.record_detection("pytorch", "Car")
    metrics_service.record_detection("pytorch", "Car")
    metrics_service.record_detection("pytorch", "Bus")

    output = metrics_service.get_prometheus_format()
    assert 'detections_total{class_name="Car",engine="pytorch"} 2' in output
    assert 'detections_total{class_name="Bus",engine="pytorch"} 1' in output


def test_record_batch_job(metrics_service):
    metrics_service.record_batch_job("openvino", "completed")
    metrics_service.record_batch_job("openvino", "completed")
    metrics_service.record_batch_job("openvino", "failed")

    output = metrics_service.get_prometheus_format()
    assert 'batch_jobs_total{engine="openvino",status="completed"} 2' in output
    assert 'batch_jobs_total{engine="openvino",status="failed"} 1' in output


def test_set_active_batch_jobs(metrics_service):
    metrics_service.set_active_batch_jobs("pytorch", 3)
    metrics_service.set_active_batch_jobs("openvino", 1)

    output = metrics_service.get_prometheus_format()
    assert 'active_batch_jobs{engine="pytorch"} 3' in output
    assert 'active_batch_jobs{engine="openvino"} 1' in output


def test_set_video_queue_size(metrics_service):
    metrics_service.set_video_queue_size(5)

    output = metrics_service.get_prometheus_format()
    assert "video_processing_queue_size 5" in output


def test_get_metrics_summary(metrics_service):
    metrics_service.increment_request_count("pytorch", "success")
    metrics_service.increment_request_count("openvino", "success")
    metrics_service.record_latency("pytorch", 0.1)
    metrics_service.record_detection("pytorch", "Car")
    metrics_service.set_active_batch_jobs("pytorch", 1)
    metrics_service.set_video_queue_size(2)

    summary = metrics_service.get_metrics_summary()

    assert summary["requests_last_minute"] == 2
    assert summary["active_batch_jobs"]["pytorch"] == 1
    assert summary["video_queue_size"] == 2
    assert summary["total_detections"]["pytorch"] == 1


def test_latency_histogram_buckets(metrics_service):
    # Record multiple latencies
    metrics_service.record_latency("pytorch", 0.03)  # Should go to 0.05 bucket
    metrics_service.record_latency("pytorch", 0.08)  # Should go to 0.1 bucket
    metrics_service.record_latency("pytorch", 0.2)  # Should go to 0.25 bucket
    metrics_service.record_latency("pytorch", 0.8)  # Should go to 1.0 bucket
    metrics_service.record_latency("pytorch", 5.0)  # Should go to 5.0 bucket
    metrics_service.record_latency("pytorch", 15.0)  # Should go to +Inf bucket

    output = metrics_service.get_prometheus_format()

    # Check that histogram is present
    assert "# TYPE vehicle_detection_latency_seconds histogram" in output
    assert 'vehicle_detection_latency_seconds_bucket{engine="pytorch",le="0.05"} 1' in output
    assert 'vehicle_detection_latency_seconds_count{engine="pytorch"} 6' in output


def test_prometheus_format_includes_all_metrics(metrics_service):
    # Populate with various metrics
    metrics_service.increment_request_count("pytorch", "success", "/api/v1/detect")
    metrics_service.increment_request_count("openvino", "error", "/api/v1/detect")
    metrics_service.record_latency("pytorch", 0.15)
    metrics_service.record_detection("pytorch", "Truck")
    metrics_service.record_batch_job("pytorch", "completed")
    metrics_service.set_active_batch_jobs("pytorch", 2)
    metrics_service.set_video_queue_size(3)

    output = metrics_service.get_prometheus_format()

    # Verify all metric types are present
    assert "# HELP vehicle_detection_requests_total" in output
    assert "# HELP vehicle_detection_latency_seconds" in output
    assert "# HELP active_batch_jobs" in output
    assert "# HELP video_processing_queue_size" in output
    assert "# HELP detections_total" in output
    assert "# HELP batch_jobs_total" in output

    # Verify some specific values
    assert 'engine="pytorch"' in output
    assert 'engine="openvino"' in output
