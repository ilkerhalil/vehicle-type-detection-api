"""
Metrics service for collecting and formatting Prometheus-compatible metrics.
"""

from collections import defaultdict
from typing import Any

from ..core.logger import setup_logger

logger = setup_logger(__name__)


class MetricsService:
    """Service for collecting API metrics."""

    def __init__(self):
        # Request counters: {(engine, status, endpoint): count}
        self._request_counts = defaultdict(int)

        # Latency histograms: {engine: {bucket: count}}
        self._latency_buckets = {"pytorch": defaultdict(int), "openvino": defaultdict(int)}
        self._latency_sum = {"pytorch": 0.0, "openvino": 0.0}
        self._latency_count = {"pytorch": 0, "openvino": 0}

        # Bucket boundaries for latency histogram (seconds)
        self._bucket_boundaries = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")]

        # Active job gauges
        self._active_batch_jobs = {"pytorch": 0, "openvino": 0}
        self._video_queue_size = 0

        # Detection counters by class
        self._detection_counts = defaultdict(lambda: defaultdict(int))

        # Batch job counters
        self._batch_jobs_total = defaultdict(lambda: defaultdict(int))

        # Video processing metrics
        self._video_durations = []

    def increment_request_count(self, engine: str, status: str, endpoint: str = "") -> None:
        """Increment request counter."""
        key = (engine, status, endpoint)
        self._request_counts[key] += 1

    def record_latency(self, engine: str, latency_seconds: float) -> None:
        """Record request latency."""
        if engine not in self._latency_buckets:
            return

        self._latency_sum[engine] += latency_seconds
        self._latency_count[engine] += 1

        # Find appropriate bucket
        for bucket in self._bucket_boundaries:
            if latency_seconds <= bucket:
                self._latency_buckets[engine][bucket] += 1
                break

    def record_detection(self, engine: str, class_name: str) -> None:
        """Record a vehicle detection."""
        self._detection_counts[engine][class_name] += 1

    def record_batch_job(self, engine: str, status: str) -> None:
        """Record batch job completion."""
        self._batch_jobs_total[engine][status] += 1

    def set_active_batch_jobs(self, engine: str, count: int) -> None:
        """Set active batch job count."""
        self._active_batch_jobs[engine] = count

    def set_video_queue_size(self, size: int) -> None:
        """Set video queue size."""
        self._video_queue_size = size

    def record_video_duration(self, duration_seconds: float) -> None:
        """Record video processing duration."""
        self._video_durations.append(duration_seconds)

    def get_request_count(self, engine: str, status: str) -> int:
        """Get request count for engine/status."""
        total = 0
        for (eng, stat, _), count in self._request_counts.items():
            if eng == engine and stat == status:
                total += count
        return total

    def get_latency_bucket_count(self, engine: str, bucket: float) -> int:
        """Get count in latency bucket."""
        return self._latency_buckets[engine].get(bucket, 0)

    def get_prometheus_format(self) -> str:
        """Generate Prometheus-formatted metrics output."""
        lines = []

        # Request counters
        lines.append("# HELP vehicle_detection_requests_total Total detection requests")
        lines.append("# TYPE vehicle_detection_requests_total counter")
        for (engine, status, endpoint), count in self._request_counts.items():
            endpoint_label = f',endpoint="{endpoint}"' if endpoint else ""
            lines.append(
                f'vehicle_detection_requests_total{{engine="{engine}",status="{status}"{endpoint_label}}} {count}'
            )

        # Latency histograms
        for engine in ["pytorch", "openvino"]:
            lines.append(f"# HELP vehicle_detection_latency_seconds Detection latency for {engine}")
            lines.append("# TYPE vehicle_detection_latency_seconds histogram")
            cumulative = 0
            for bucket in self._bucket_boundaries:
                if bucket == float("inf"):
                    bucket_label = "+Inf"
                else:
                    bucket_label = str(bucket)
                cumulative += self._latency_buckets[engine].get(bucket, 0)
                lines.append(
                    f'vehicle_detection_latency_seconds_bucket{{engine="{engine}",le="{bucket_label}"}} {cumulative}'
                )
            lines.append(f'vehicle_detection_latency_seconds_sum{{engine="{engine}"}} {self._latency_sum[engine]}')
            lines.append(f'vehicle_detection_latency_seconds_count{{engine="{engine}"}} {self._latency_count[engine]}')

        # Active batch jobs gauge
        lines.append("# HELP active_batch_jobs Currently running batch jobs")
        lines.append("# TYPE active_batch_jobs gauge")
        for engine, count in self._active_batch_jobs.items():
            lines.append(f'active_batch_jobs{{engine="{engine}"}} {count}')

        # Video queue size gauge
        lines.append("# HELP video_processing_queue_size Pending video jobs")
        lines.append("# TYPE video_processing_queue_size gauge")
        lines.append(f"video_processing_queue_size {self._video_queue_size}")

        # Detection counters by class
        lines.append("# HELP detections_total Total vehicles detected by class")
        lines.append("# TYPE detections_total counter")
        for engine, classes in self._detection_counts.items():
            for class_name, count in classes.items():
                lines.append(f'detections_total{{class_name="{class_name}",engine="{engine}"}} {count}')

        # Batch job counters
        lines.append("# HELP batch_jobs_total Total batch jobs processed")
        lines.append("# TYPE batch_jobs_total counter")
        for engine, statuses in self._batch_jobs_total.items():
            for status, count in statuses.items():
                lines.append(f'batch_jobs_total{{engine="{engine}",status="{status}"}} {count}')

        return "\n".join(lines)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get metrics summary for health checks."""
        total_requests = sum(self._request_counts.values())
        avg_latency = {}
        for engine in ["pytorch", "openvino"]:
            if self._latency_count[engine] > 0:
                avg_latency[engine] = self._latency_sum[engine] / self._latency_count[engine]
            else:
                avg_latency[engine] = 0

        return {
            "requests_last_minute": total_requests,  # Simplified
            "average_latency_seconds": avg_latency,
            "active_batch_jobs": dict(self._active_batch_jobs),
            "video_queue_size": self._video_queue_size,
            "total_detections": {engine: sum(classes.values()) for engine, classes in self._detection_counts.items()},
        }
