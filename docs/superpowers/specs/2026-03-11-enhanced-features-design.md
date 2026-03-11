# Enhanced API Features + Monitoring & Observability Design

**Date:** 2026-03-11
**Project:** Vehicle Type Detection API
**Scope:** Batch Processing, Video Support, Monitoring & Observability

---

## Executive Summary

This design extends the Vehicle Type Detection API with production-ready features:

1. **Batch Processing**: Synchronous (small batches) + Asynchronous (large batches) with job queue
2. **Video Processing**: Frame extraction, annotated video generation, and real-time WebSocket streaming
3. **Monitoring**: Prometheus-compatible metrics endpoint with request tracking
4. **Observability**: Structured JSON logging with correlation IDs

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Application                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Routes Layer                                                           │
│  ├── /api/v1/detect (existing)                                          │
│  ├── /api/v1/batch/detect (sync batch)                                  │
│  ├── /api/v1/jobs (async job management)                                │
│  ├── /api/v1/video/process (video upload)                               │
│  ├── /api/v1/stream (WebSocket real-time)                               │
│  └── /metrics (Prometheus-compatible metrics)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Services Layer                                                         │
│  ├── BatchProcessingService (sync + async)                              │
│  ├── VideoProcessingService (frame extraction, annotation)              │
│  ├── JobQueueService (SQLite/Redis backend)                            │
│  └── MetricsService (request counters, latency tracking)                │
├─────────────────────────────────────────────────────────────────────────┤
│  Adapters Layer                                                         │
│  ├── VideoProcessingPort (interface)                                    │
│  ├── OpenCVVideoAdapter (implementation)                                │
│  └── JobStoragePort (interface)                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Infrastructure                                                         │
│  ├── StructuredLogger (JSON + correlation IDs)                           │
│  ├── MetricsCollector (Prometheus client)                               │
│  └── JobQueue (SQLite for dev, Redis for prod)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Job Queue Backend | SQLite (dev) / Redis (prod) | Zero config for dev, scalable for production |
| Video Processing | Background task queue | Avoids blocking HTTP requests |
| Metrics Format | Prometheus-compatible | Industry standard, works with/without external tools |
| Logging | Structured JSON with correlation IDs | Enables distributed tracing and log aggregation |
| Configuration | Environment variables | 12-factor app compliance |

---

## Feature 1: Batch Processing

### 1.1 Synchronous Batch Endpoint

**Endpoint:** `POST /api/v1/batch/detect`

**Request Body:**
```json
{
  "engine": "pytorch",
  "files": ["base64_image_1", "base64_image_2", "base64_image_3"],
  "confidence_threshold": 0.5,
  "max_concurrent": 3,
  "iou_threshold": 0.45
}
```

**Response (200 OK):**
```json
{
  "batch_id": "uuid-123",
  "total_images": 3,
  "processing_time_ms": 450,
  "results": [
    {
      "filename": "image1.jpg",
      "status": "success",
      "detections": [
        {
          "class_name": "Vehicle",
          "original_class": "Car",
          "confidence": 0.92,
          "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
        }
      ],
      "processing_time_ms": 150
    },
    {
      "filename": "image2.jpg",
      "status": "success",
      "detections": [],
      "processing_time_ms": 145
    },
    {
      "filename": "image3.jpg",
      "status": "failed",
      "error": "Invalid image format",
      "error_code": "INVALID_IMAGE"
    }
  ],
  "summary": {
    "successful": 2,
    "failed": 1,
    "total_detections": 5
  }
}
```

**Limits:**
- Maximum 10 images per batch
- 30-second timeout
- Max 5 concurrent processing threads

**Error Codes:**
- `BATCH_SIZE_EXCEEDED`: More than 10 images
- `INVALID_IMAGE`: Image decoding failed
- `TIMEOUT`: Processing exceeded 30 seconds
- `ENGINE_UNAVAILABLE`: Selected AI engine not available

### 1.2 Asynchronous Batch Endpoint

**Endpoint:** `POST /api/v1/jobs/batch`

**Request Body:**
```json
{
  "engine": "openvino",
  "files": ["base64_img_1", "base64_img_2", "...50 images"],
  "confidence_threshold": 0.5,
  "webhook_url": "https://example.com/webhook",
  "callback_on_completion": true
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "job-uuid-456",
  "status": "queued",
  "created_at": "2026-03-11T10:00:00Z",
  "estimated_completion": "2026-03-11T10:30:00Z",
  "position_in_queue": 3,
  "results_url": "/api/v1/jobs/job-uuid-456",
  "webhook_url": "https://example.com/webhook"
}
```

### 1.3 Job Status Endpoint

**Endpoint:** `GET /api/v1/jobs/{job_id}`

**Response:**
```json
{
  "job_id": "job-uuid-456",
  "status": "processing",
  "progress": {
    "processed": 25,
    "total": 50,
    "percentage": 50
  },
  "created_at": "2026-03-11T10:00:00Z",
  "started_at": "2026-03-11T10:05:00Z",
  "estimated_completion": "2026-03-11T10:30:00Z",
  "completed_at": null,
  "results": null
}
```

**Status Values:**
- `queued`: Waiting to start
- `processing`: Currently running
- `completed`: Finished successfully
- `failed`: Error occurred
- `cancelled`: User cancelled

### 1.4 Job Results Endpoint

**Endpoint:** `GET /api/v1/jobs/{job_id}/results`

**Response (completed job):**
```json
{
  "job_id": "job-uuid-456",
  "status": "completed",
  "summary": {
    "total_images": 50,
    "successful": 48,
    "failed": 2,
    "total_detections": 245,
    "average_processing_time_ms": 120
  },
  "results": [
    {
      "filename": "image1.jpg",
      "status": "success",
      "detections": [...]
    }
  ],
  "failed_files": [
    {
      "filename": "corrupted.jpg",
      "error": "Image decoding failed",
      "error_code": "INVALID_IMAGE"
    }
  ]
}
```

### 1.5 Job List Endpoint

**Endpoint:** `GET /api/v1/jobs`

**Query Parameters:**
- `status`: Filter by status (queued, processing, completed, failed)
- `limit`: Max results (default 20, max 100)
- `offset`: Pagination offset

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job-uuid-456",
      "status": "processing",
      "type": "batch",
      "created_at": "2026-03-11T10:00:00Z",
      "progress": {"percentage": 50}
    }
  ],
  "total": 15,
  "limit": 20,
  "offset": 0
}
```

---

## Feature 2: Video Processing

### 2.1 Video Upload Endpoint

**Endpoint:** `POST /api/v1/video/process`

**Multipart Form Data:**
- `file`: Video file (MP4, AVI, MOV)
- `engine`: "pytorch" or "openvino"
- `output_format`: "json" | "annotated_video" | "both"
- `frame_interval`: Seconds between processed frames (default: 1)
- `confidence_threshold`: Float (0.0-1.0, default: 0.5)
- `enable_tracking`: Boolean (track vehicles across frames)

**Response (202 Accepted):**
```json
{
  "video_id": "vid-uuid-789",
  "status": "processing",
  "original_filename": "traffic.mp4",
  "duration_seconds": 120,
  "frames_to_process": 120,
  "estimated_completion": "2026-03-11T10:45:00Z",
  "results_url": "/api/v1/jobs/vid-uuid-789",
  "webhook_url": null
}
```

### 2.2 Video Processing Results

**Endpoint:** `GET /api/v1/jobs/{video_id}`

**Response (completed):**
```json
{
  "video_id": "vid-uuid-789",
  "status": "completed",
  "input": {
    "filename": "traffic.mp4",
    "duration_seconds": 120,
    "resolution": "1920x1080",
    "fps": 30,
    "total_frames": 3600
  },
  "processing": {
    "frames_processed": 120,
    "frame_interval": 1,
    "processing_time_seconds": 45,
    "engine": "openvino",
    "confidence_threshold": 0.5
  },
  "detections": [
    {
      "timestamp": 0.0,
      "frame_number": 0,
      "detections": [
        {
          "class_name": "Vehicle",
          "original_class": "Car",
          "confidence": 0.92,
          "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
          "track_id": 1
        }
      ]
    }
  ],
  "summary": {
    "total_vehicles_detected": 245,
    "unique_vehicles_tracked": 12,
    "average_confidence": 0.84,
    "processing_fps": 2.67,
    "peak_vehicles_per_frame": 8
  },
  "annotated_video_url": "/api/v1/jobs/vid-uuid-789/annotated_video"
}
```

### 2.3 Annotated Video Download

**Endpoint:** `GET /api/v1/jobs/{video_id}/annotated_video`

**Response:** Video file stream (MP4)

### 2.4 Real-Time WebSocket Streaming

**Endpoint:** `ws://localhost:8000/api/v1/stream`

**Protocol:**

1. Client connects via WebSocket
2. Client sends start message:
```json
{
  "action": "start_stream",
  "source": "camera_1",
  "engine": "openvino",
  "confidence_threshold": 0.5,
  "fps_target": 15
}
```

3. Server acknowledges:
```json
{
  "type": "status",
  "status": "started",
  "stream_id": "stream-uuid-001",
  "fps_target": 15
}
```

4. Server sends continuous detections:
```json
{
  "type": "detection",
  "stream_id": "stream-uuid-001",
  "timestamp": "2026-03-11T10:00:00.123Z",
  "frame_number": 152,
  "latency_ms": 45,
  "detections": [
    {
      "class_name": "Vehicle",
      "original_class": "Car",
      "confidence": 0.91,
      "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
      "track_id": 5
    }
  ],
  "fps": 15
}
```

5. Client stops stream:
```json
{"action": "stop_stream"}
```

### 2.5 Video Processing Pipeline

```
Upload Video → Save Temp → Extract Frames → Parallel Detection
                                              ↓
                    Annotated Video ← Generate Output ← Aggregate Results
                                              ↓
                                        Return Job ID
```

**Supported Formats:** MP4, AVI, MOV, MKV
**Max Duration:** 10 minutes (configurable)
**Max Resolution:** 4K (3840x2160)

---

## Feature 3: Monitoring & Metrics

### 3.1 Prometheus-Compatible Metrics Endpoint

**Endpoint:** `GET /metrics`

**Response Format (text/plain):**
```
# HELP vehicle_detection_requests_total Total detection requests
# TYPE vehicle_detection_requests_total counter
vehicle_detection_requests_total{engine="pytorch",status="success"} 1523
vehicle_detection_requests_total{engine="openvino",status="success"} 2456
vehicle_detection_requests_total{engine="pytorch",status="error"} 23
vehicle_detection_requests_total{engine="openvino",status="error"} 12

# HELP vehicle_detection_latency_seconds Detection latency
# TYPE vehicle_detection_latency_seconds histogram
vehicle_detection_latency_seconds_bucket{engine="pytorch",le="0.1"} 523
vehicle_detection_latency_seconds_bucket{engine="pytorch",le="0.5"} 1489
vehicle_detection_latency_seconds_bucket{engine="pytorch",le="1.0"} 1523
vehicle_detection_latency_seconds_sum{engine="pytorch"} 456.7
vehicle_detection_latency_seconds_count{engine="pytorch"} 1523

vehicle_detection_latency_seconds_bucket{engine="openvino",le="0.1"} 1845
vehicle_detection_latency_seconds_bucket{engine="openvino",le="0.5"} 2456
vehicle_detection_latency_seconds_sum{engine="openvino"} 245.8
vehicle_detection_latency_seconds_count{engine="openvino"} 2456

# HELP active_batch_jobs Currently running batch jobs
# TYPE active_batch_jobs gauge
active_batch_jobs 3

# HELP video_processing_queue_size Pending video jobs
# TYPE video_processing_queue_size gauge
video_processing_queue_size 2

# HELP job_queue_depth Total jobs in queue
# TYPE job_queue_depth gauge
job_queue_depth{status="queued"} 5
job_queue_depth{status="processing"} 3

# HELP requests_in_flight Current requests being processed
# TYPE requests_in_flight gauge
requests_in_flight 12

# HELP batch_processing_duration_seconds Batch job duration
# TYPE batch_processing_duration_seconds histogram
batch_processing_duration_seconds_bucket{le="10"} 45
batch_processing_duration_seconds_bucket{le="60"} 89
batch_processing_duration_seconds_bucket{le="300"} 98

# HELP video_processing_duration_seconds Video job duration
# TYPE video_processing_duration_seconds histogram
video_processing_duration_seconds_bucket{le="60"} 23
video_processing_duration_seconds_bucket{le="300"} 45
video_processing_duration_seconds_bucket{le="600"} 48
```

### 3.2 Enhanced Health Check

**Endpoint:** `GET /api/v1/health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.1.0",
  "timestamp": "2026-03-11T10:00:00Z",
  "uptime_seconds": 86400,
  "engines": ["PyTorch", "OpenVINO"],
  "components": {
    "api": {
      "status": "up",
      "latency_ms": 5
    },
    "pytorch": {
      "status": "up",
      "model_loaded": true,
      "inference_latency_ms": 120,
      "queue_depth": 0
    },
    "openvino": {
      "status": "up",
      "model_loaded": true,
      "inference_latency_ms": 45,
      "queue_depth": 0
    },
    "job_queue": {
      "status": "up",
      "backend": "sqlite",
      "pending_jobs": 3,
      "processing_jobs": 2
    },
    "metrics_collector": {
      "status": "up",
      "metrics_count": 12
    }
  },
  "metrics": {
    "requests_last_minute": 45,
    "average_latency_ms": 85,
    "error_rate": 0.02,
    "active_connections": 8
  }
}
```

### 3.3 System Stats Endpoint

**Endpoint:** `GET /api/v1/stats`

**Response:**
```json
{
  "period": "last_24h",
  "requests": {
    "total": 15234,
    "by_engine": {
      "pytorch": 5234,
      "openvino": 10000
    },
    "by_endpoint": {
      "/api/v1/detect": 12000,
      "/api/v1/batch/detect": 1500,
      "/api/v1/video/process": 734
    },
    "success_rate": 0.985
  },
  "jobs": {
    "total_completed": 1234,
    "total_failed": 23,
    "average_processing_time_seconds": 45.5,
    "current_queue_depth": 8
  },
  "detections": {
    "total_vehicles_detected": 45678,
    "average_per_image": 3.2,
    "by_class": {
      "Car": 23456,
      "Truck": 8901,
      "Motorcycle": 6789,
      "Bus": 4321,
      "Bicycle": 2211
    }
  },
  "system": {
    "uptime_seconds": 86400,
    "memory_usage_mb": 512,
    "cpu_usage_percent": 45,
    "active_threads": 24
  }
}
```

---

## Feature 4: Structured Logging

### 4.1 Configuration

Environment variables:
```bash
LOG_FORMAT=json              # json | text
LOG_LEVEL=INFO               # DEBUG | INFO | WARNING | ERROR
ENABLE_CORRELATION_IDS=true  # boolean
LOG_TIMESTAMP_FORMAT=ISO8601 # ISO8601 | UNIX
```

### 4.2 Log Schema

**Request Log:**
```json
{
  "timestamp": "2026-03-11T10:00:00.123Z",
  "level": "INFO",
  "correlation_id": "abc-123-xyz",
  "request_id": "req-456",
  "trace_id": "trace-789",
  "method": "POST",
  "path": "/api/v1/openvino/detect",
  "duration_ms": 125,
  "status_code": 200,
  "client_ip": "192.168.1.100",
  "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
  "engine": "openvino",
  "detections_count": 3,
  "image_size_bytes": 245760,
  "model_inference_time_ms": 45,
  "total_processing_time_ms": 125
}
```

**Batch Job Log:**
```json
{
  "timestamp": "2026-03-11T10:00:00.123Z",
  "level": "INFO",
  "correlation_id": "batch-123",
  "job_id": "job-uuid-456",
  "job_type": "batch",
  "event": "job_started",
  "total_images": 50,
  "engine": "openvino",
  "estimated_duration_seconds": 120
}
```

**Video Job Log:**
```json
{
  "timestamp": "2026-03-11T10:00:00.123Z",
  "level": "INFO",
  "correlation_id": "video-789",
  "job_id": "vid-uuid-789",
  "job_type": "video",
  "event": "frame_processed",
  "frame_number": 15,
  "timestamp_seconds": 15.0,
  "detections_count": 2,
  "processing_time_ms": 85
}
```

**Error Log:**
```json
{
  "timestamp": "2026-03-11T10:00:01.456Z",
  "level": "ERROR",
  "correlation_id": "abc-123-xyz",
  "request_id": "req-457",
  "method": "POST",
  "path": "/api/v1/batch/detect",
  "error_type": "ValidationError",
  "error_code": "BATCH_SIZE_EXCEEDED",
  "error_message": "Batch size exceeds limit of 10 images",
  "client_ip": "192.168.1.100",
  "stack_trace": "Traceback (most recent call last):\n  File ...",
  "context": {
    "requested_images": 25,
    "max_allowed": 10
  }
}
```

### 4.3 Correlation ID Flow

```
Client Request
    ↓ (generate correlation_id)
FastAPI Middleware (attach to context)
    ↓
Service Layer (propagate in logs)
    ↓
Adapter Layer (propagate in logs)
    ↓
External Calls (webhooks, include in payload)
```

---

## Data Models

### Job Model
```python
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobType(str, Enum):
    BATCH = "batch"
    VIDEO = "video"

class Job(BaseModel):
    job_id: str  # UUID
    job_type: JobType
    status: JobStatus
    engine: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_items: int
    processed_items: int
    failed_items: int
    error_message: Optional[str]
    results: Optional[Dict]
    webhook_url: Optional[str]
    metadata: Dict[str, Any]
```

### VideoJob Model (extends Job)
```python
class VideoJob(Job):
    input_filename: str
    duration_seconds: Optional[float]
    resolution: Optional[str]
    fps: Optional[float]
    frame_interval: float
    frames_processed: int
    total_frames: Optional[int]
    annotated_video_path: Optional[str]
    detections: List[FrameDetection]

class FrameDetection(BaseModel):
    timestamp: float
    frame_number: int
    detections: List[DetectionResult]
```

---

## Error Handling

### HTTP Status Codes

| Status Code | Scenario |
|-------------|----------|
| 200 | Success (sync endpoints) |
| 202 | Accepted (async job created) |
| 400 | Bad request (validation error) |
| 404 | Job not found |
| 413 | Payload too large (batch/video too big) |
| 422 | Unprocessable (invalid image/video) |
| 429 | Rate limited |
| 500 | Server error |
| 503 | Service unavailable (engine not ready) |

### Error Response Format
```json
{
  "error": {
    "code": "BATCH_SIZE_EXCEEDED",
    "message": "Batch size exceeds limit of 10 images",
    "details": {
      "requested": 25,
      "maximum": 10
    },
    "correlation_id": "abc-123-xyz",
    "request_id": "req-456",
    "timestamp": "2026-03-11T10:00:00Z"
  }
}
```

---

## Performance Requirements

| Metric | Target |
|--------|--------|
| Single image detection | < 200ms (p95) |
| Sync batch (10 images) | < 2 seconds |
| Async batch throughput | > 100 images/minute |
| Video processing | > 2 FPS |
| WebSocket streaming | > 15 FPS |
| Metrics endpoint | < 10ms |
| Health check | < 50ms |

---

## Security Considerations

1. **Input Validation**: Strict file type validation, size limits
2. **Rate Limiting**: Per-IP and global rate limits
3. **Webhook Security**: Signature verification for webhooks
4. **CORS**: Configurable allowed origins
5. **File Storage**: Temp files with restricted permissions
6. **Resource Limits**: Memory, CPU, and timeout constraints

---

## Configuration

### Environment Variables

```bash
# API
API_TITLE="Vehicle Type Detection API"
API_VERSION="1.1.0"
HOST="0.0.0.0"
PORT=8000

# Logging
LOG_FORMAT=json
LOG_LEVEL=INFO
ENABLE_CORRELATION_IDS=true

# Metrics
ENABLE_METRICS=true
METRICS_ENDPOINT=/metrics

# Batch Processing
MAX_SYNC_BATCH_SIZE=10
MAX_ASYNC_BATCH_SIZE=100
BATCH_TIMEOUT_SECONDS=30

# Video Processing
MAX_VIDEO_DURATION_SECONDS=600
MAX_VIDEO_RESOLUTION="3840x2160"
VIDEO_FRAME_INTERVAL=1.0

# Job Queue
JOB_QUEUE_BACKEND=sqlite
REDIS_URL=redis://localhost:6379/0
MAX_CONCURRENT_JOBS=4

# WebSocket
WS_MAX_CONNECTIONS=100
WS_HEARTBEAT_INTERVAL=30

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST=10
```

---

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/health | Health check with component status |
| GET | /api/v1/stats | System statistics |
| GET | /metrics | Prometheus metrics |
| POST | /api/v1/batch/detect | Synchronous batch detection |
| POST | /api/v1/jobs/batch | Create async batch job |
| GET | /api/v1/jobs | List jobs |
| GET | /api/v1/jobs/{job_id} | Get job status |
| GET | /api/v1/jobs/{job_id}/results | Get job results |
| DELETE | /api/v1/jobs/{job_id} | Cancel job |
| POST | /api/v1/video/process | Upload video for processing |
| GET | /api/v1/jobs/{video_id}/annotated_video | Download annotated video |
| WS | /api/v1/stream | Real-time streaming |

---

## Testing Strategy

1. **Unit Tests**: All new services and adapters
2. **Integration Tests**: End-to-end batch and video flows
3. **Load Tests**: Batch processing under concurrent load
4. **WebSocket Tests**: Connection, message flow, error handling
5. **Metrics Tests**: Counter accuracy, latency buckets
6. **Logging Tests**: Correlation ID propagation, JSON format

---

## Deployment Notes

1. **SQLite (Development)**: Zero configuration, file-based
2. **Redis (Production)**: Requires Redis server, set `JOB_QUEUE_BACKEND=redis`
3. **Docker**: Include Redis service in docker-compose for production
4. **Monitoring**: Export Prometheus metrics to monitoring stack
5. **Logging**: Configure log aggregation (ELK, Fluentd, etc.)

---

## Future Enhancements (Out of Scope)

- Multi-GPU support for batch processing
- Model versioning and A/B testing
- Advanced vehicle tracking (DeepSORT)
- GPU-accelerated video encoding
- Distributed job queue (Celery, RQ)
- Authentication and API key management
- Usage quotas and billing integration
