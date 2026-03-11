# Enhanced API Features and Monitoring Design

> **Design Date:** 2026-03-11
> **Project:** Vehicle Type Detection API
> **Scope:** Batch Processing, Video Support, Monitoring & Observability

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Batch Processing Design](#3-batch-processing-design)
4. [Video Processing Design](#4-video-processing-design)
5. [Monitoring & Observability Design](#5-monitoring--observability-design)
6. [API Specifications](#6-api-specifications)
7. [Data Models](#7-data-models)
8. [Configuration](#8-configuration)
9. [Error Handling](#9-error-handling)
10. [Testing Strategy](#10-testing-strategy)

---

## 1. Executive Summary

### Goals

Enhance the Vehicle Type Detection API with production-ready features for handling batch workloads, video processing, and comprehensive observability.

### Scope

| Feature | Description | Priority |
|---------|-------------|----------|
| Sync Batch | Process up to 10 images synchronously | P1 |
| Async Batch | Process up to 100 images via job queue | P1 |
| Video Processing | Extract frames, detect vehicles, annotated output | P1 |
| Real-time Streaming | WebSocket for continuous video streams | P2 |
| Metrics | Prometheus-compatible `/metrics` endpoint | P1 |
| Structured Logging | JSON logs with correlation IDs | P1 |

### Out of Scope

- Model training/retraining
- User authentication (covered separately)
- Database persistence of results
- Multi-model ensemble

---

## 2. Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Application                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Routes Layer                                                           │
│  ├── /api/v1/detect (existing)                                          │
│  ├── /api/v1/batch/detect (sync batch)                                  │
│  ├── /api/v1/jobs (async job management)                                  │
│  ├── /api/v1/video/process (video upload)                                 │
│  ├── /api/v1/stream (WebSocket real-time)                                 │
│  └── /metrics (Prometheus-compatible metrics)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Services Layer                                                         │
│  ├── BatchProcessingService (sync + async)                              │
│  ├── VideoProcessingService (frame extraction, annotation)              │
│  ├── JobQueueService (SQLite/Redis backend)                             │
│  └── MetricsService (request counters, latency tracking)                │
├─────────────────────────────────────────────────────────────────────────┤
│  Adapters Layer                                                         │
│  ├── VideoProcessingPort (interface)                                    │
│  ├── OpenCVVideoAdapter (implementation)                                  │
│  └── JobStoragePort (interface)                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Infrastructure                                                         │
│  ├── StructuredLogger (JSON + correlation IDs)                            │
│  ├── MetricsCollector (Prometheus client)                                 │
│  └── JobQueue (SQLite for dev, Redis for prod)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **Job Queue Backend: SQLite for dev, Redis for prod** | Zero config for development, scalable for production |
| **Video Processing: Background task queue** | Avoids request blocking for long-running video processing |
| **Metrics: Prometheus-compatible format** | Industry standard, works with any monitoring stack |
| **Logging: Structured JSON with correlation IDs** | Enables distributed tracing and log aggregation |
| **Existing adapters unchanged** | Hexagonal architecture allows extension without modification |

### File Structure

```
vehicle_type_detection_api/src/
├── core/
│   ├── config.py                    # Extended with new settings
│   ├── logger.py                    # Enhanced structured logging
│   └── correlation.py               # Correlation ID context manager
├── adapters/
│   ├── ports.py                     # Add VideoProcessingPort
│   ├── video_adapter.py             # OpenCV video processing
│   └── job_storage_adapter.py       # Job persistence interface
├── services/
│   ├── batch_service.py             # Batch processing logic
│   ├── video_service.py             # Video processing orchestration
│   ├── job_queue_service.py         # Job queue management
│   └── metrics_service.py           # Metrics collection
├── routers/
│   ├── batch.py                     # Batch endpoints
│   ├── video.py                     # Video endpoints
│   └── metrics.py                   # Metrics endpoint
├── infrastructure/
│   ├── job_queue.py                 # SQLite/Redis implementations
│   ├── metrics_collector.py         # Prometheus client wrapper
│   └── structured_logger.py         # JSON logging handler
└── middleware/
    ├── correlation_middleware.py    # Inject correlation IDs
    └── metrics_middleware.py        # Track request metrics
```

---

## 3. Batch Processing Design

### 3.1 Sync Batch Endpoint

**Endpoint:** `POST /api/v1/batch/detect`

**Request:**
```json
{
  "engine": "pytorch",
  "images": [
    {"filename": "car1.jpg", "data": "base64_encoded_image_1"},
    {"filename": "car2.jpg", "data": "base64_encoded_image_2"},
    {"filename": "car3.jpg", "data": "base64_encoded_image_3"}
  ],
  "confidence_threshold": 0.5,
  "max_concurrent": 3
}
```

**Response:**
```json
{
  "batch_id": "batch-uuid-123",
  "status": "completed",
  "engine": "pytorch",
  "total_images": 3,
  "processing_time_ms": 450,
  "results": [
    {
      "filename": "car1.jpg",
      "status": "success",
      "detections": [
        {
          "class_name": "Vehicle",
          "original_class": "Car",
          "confidence": 0.92,
          "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 400}
        }
      ],
      "processing_time_ms": 145
    },
    {
      "filename": "car2.jpg",
      "status": "success",
      "detections": [],
      "processing_time_ms": 132
    },
    {
      "filename": "car3.jpg",
      "status": "error",
      "error": "Invalid image format",
      "error_code": "INVALID_IMAGE"
    }
  ],
  "summary": {
    "successful": 2,
    "failed": 1,
    "total_detections": 1,
    "average_processing_time_ms": 138.5
  }
}
```

**Constraints:**
- Maximum 10 images per batch
- 30-second timeout
- Parallel processing with configurable concurrency

### 3.2 Async Batch Endpoint

**Endpoint:** `POST /api/v1/jobs/batch`

**Request:**
```json
{
  "engine": "openvino",
  "images": [
    {"filename": "car1.jpg", "data": "base64_encoded_image_1"},
    {"filename": "car2.jpg", "data": "base64_encoded_image_2"}
    // ... up to 100 images
  ],
  "webhook_url": "https://example.com/webhook"
}
```

**Response:**
```json
{
  "job_id": "job-uuid-456",
  "status": "queued",
  "engine": "openvino",
  "total_images": 50,
  "position_in_queue": 3,
  "estimated_completion": "2026-03-11T10:30:00Z",
  "results_url": "/api/v1/jobs/job-uuid-456",
  "created_at": "2026-03-11T10:00:00Z"
}
```

### 3.3 Job Status Endpoint

**Endpoint:** `GET /api/v1/jobs/{job_id}`

**Response (Processing):**
```json
{
  "job_id": "job-uuid-456",
  "status": "processing",
  "engine": "openvino",
  "total_images": 50,
  "progress": {
    "processed": 25,
    "total": 50,
    "percentage": 50,
    "current_file": "car25.jpg"
  },
  "timestamps": {
    "created": "2026-03-11T10:00:00Z",
    "started": "2026-03-11T10:05:00Z",
    "estimated_completion": "2026-03-11T10:30:00Z"
  }
}
```

**Response (Completed):**
```json
{
  "job_id": "job-uuid-456",
  "status": "completed",
  "engine": "openvino",
  "total_images": 50,
  "progress": {"processed": 50, "total": 50, "percentage": 100},
  "timestamps": {
    "created": "2026-03-11T10:00:00Z",
    "started": "2026-03-11T10:05:00Z",
    "completed": "2026-03-11T10:25:00Z"
  },
  "processing_time_seconds": 1200,
  "results": [
    {
      "filename": "car1.jpg",
      "status": "success",
      "detections": [...],
      "processing_time_ms": 45
    }
    // ... all results
  ],
  "summary": {
    "successful": 48,
    "failed": 2,
    "total_detections": 125,
    "average_confidence": 0.87
  }
}
```

### 3.4 Job Queue Implementation

**Interface: JobStoragePort**
```python
class JobStoragePort(ABC):
    @abstractmethod
    async def create_job(self, job_data: dict) -> str: ...

    @abstractmethod
    async def get_job(self, job_id: str) -> dict | None: ...

    @abstractmethod
    async def update_job(self, job_id: str, updates: dict) -> None: ...

    @abstractmethod
    async def get_next_pending_job(self) -> dict | None: ...

    @abstractmethod
    async def list_jobs(self, status: str | None = None) -> list[dict]: ...
```

**SQLite Implementation:**
- Table: `jobs` with columns: id, status, engine, data, created_at, updated_at, started_at, completed_at, results
- Polling interval: 1 second
- Cleanup: Jobs older than 7 days auto-deleted

**Redis Implementation (Production):**
- Hash: `job:{id}` for job data
- List: `jobs:pending` for queue
- Set: `jobs:processing` for active jobs
- TTL: 24 hours for completed jobs

---

## 4. Video Processing Design

### 4.1 Video Upload Endpoint

**Endpoint:** `POST /api/v1/video/process`

**Request (multipart/form-data):**
```
file: video.mp4 (binary)
engine: "pytorch" | "openvino"
output_format: "json" | "annotated_video" | "both"
frame_interval: 1.0              # seconds between frames
confidence_threshold: 0.5
webhook_url: "https://example.com/webhook" (optional)
```

**Response:**
```json
{
  "video_id": "vid-uuid-789",
  "status": "processing",
  "original_filename": "traffic.mp4",
  "input_metadata": {
    "duration_seconds": 120,
    "fps": 30,
    "resolution": "1920x1080",
    "codec": "h264",
    "total_frames": 3600
  },
  "processing_config": {
    "engine": "openvino",
    "frame_interval": 1.0,
    "frames_to_process": 120,
    "output_format": "both"
  },
  "results_url": "/api/v1/jobs/vid-uuid-789",
  "created_at": "2026-03-11T10:00:00Z"
}
```

### 4.2 Video Results Endpoint

**Endpoint:** `GET /api/v1/jobs/{video_id}`

**Response (Completed):**
```json
{
  "video_id": "vid-uuid-789",
  "status": "completed",
  "input": {
    "filename": "traffic.mp4",
    "duration_seconds": 120,
    "resolution": "1920x1080",
    "fps": 30
  },
  "processing": {
    "engine": "openvino",
    "frame_interval": 1.0,
    "frames_processed": 120,
    "processing_time_seconds": 45,
    "processing_fps": 2.67,
    "confidence_threshold": 0.5
  },
  "frames": [
    {
      "timestamp": 0.0,
      "frame_number": 0,
      "detections": [
        {
          "class_name": "Vehicle",
          "original_class": "Car",
          "confidence": 0.92,
          "bbox": {"x1": 441, "y1": 81, "x2": 558, "y2": 230},
          "track_id": 1
        }
      ]
    },
    {
      "timestamp": 1.0,
      "frame_number": 30,
      "detections": [
        {"class_name": "Vehicle", "original_class": "Car", "confidence": 0.88, "bbox": {...}, "track_id": 1},
        {"class_name": "Vehicle", "original_class": "Bus", "confidence": 0.76, "bbox": {...}, "track_id": 2}
      ]
    }
  ],
  "summary": {
    "total_frames_processed": 120,
    "total_vehicles_detected": 245,
    "unique_vehicles_tracked": 12,
    "average_vehicles_per_frame": 2.04,
    "average_confidence": 0.84,
    "detection_by_class": {
      "Car": 180,
      "Bus": 25,
      "Truck": 30,
      "Motorcycle": 8,
      "Bicycle": 2
    }
  },
  "outputs": {
    "json_url": "/api/v1/jobs/vid-uuid-789/results",
    "annotated_video_url": "/api/v1/jobs/vid-uuid-789/annotated_video"
  },
  "timestamps": {
    "created": "2026-03-11T10:00:00Z",
    "started": "2026-03-11T10:01:00Z",
    "completed": "2026-03-11T10:46:00Z"
  }
}
```

### 4.3 Annotated Video Download

**Endpoint:** `GET /api/v1/jobs/{video_id}/annotated_video`

**Response:** Streaming binary (video/mp4)

### 4.4 WebSocket Real-Time Streaming

**Endpoint:** `ws://localhost:8000/api/v1/stream`

**Protocol:**

1. **Client connects** to WebSocket endpoint
2. **Client sends start message:**
```json
{
  "action": "start_stream",
  "source": "camera_1",
  "engine": "openvino",
  "confidence_threshold": 0.5,
  "frame_skip": 2
}
```

3. **Server sends continuous detections:**
```json
{
  "type": "detection",
  "timestamp": "2026-03-11T10:00:00.123Z",
  "frame_number": 152,
  "source": "camera_1",
  "detections": [
    {
      "class_name": "Vehicle",
      "original_class": "Car",
      "confidence": 0.91,
      "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 400},
      "track_id": 5
    }
  ],
  "fps": 15,
  "latency_ms": 45,
  "inference_time_ms": 35
}
```

4. **Client sends stop message:**
```json
{"action": "stop_stream"}
```

**Video Processing Pipeline:**

```
Upload Video
    ↓
Save to Temp Storage
    ↓
Extract Frames (OpenCV)
    ↓
[Parallel Processing Pool]
    ↓
Run Detection on Each Frame
    ↓
[Optional] Track Vehicles (Simple IoU tracking)
    ↓
Aggregate Results
    ↓
[If requested] Generate Annotated Video
    ↓
Store Results → Return Job ID
```

**VideoProcessingPort Interface:**
```python
class VideoProcessingPort(ABC):
    @abstractmethod
    def extract_frames(self, video_path: str, interval_seconds: float) -> Iterator[np.ndarray]: ...

    @abstractmethod
    def get_video_info(self, video_path: str) -> dict: ...

    @abstractmethod
    def create_annotated_video(
        self,
        original_path: str,
        output_path: str,
        frame_detections: list[dict]
    ) -> None: ...

    @abstractmethod
    def get_frame_timestamp(self, frame_number: int, fps: float) -> float: ...
```

---

## 5. Monitoring & Observability Design

### 5.1 Metrics Endpoint

**Endpoint:** `GET /metrics`

**Response (Prometheus format):**
```yaml
# HELP vehicle_detection_requests_total Total detection requests
# TYPE vehicle_detection_requests_total counter
vehicle_detection_requests_total{engine="pytorch",status="success",endpoint="/api/v1/pytorch/detect"} 1523
vehicle_detection_requests_total{engine="openvino",status="success",endpoint="/api/v1/openvino/detect"} 2456
vehicle_detection_requests_total{engine="pytorch",status="error",endpoint="/api/v1/pytorch/detect"} 23

# HELP vehicle_detection_latency_seconds Detection latency in seconds
# TYPE vehicle_detection_latency_seconds histogram
vehicle_detection_latency_seconds_bucket{engine="pytorch",le="0.1"} 523
vehicle_detection_latency_seconds_bucket{engine="pytorch",le="0.25"} 1200
vehicle_detection_latency_seconds_bucket{engine="pytorch",le="0.5"} 1489
vehicle_detection_latency_seconds_bucket{engine="pytorch",le="1.0"} 1520
vehicle_detection_latency_seconds_sum{engine="pytorch"} 450.5
vehicle_detection_latency_seconds_count{engine="pytorch"} 1546

vehicle_detection_latency_seconds_bucket{engine="openvino",le="0.05"} 1200
vehicle_detection_latency_seconds_bucket{engine="openvino",le="0.1"} 1845
vehicle_detection_latency_seconds_bucket{engine="openvino",le="0.25"} 2400
vehicle_detection_latency_seconds_sum{engine="openvino"} 180.2
vehicle_detection_latency_seconds_count{engine="openvino"} 2456

# HELP batch_jobs_total Total batch jobs processed
# TYPE batch_jobs_total counter
batch_jobs_total{status="completed",engine="pytorch"} 45
batch_jobs_total{status="failed",engine="pytorch"} 2
batch_jobs_total{status="completed",engine="openvino"} 89
batch_jobs_total{status="failed",engine="openvino"} 1

# HELP active_batch_jobs Currently running batch jobs
# TYPE active_batch_jobs gauge
active_batch_jobs{engine="pytorch"} 1
active_batch_jobs{engine="openvino"} 2

# HELP video_processing_queue_size Pending video jobs in queue
# TYPE video_processing_queue_size gauge
video_processing_queue_size 2

# HELP video_processing_duration_seconds Video processing duration
# TYPE video_processing_duration_seconds histogram
video_processing_duration_seconds_bucket{le="30"} 5
video_processing_duration_seconds_bucket{le="60"} 12
video_processing_duration_seconds_bucket{le="120"} 20
video_processing_duration_seconds_count 25
video_processing_duration_seconds_sum 1800

# HELP detections_total Total vehicles detected
# TYPE detections_total counter
detections_total{class_name="Car",engine="pytorch"} 5000
detections_total{class_name="Bus",engine="pytorch"} 800
detections_total{class_name="Car",engine="openvino"} 8200
detections_total{class_name="Bus",engine="openvino"} 1200

# HELP api_info API information
# TYPE api_info gauge
api_info{version="1.1.0"} 1
```

### 5.2 Enhanced Health Check

**Endpoint:** `GET /api/v1/health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.1.0",
  "timestamp": "2026-03-11T10:00:00Z",
  "uptime_seconds": 86400,
  "correlation_id": "corr-uuid-123",
  "engines": ["PyTorch", "OpenVINO"],
  "components": {
    "api": {
      "status": "up",
      "latency_ms": 5
    },
    "pytorch": {
      "status": "up",
      "model_loaded": true,
      "model_path": "/app/models/best.pt",
      "inference_latency_ms": 120,
      "last_error": null
    },
    "openvino": {
      "status": "up",
      "model_loaded": true,
      "model_path": "/app/models/best_openvino_model",
      "inference_latency_ms": 45,
      "last_error": null
    },
    "job_queue": {
      "status": "up",
      "backend": "sqlite",
      "pending_jobs": 3,
      "processing_jobs": 2,
      "completed_jobs_last_hour": 45
    }
  },
  "metrics": {
    "requests_last_minute": 45,
    "requests_last_hour": 1250,
    "average_latency_ms": 85,
    "p95_latency_ms": 150,
    "p99_latency_ms": 250,
    "error_rate": 0.02
  }
}
```

### 5.3 Structured Logging

**Configuration Options:**

```bash
# Environment Variables
LOG_FORMAT=json              # json | text
LOG_LEVEL=INFO               # DEBUG | INFO | WARNING | ERROR
ENABLE_CORRELATION_IDS=true  # Include correlation IDs in all logs
LOG_OUTPUT=stdout            # stdout | file | both
LOG_FILE=/var/log/vehicle-detection.log
```

**Log Format (JSON):**

```json
// Request log
{
  "timestamp": "2026-03-11T10:00:00.123Z",
  "level": "INFO",
  "logger": "vehicle_detection_api.routers.detect",
  "correlation_id": "abc-123-xyz",
  "request_id": "req-456",
  "method": "POST",
  "path": "/api/v1/openvino/detect",
  "query_params": {"confidence_threshold": 0.5},
  "status_code": 200,
  "duration_ms": 125,
  "client_ip": "192.168.1.100",
  "user_agent": "Mozilla/5.0 (Python requests)",
  "engine": "openvino",
  "detections_count": 3,
  "image_size_bytes": 245760,
  "model_inference_time_ms": 45,
  "message": "Vehicle detection completed"
}

// Batch processing log
{
  "timestamp": "2026-03-11T10:05:00.456Z",
  "level": "INFO",
  "logger": "vehicle_detection_api.services.batch",
  "correlation_id": "def-456-uvw",
  "job_id": "job-uuid-789",
  "action": "batch_completed",
  "total_images": 50,
  "successful": 48,
  "failed": 2,
  "processing_time_seconds": 120,
  "average_time_per_image_ms": 2400,
  "message": "Batch job completed successfully"
}

// Error log
{
  "timestamp": "2026-03-11T10:10:01.789Z",
  "level": "ERROR",
  "logger": "vehicle_detection_api.adapters.torch_yolo",
  "correlation_id": "ghi-789-rst",
  "request_id": "req-789",
  "method": "POST",
  "path": "/api/v1/batch/detect",
  "error_type": "ValidationError",
  "error_code": "BATCH_SIZE_EXCEEDED",
  "error_message": "Batch size exceeds limit of 10 images",
  "stack_trace": "Traceback (most recent call last):\n  File ...",
  "client_ip": "192.168.1.100",
  "message": "Batch validation failed"
}

// Video processing log
{
  "timestamp": "2026-03-11T10:15:00.000Z",
  "level": "INFO",
  "logger": "vehicle_detection_api.services.video",
  "correlation_id": "jkl-012-opq",
  "video_id": "vid-uuid-123",
  "action": "video_processing_started",
  "filename": "traffic.mp4",
  "duration_seconds": 120,
  "frames_to_process": 120,
  "engine": "openvino",
  "message": "Started video processing job"
}
```

**Log Format (Text - when LOG_FORMAT=text):**

```
2026-03-11T10:00:00.123Z [INFO] [abc-123-xyz] Vehicle detection completed: method=POST path=/api/v1/openvino/detect status=200 duration_ms=125 engine=openvino detections=3

2026-03-11T10:10:01.789Z [ERROR] [ghi-789-rst] Batch validation failed: error_code=BATCH_SIZE_EXCEEDED message="Batch size exceeds limit of 10 images"
```

### 5.4 Correlation ID Propagation

**Middleware Flow:**

```
Request Received
    ↓
[CorrelationMiddleware] Generate or extract correlation_id from header X-Correlation-ID
    ↓
Store in contextvar (async-safe)
    ↓
[MetricsMiddleware] Start timer, track request
    ↓
Router Handler
    ↓
Service Layer (has access to correlation_id via context)
    ↓
Adapter Layer
    ↓
[MetricsMiddleware] Record metrics, log response
    ↓
Response Sent
```

**Implementation:**
```python
# Context variable for correlation ID
correlation_id_var = contextvars.ContextVar('correlation_id', default=None)

class CorrelationMiddleware:
    async def __call__(self, request: Request, call_next):
        correlation_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
        correlation_id_var.set(correlation_id)

        response = await call_next(request)
        response.headers['X-Correlation-ID'] = correlation_id
        return response

# Usage in logger
class StructuredLogger:
    def _log(self, level: str, message: str, **kwargs):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'correlation_id': correlation_id_var.get(),
            **kwargs
        }
        print(json.dumps(log_entry))
```

---

## 6. API Specifications

### 6.1 New Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/batch/detect` | Sync batch detection |
| POST | `/api/v1/jobs/batch` | Async batch detection |
| GET | `/api/v1/jobs` | List all jobs |
| GET | `/api/v1/jobs/{id}` | Get job status/results |
| DELETE | `/api/v1/jobs/{id}` | Cancel/delete job |
| POST | `/api/v1/video/process` | Submit video for processing |
| GET | `/api/v1/jobs/{id}/annotated_video` | Download annotated video |
| GET | `/metrics` | Prometheus metrics |
| WS | `/api/v1/stream` | Real-time streaming |

### 6.2 Existing Endpoint Enhancements

**Health Check** (`/api/v1/health`)
- Added: Component status details
- Added: Real-time metrics (requests/minute, latency)
- Added: Correlation ID in response

### 6.3 Rate Limiting Headers

All responses include rate limiting headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1647000000
X-Correlation-ID: abc-123-xyz
```

---

## 7. Data Models

### 7.1 Pydantic Models

```python
# Batch Request Models
class ImageInput(BaseModel):
    filename: str
    data: str  # base64 encoded

class BatchDetectRequest(BaseModel):
    engine: Literal["pytorch", "openvino"]
    images: list[ImageInput]
    confidence_threshold: float = 0.5
    max_concurrent: int = Field(default=3, ge=1, le=5)

class BatchDetectResponse(BaseModel):
    batch_id: str
    status: Literal["completed", "partial", "failed"]
    engine: str
    total_images: int
    processing_time_ms: int
    results: list[ImageResult]
    summary: BatchSummary

class ImageResult(BaseModel):
    filename: str
    status: Literal["success", "error"]
    detections: list[Detection] | None
    processing_time_ms: int | None
    error: str | None
    error_code: str | None

# Video Models
class VideoProcessRequest(BaseModel):
    engine: Literal["pytorch", "openvino"]
    output_format: Literal["json", "annotated_video", "both"]
    frame_interval: float = Field(default=1.0, ge=0.1, le=60.0)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    webhook_url: HttpUrl | None = None

class VideoMetadata(BaseModel):
    duration_seconds: float
    fps: float
    resolution: str
    codec: str
    total_frames: int

class VideoFrameResult(BaseModel):
    timestamp: float
    frame_number: int
    detections: list[Detection]

class VideoResult(BaseModel):
    video_id: str
    status: JobStatus
    input: VideoMetadata
    processing: ProcessingInfo
    frames: list[VideoFrameResult]
    summary: VideoSummary
    outputs: OutputUrls

# Job Models
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Job(BaseModel):
    job_id: str
    status: JobStatus
    job_type: Literal["batch", "video"]
    engine: str
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    progress: JobProgress | None
    error: JobError | None

class JobProgress(BaseModel):
    processed: int
    total: int
    percentage: int
    current_file: str | None

class JobError(BaseModel):
    code: str
    message: str
    details: dict | None = None
    timestamp: datetime

# Detection Models (extend existing)
class Detection(BaseModel):
    class_name: str
    original_class: str
    confidence: float
    bbox: BoundingBox
    track_id: int | None = None

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
```

---

## 8. Configuration

### 8.1 Environment Variables

```python
class Settings(BaseSettings):
    # Existing settings...

    # Batch Processing
    BATCH_SYNC_MAX_IMAGES: int = 10
    BATCH_ASYNC_MAX_IMAGES: int = 100
    BATCH_SYNC_TIMEOUT_SECONDS: int = 30
    BATCH_CONCURRENT_WORKERS: int = 3

    # Video Processing
    VIDEO_MAX_DURATION_SECONDS: int = 600  # 10 minutes
    VIDEO_MAX_FILE_SIZE_MB: int = 500
    VIDEO_TEMP_DIR: Path = Path("/tmp/video_processing")
    VIDEO_FRAME_INTERVAL_DEFAULT: float = 1.0

    # Job Queue
    JOB_QUEUE_BACKEND: Literal["sqlite", "redis"] = "sqlite"
    JOB_QUEUE_SQLITE_PATH: Path = PROJECT_ROOT / "data" / "jobs.db"
    JOB_QUEUE_REDIS_URL: str = "redis://localhost:6379/0"
    JOB_MAX_CONCURRENT: int = 4
    JOB_CLEANUP_DAYS: int = 7

    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_ENDPOINT: str = "/metrics"

    # Logging
    LOG_FORMAT: Literal["json", "text"] = "json"
    LOG_LEVEL: str = "INFO"
    ENABLE_CORRELATION_IDS: bool = True
    LOG_OUTPUT: Literal["stdout", "file", "both"] = "stdout"
    LOG_FILE_PATH: Path | None = None

    # WebSocket
    WS_MAX_CONNECTIONS: int = 10
    WS_PING_INTERVAL: int = 30
```

### 8.2 Default Configuration File

```yaml
# config/default.yaml
batch:
  sync_max_images: 10
  async_max_images: 100
  sync_timeout_seconds: 30
  concurrent_workers: 3

video:
  max_duration_seconds: 600
  max_file_size_mb: 500
  temp_dir: /tmp/video_processing
  frame_interval_default: 1.0

job_queue:
  backend: sqlite
  sqlite_path: ./data/jobs.db
  max_concurrent: 4
  cleanup_days: 7

monitoring:
  enable_metrics: true
  metrics_endpoint: /metrics

logging:
  format: json
  level: INFO
  enable_correlation_ids: true
  output: stdout

websocket:
  max_connections: 10
  ping_interval: 30
```

---

## 9. Error Handling

### 9.1 Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `BATCH_SIZE_EXCEEDED` | 400 | Sync batch > 10 images |
| `BATCH_ASYNC_SIZE_EXCEEDED` | 400 | Async batch > 100 images |
| `INVALID_IMAGE_FORMAT` | 400 | Image cannot be decoded |
| `VIDEO_TOO_LARGE` | 400 | Video exceeds max size |
| `VIDEO_TOO_LONG` | 400 | Video exceeds max duration |
| `UNSUPPORTED_VIDEO_FORMAT` | 400 | Video codec not supported |
| `JOB_NOT_FOUND` | 404 | Job ID doesn't exist |
| `JOB_EXPIRED` | 410 | Job results expired |
| `ENGINE_NOT_AVAILABLE` | 503 | PyTorch/OpenVINO not loaded |
| `QUEUE_FULL` | 503 | Job queue at capacity |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |

### 9.2 Error Response Format

```json
{
  "error": {
    "code": "BATCH_SIZE_EXCEEDED",
    "message": "Batch size of 25 exceeds maximum of 10 for sync processing",
    "details": {
      "provided": 25,
      "maximum": 10,
      "suggestion": "Use async batch endpoint for large batches"
    },
    "correlation_id": "abc-123-xyz",
    "timestamp": "2026-03-11T10:00:00Z"
  }
}
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

| Component | Test Coverage |
|-----------|---------------|
| BatchService | Sync batch, async batch, validation, error handling |
| VideoService | Frame extraction, video info, annotated output |
| JobQueueService | Create, update, query jobs, queue management |
| MetricsService | Counter increments, histogram buckets, gauges |
| StructuredLogger | JSON format, correlation ID injection |
| VideoAdapter | Frame extraction, video metadata, encoding |

### 10.2 Integration Tests

| Scenario | Test |
|----------|------|
| Sync batch | Upload 5 images, verify all processed |
| Async batch | Submit 50 images, poll for completion |
| Video processing | Upload 30s video, verify frame detections |
| WebSocket | Connect, stream frames, verify detections |
| Metrics | Make requests, verify `/metrics` output |
| Correlation IDs | Request with header, verify in response/logs |

### 10.3 Load Tests

| Scenario | Target |
|----------|--------|
| Sync batch | 10 req/sec, 95% < 5s latency |
| Async batch | 100 jobs/minute enqueue |
| Video processing | 5 concurrent videos |
| WebSocket | 10 concurrent streams |
| Metrics endpoint | 100 req/sec |

---

## 11. Deployment Considerations

### 11.1 Resource Requirements

| Component | Memory | CPU | Notes |
|-----------|--------|-----|-------|
| API Server | 512MB | 1 | FastAPI + routers |
| PyTorch Model | 1GB | 1-2 | Model loaded in memory |
| OpenVINO Model | 512MB | 1 | Model loaded in memory |
| Video Processing | 2GB | 2 | Temporary for frame extraction |
| Job Queue (Redis) | 256MB | 0.5 | If using Redis backend |

### 11.2 Docker Considerations

- Mount `/tmp/video_processing` as tmpfs for performance
- Set `VIDEO_TEMP_DIR` to Docker volume for persistence
- Redis container for production job queue
- Health check endpoint for container orchestration

### 11.3 Monitoring Alerts

| Alert | Condition |
|-------|-----------|
| High Error Rate | `error_rate > 5%` for 5 minutes |
| High Latency | `p95_latency > 2s` for 10 minutes |
| Queue Backlog | `queue_size > 20` for 15 minutes |
| Model Down | `engine_status == down` |
| Disk Space | `disk_usage > 85%` |

---

## Appendix A: Sequence Diagrams

### A.1 Sync Batch Processing

```
Client → API: POST /api/v1/batch/detect
API → BatchService: process_sync_batch(images, engine)
BatchService → ImageAdapter: decode_image(img) [parallel]
BatchService → DetectionAdapter: detect_objects(img) [parallel]
BatchService → BatchService: aggregate_results()
API → Client: 200 OK with results
```

### A.2 Async Batch Processing

```
Client → API: POST /api/v1/jobs/batch
API → JobQueueService: create_job(batch_data)
JobQueueService → SQLite: INSERT job
API → Client: 202 Accepted with job_id

[Background Worker]
JobQueueService → SQLite: SELECT next_pending_job()
JobQueueService → BatchService: process_async_batch(job)
BatchService → [process images]
BatchService → JobQueueService: update_job(completed)
JobQueueService → SQLite: UPDATE job
[Webhook if configured]

Client → API: GET /api/v1/jobs/{job_id}
API → JobQueueService: get_job(job_id)
JobQueueService → SQLite: SELECT job
API → Client: 200 OK with status/results
```

### A.3 Video Processing

```
Client → API: POST /api/v1/video/process
API → VideoService: submit_video(file, config)
VideoService → JobQueueService: create_job(video_data)
API → Client: 202 Accepted with video_id

[Background Worker]
JobQueueService → VideoService: process_video(job)
VideoService → VideoAdapter: extract_frames(video_path)
VideoAdapter → VideoService: yield frames
VideoService → DetectionAdapter: detect_objects(frame) [parallel]
VideoService → [optional] track_vehicles()
VideoService → [optional] VideoAdapter: create_annotated_video()
VideoService → JobQueueService: update_job(completed)
```

---

## Appendix B: Database Schema (SQLite)

```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,  -- queued, processing, completed, failed, cancelled
    job_type TEXT NOT NULL,  -- batch, video
    engine TEXT NOT NULL,
    data TEXT NOT NULL,  -- JSON blob
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    results TEXT,  -- JSON blob
    error TEXT,
    webhook_url TEXT,
    progress_current INTEGER,
    progress_total INTEGER
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created ON jobs(created_at);
CREATE INDEX idx_jobs_type ON jobs(job_type);

-- Cleanup old jobs
DELETE FROM jobs WHERE created_at < datetime('now', '-7 days');
```

---

**Document Version:** 1.0
**Last Updated:** 2026-03-11
**Status:** Ready for Implementation
