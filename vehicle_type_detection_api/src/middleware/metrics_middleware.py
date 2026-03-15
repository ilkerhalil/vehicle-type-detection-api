"""
Middleware for tracking request metrics.
"""

import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..services.metrics_service import MetricsService


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics."""

    def __init__(self, app, metrics_service: MetricsService):
        super().__init__(app)
        self.metrics_service = metrics_service

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Determine engine from path
        path = request.url.path
        engine = self._extract_engine_from_path(path)
        endpoint = path

        response = await call_next(request)

        # Calculate latency
        latency = time.time() - start_time

        # Determine status
        status = "success" if response.status_code < 400 else "error"

        # Record metrics
        self.metrics_service.increment_request_count(engine, status, endpoint)
        self.metrics_service.record_latency(engine, latency)

        return response

    def _extract_engine_from_path(self, path: str) -> str:
        """Extract engine type from request path."""
        if "pytorch" in path:
            return "pytorch"
        elif "openvino" in path:
            return "openvino"
        return "unknown"
