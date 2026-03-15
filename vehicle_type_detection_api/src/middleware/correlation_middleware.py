"""
Middleware for injecting correlation IDs into requests.
"""

import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.correlation import clear_correlation_id, set_correlation_id


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation ID to each request."""

    async def dispatch(self, request: Request, call_next):
        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Set in context
        set_correlation_id(correlation_id)

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        # Clear context
        clear_correlation_id()

        return response
