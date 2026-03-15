"""
Metrics endpoint for Prometheus scraping.
"""

from fastapi import APIRouter, Response

from ..core.config import get_settings
from ..services.metrics_service import MetricsService

router = APIRouter(tags=["Monitoring"])
settings = get_settings()

# Global metrics instance
_metrics_service = MetricsService()


def get_metrics_service() -> MetricsService:
    """Get the global metrics service instance."""
    return _metrics_service


@router.get(settings.METRICS_ENDPOINT)
async def metrics() -> Response:
    """Prometheus-compatible metrics endpoint."""
    if not settings.ENABLE_METRICS:
        return Response(content="Metrics disabled", status_code=404)

    metrics_service = get_metrics_service()
    content = metrics_service.get_prometheus_format()

    return Response(content=content, media_type="text/plain")
