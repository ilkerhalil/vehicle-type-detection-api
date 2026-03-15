"""
Correlation ID management for request tracing.
Uses contextvars for async-safe correlation ID storage.
"""

import contextvars
import uuid
from typing import Optional

# Context variable for storing correlation ID per request
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("correlation_id", default=None)


def generate_correlation_id() -> str:
    """Generate a new unique correlation ID."""
    return str(uuid.uuid4())


def get_correlation_id() -> str:
    """
    Get the current correlation ID.
    If none exists, generates a new one.
    """
    cid = correlation_id_var.get()
    if cid is None:
        cid = generate_correlation_id()
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_var.set(correlation_id)


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    correlation_id_var.set(None)
