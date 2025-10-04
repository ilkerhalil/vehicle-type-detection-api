import pytest
from vehicle_type_detection_api.src.core.config import get_settings, Settings

def test_get_settings_returns_settings():
    settings = get_settings()
    assert isinstance(settings, Settings)
