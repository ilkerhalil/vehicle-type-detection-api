import contextvars

from vehicle_type_detection_api.src.core.correlation import (
    clear_correlation_id,
    correlation_id_var,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
)


def test_generate_correlation_id():
    cid = generate_correlation_id()
    assert isinstance(cid, str)
    assert len(cid) == 36  # UUID format


def test_set_and_get_correlation_id():
    test_id = "test-correlation-id-123"
    set_correlation_id(test_id)
    assert get_correlation_id() == test_id


def test_get_correlation_id_default():
    clear_correlation_id()
    cid = get_correlation_id()
    assert cid is not None
    assert isinstance(cid, str)


def test_clear_correlation_id():
    test_id = "test-correlation-id-456"
    set_correlation_id(test_id)
    assert get_correlation_id() == test_id
    clear_correlation_id()
    # After clear, get_correlation_id should generate a new one
    new_cid = get_correlation_id()
    assert new_cid is not None
    assert isinstance(new_cid, str)


def test_correlation_id_var_is_contextvar():
    assert isinstance(correlation_id_var, contextvars.ContextVar)
    assert correlation_id_var.name == "correlation_id"
