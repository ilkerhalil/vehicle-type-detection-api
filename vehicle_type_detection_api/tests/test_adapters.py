from abc import ABC

import pytest
from vehicle_type_detection_api.src.adapters.image_adapter import (
    OpenCVImageProcessingAdapter,
)
from vehicle_type_detection_api.src.adapters.ports import (
    JobStoragePort,
    VideoProcessingPort,
)


def test_open_cv_adapter_instantiation():
    adapter = OpenCVImageProcessingAdapter()
    assert hasattr(adapter, "preprocess_image")
    assert hasattr(adapter, "decode_image_from_bytes")
    assert hasattr(adapter, "draw_bounding_boxes")
    assert hasattr(adapter, "encode_image_to_bytes")


def test_openvino_adapter_instantiation(tmp_path):
    # Skip this test as it requires a valid OpenVINO model file
    # The adapter initialization requires actual model files which are not available in tests
    pytest.skip("Requires valid OpenVINO model files")


def test_torch_yolo_adapter_instantiation(tmp_path):
    # Skip this test as it requires a valid PyTorch model file
    # The adapter initialization requires actual model files which are not available in tests
    pytest.skip("Requires valid PyTorch model files")


def test_job_storage_port_is_abstract():
    """Test that JobStoragePort is an abstract base class"""
    assert issubclass(JobStoragePort, ABC)


def test_job_storage_port_has_required_methods():
    """Test that JobStoragePort defines all required abstract methods"""
    required_methods = ["create_job", "get_job", "update_job", "get_next_pending_job", "list_jobs", "delete_job"]
    for method in required_methods:
        assert hasattr(JobStoragePort, method), f"Missing method: {method}"
        # Check that it's an abstract method
        assert getattr(JobStoragePort, method).__isabstractmethod__, f"{method} should be abstract"


def test_job_storage_port_method_signatures():
    """Test that JobStoragePort methods have correct signatures"""
    import inspect

    # Test create_job signature
    sig = inspect.signature(JobStoragePort.create_job)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "job_data" in params

    # Test get_job signature
    sig = inspect.signature(JobStoragePort.get_job)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "job_id" in params

    # Test update_job signature
    sig = inspect.signature(JobStoragePort.update_job)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "job_id" in params
    assert "updates" in params

    # Test get_next_pending_job signature
    sig = inspect.signature(JobStoragePort.get_next_pending_job)
    params = list(sig.parameters.keys())
    assert "self" in params

    # Test list_jobs signature
    sig = inspect.signature(JobStoragePort.list_jobs)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "status" in params
    assert "job_type" in params
    assert "limit" in params
    assert "offset" in params

    # Test delete_job signature
    sig = inspect.signature(JobStoragePort.delete_job)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "job_id" in params


def test_video_processing_port_is_abstract():
    """Test that VideoProcessingPort is an abstract base class"""
    assert issubclass(VideoProcessingPort, ABC)


def test_video_processing_port_has_required_methods():
    """Test that VideoProcessingPort defines all required abstract methods"""
    required_methods = ["extract_frames", "get_video_info", "create_annotated_video", "get_frame_timestamp"]
    for method in required_methods:
        assert hasattr(VideoProcessingPort, method), f"Missing method: {method}"
        assert getattr(VideoProcessingPort, method).__isabstractmethod__, f"{method} should be abstract"


class TestJobStoragePort:
    """Tests for JobStoragePort abstract interface"""

    def test_job_storage_port_is_abstract(self):
        """JobStoragePort should be an abstract base class"""
        assert issubclass(JobStoragePort, ABC)

    def test_job_storage_port_has_required_methods(self):
        """JobStoragePort should define all required abstract methods"""
        required_methods = ["create_job", "get_job", "update_job", "get_next_pending_job", "list_jobs", "delete_job"]

        for method_name in required_methods:
            assert hasattr(JobStoragePort, method_name), f"Missing method: {method_name}"
            method = getattr(JobStoragePort, method_name)
            assert getattr(method, "__isabstractmethod__", False), f"{method_name} should be abstract"

    def test_create_job_signature(self):
        """create_job should accept job_data dict and return job_id string"""
        import inspect

        sig = inspect.signature(JobStoragePort.create_job)
        params = list(sig.parameters.keys())
        assert "job_data" in params

    def test_get_job_signature(self):
        """get_job should accept job_id string and return job dict or None"""
        import inspect

        sig = inspect.signature(JobStoragePort.get_job)
        params = list(sig.parameters.keys())
        assert "job_id" in params

    def test_update_job_signature(self):
        """update_job should accept job_id and updates dict"""
        import inspect

        sig = inspect.signature(JobStoragePort.update_job)
        params = list(sig.parameters.keys())
        assert "job_id" in params
        assert "updates" in params

    def test_get_next_pending_job_signature(self):
        """get_next_pending_job should take no args and return job or None"""
        import inspect

        sig = inspect.signature(JobStoragePort.get_next_pending_job)
        # Should only have 'self'
        params = [p for p in sig.parameters.keys() if p != "self"]
        assert len(params) == 0

    def test_list_jobs_signature(self):
        """list_jobs should accept status, job_type, limit, offset filters"""
        import inspect

        sig = inspect.signature(JobStoragePort.list_jobs)
        params = list(sig.parameters.keys())
        assert "status" in params
        assert "job_type" in params
        assert "limit" in params
        assert "offset" in params

    def test_delete_job_signature(self):
        """delete_job should accept job_id and return bool"""
        import inspect

        sig = inspect.signature(JobStoragePort.delete_job)
        params = list(sig.parameters.keys())
        assert "job_id" in params

    @pytest.mark.asyncio
    async def test_job_storage_port_cannot_be_instantiated(self):
        """JobStoragePort abstract class cannot be instantiated directly"""
        with pytest.raises(TypeError):
            JobStoragePort()


class TestVideoProcessingPort:
    """Tests for VideoProcessingPort abstract interface"""

    def test_video_processing_port_is_abstract(self):
        """VideoProcessingPort should be an abstract base class"""
        assert issubclass(VideoProcessingPort, ABC)

    def test_video_processing_port_has_required_methods(self):
        """VideoProcessingPort should define all required abstract methods"""
        required_methods = ["extract_frames", "get_video_info", "create_annotated_video", "get_frame_timestamp"]

        for method_name in required_methods:
            assert hasattr(VideoProcessingPort, method_name), f"Missing method: {method_name}"
            method = getattr(VideoProcessingPort, method_name)
            assert getattr(method, "__isabstractmethod__", False), f"{method_name} should be abstract"
