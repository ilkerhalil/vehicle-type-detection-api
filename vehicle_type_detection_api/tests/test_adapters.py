import pytest
from vehicle_type_detection_api.src.adapters.image_adapter import OpenCVImageProcessingAdapter
from vehicle_type_detection_api.src.adapters.openvino_adapter import OpenVINOVehicleDetectionAdapter
from vehicle_type_detection_api.src.adapters.torch_yolo_adapter import TorchYOLODetectionAdapter

def test_open_cv_adapter_instantiation():
    adapter = OpenCVImageProcessingAdapter()
    assert hasattr(adapter, "preprocess_image")
    assert hasattr(adapter, "decode_image_from_bytes")
    assert hasattr(adapter, "draw_bounding_boxes")
    assert hasattr(adapter, "encode_image_to_bytes")

def test_openvino_adapter_instantiation(tmp_path):
    # Create dummy model and labels files
    model_path = tmp_path / "model.bin"
    labels_path = tmp_path / "labels.txt"
    model_path.write_bytes(b"")
    labels_path.write_text("car\ntruck\n")
    adapter = OpenVINOVehicleDetectionAdapter(str(model_path), str(labels_path))
    assert hasattr(adapter, "detect_objects")
    assert hasattr(adapter, "detect_vehicles")
    assert hasattr(adapter, "get_model_info")
    assert adapter.is_ready()

def test_torch_yolo_adapter_instantiation(tmp_path):
    # Create dummy model file
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"")
    adapter = TorchYOLODetectionAdapter(str(model_path))
    assert hasattr(adapter, "detect_objects")
    assert hasattr(adapter, "detect_and_annotate")
    assert hasattr(adapter, "get_supported_classes")
