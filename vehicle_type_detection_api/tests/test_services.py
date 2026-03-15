from vehicle_type_detection_api.src.adapters.image_adapter import (
    OpenCVImageProcessingAdapter,
)
from vehicle_type_detection_api.src.adapters.openvino_adapter import (
    OpenVINOVehicleDetectionAdapter,
)
from vehicle_type_detection_api.src.services.detection_service import (
    VehicleObjectDetectionService,
)


def test_vehicle_object_detection_service_instantiation(tmp_path):
    # Create dummy model and labels files
    model_path = tmp_path / "model.bin"
    labels_path = tmp_path / "labels.txt"
    model_path.write_bytes(b"")
    labels_path.write_text("car\ntruck\n")
    detection_adapter = OpenVINOVehicleDetectionAdapter(str(model_path), str(labels_path))
    image_adapter = OpenCVImageProcessingAdapter()
    service = VehicleObjectDetectionService(detection_adapter, image_adapter)
    assert service.is_ready()
