"""
Vehicle Detection API Routes - Object Detection
FastAPI routes for vehicle object detection using YOLO models
"""

import io
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from ..adapters.dependencies import (
    OPENVINO_AVAILABLE,
    TORCH_AVAILABLE,
    get_openvino_vehicle_object_detection_service,
    get_torch_vehicle_object_detection_service,
)
from ..core.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Vehicle Detection"])


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Unified health check endpoint for all adapters

    Returns:
        Dictionary with health status and service information for all available adapters
    """
    health_data = {
        "status": "healthy",
        "version": "1.0.0",
        "detection_type": "object_detection",
        "engines": ["PyTorch", "OpenVINO"],
        "adapters": {},
    }

    # Check PyTorch adapter
    if TORCH_AVAILABLE:
        try:
            torch_service = get_torch_vehicle_object_detection_service()
            health_data["adapters"]["pytorch"] = {
                "available": True,
                "ready": torch_service.is_ready(),
                "model_type": "PyTorch YOLO",
                "backend": "ultralytics",
                "supported_classes": torch_service.get_supported_classes(),
                "num_classes": len(torch_service.get_supported_classes()),
                "endpoints": {"detect": "/api/v1/pytorch/detect", "annotated": "/api/v1/pytorch/detect/annotated"},
            }
        except Exception as e:
            health_data["adapters"]["pytorch"] = {"available": False, "error": str(e), "model_type": "PyTorch YOLO"}
    else:
        health_data["adapters"]["pytorch"] = {
            "available": False,
            "reason": "PyTorch not available",
            "model_type": "PyTorch YOLO",
        }

    # Check OpenVINO adapter
    if OPENVINO_AVAILABLE:
        try:
            openvino_service = get_openvino_vehicle_object_detection_service()
            health_data["adapters"]["openvino"] = {
                "available": True,
                "ready": openvino_service.is_ready(),
                "model_type": "OpenVINO YOLO",
                "backend": "openvino",
                "supported_classes": openvino_service.get_supported_classes(),
                "num_classes": len(openvino_service.get_supported_classes()),
                "endpoints": {"detect": "/api/v1/openvino/detect", "annotated": "/api/v1/openvino/detect/annotated"},
            }
        except Exception as e:
            health_data["adapters"]["openvino"] = {"available": False, "error": str(e), "model_type": "OpenVINO YOLO"}
    else:
        health_data["adapters"]["openvino"] = {
            "available": False,
            "reason": "OpenVINO not available",
            "model_type": "OpenVINO YOLO",
        }

    # Determine overall status
    available_adapters = [name for name, adapter in health_data["adapters"].items() if adapter.get("available", False)]
    ready_adapters = [name for name, adapter in health_data["adapters"].items() if adapter.get("ready", False)]

    if not available_adapters:
        health_data["status"] = "error"
        health_data["message"] = "No adapters available"
    elif not ready_adapters:
        health_data["status"] = "starting"
        health_data["message"] = "Adapters available but not ready"
    else:
        health_data["status"] = "healthy"
        health_data["message"] = f"Ready adapters: {', '.join(ready_adapters)}"

    health_data["available_adapters"] = available_adapters
    health_data["ready_adapters"] = ready_adapters

    return health_data


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint - simpler version of health for load balancers

    Returns:
        Dictionary with basic readiness information
    """
    ready_adapters = []
    ready_adapters.extend(_check_pytorch_readiness())
    ready_adapters.extend(_check_openvino_readiness())

    is_ready = len(ready_adapters) > 0
    return {"ready": is_ready, "ready_adapters": ready_adapters, "status": "ready" if is_ready else "not_ready"}


def _check_pytorch_readiness() -> List[str]:
    """Check PyTorch adapter readiness"""
    if TORCH_AVAILABLE:
        try:
            torch_service = get_torch_vehicle_object_detection_service()
            if torch_service.is_ready():
                return ["pytorch"]
        except Exception:
            pass
    return []


def _check_openvino_readiness() -> List[str]:
    """Check OpenVINO adapter readiness"""
    if OPENVINO_AVAILABLE:
        try:
            openvino_service = get_openvino_vehicle_object_detection_service()
            if openvino_service.is_ready():
                return ["openvino"]
        except Exception:
            pass
    return []


# ====================================
# PyTorch YOLO Engine Endpoints
# ====================================


@router.get("/pytorch/classes")
async def get_pytorch_detection_classes(
    detection_service=Depends(get_torch_vehicle_object_detection_service),
) -> Dict[str, Any]:
    """
    Get supported detection classes for PyTorch adapter

    Returns:
        Dictionary with supported classes information
    """
    try:
        classes = detection_service.get_supported_classes()
        return {
            "supported_classes": classes,
            "total_classes": len(classes),
            "detection_type": "object_detection",
            "model_type": "PyTorch YOLO",
            "backend": "ultralytics",
        }
    except Exception as e:
        logger.error(f"Failed to get PyTorch detection classes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get classes: {str(e)}")


@router.post("/pytorch/detect")
async def pytorch_detect_objects(
    file: UploadFile = File(...),
    detection_service=Depends(get_torch_vehicle_object_detection_service),
) -> JSONResponse:
    """
    Detect vehicles in uploaded image using PyTorch YOLO model

    Args:
        file: Uploaded image file
        detection_service: Vehicle detection service dependency

    Returns:
        JSON response with detected vehicles and bounding boxes
    """
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process image
        image_data = await file.read()
        detections = detection_service.detect_vehicles_in_image(image_data)

        # Prepare response
        if hasattr(detections, "vehicle_detections"):
            vehicle_list = detections.vehicle_detections
        else:
            vehicle_list = detections.get("vehicle_detections", [])

        response_data = {
            "message": "Vehicle detection completed (PyTorch)",
            "detections": vehicle_list,
            "filename": file.filename,
            "filesize": len(image_data),
            "model_type": "PyTorch YOLO",
            "backend": "ultralytics",
        }

        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        logger.error(f"PyTorch detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/pytorch/detect/annotated")
async def pytorch_detect_annotated(
    file: UploadFile = File(...),
    detection_service=Depends(get_torch_vehicle_object_detection_service),
):
    """
    Detect vehicles and return annotated image using PyTorch
    """
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type")

        logger.info(f"Processing PyTorch annotated detection for: {file.filename}")

        image_data = await file.read()
        annotated_image_bytes = await detection_service.detect_and_annotate_vehicles_from_bytes(image_data)

        logger.info(f"PyTorch annotated detection completed for: {file.filename}")

        return StreamingResponse(
            io.BytesIO(annotated_image_bytes),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename=annotated_{file.filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PyTorch annotated detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


# ====================================
# OpenVINO Engine Endpoints
# ====================================


@router.get("/openvino/classes")
async def get_openvino_detection_classes(
    detection_service=Depends(get_openvino_vehicle_object_detection_service),
) -> Dict[str, Any]:
    """
    Get supported detection classes for OpenVINO adapter

    Returns:
        Dictionary with supported classes information
    """
    try:
        classes = detection_service.get_supported_classes()
        return {
            "supported_classes": classes,
            "total_classes": len(classes),
            "detection_type": "object_detection",
            "model_type": "OpenVINO YOLO",
            "backend": "openvino",
        }
    except Exception as e:
        logger.error(f"Failed to get OpenVINO detection classes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get classes: {str(e)}")


@router.post("/openvino/detect")
async def detect_vehicles_openvino(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    service=Depends(get_openvino_vehicle_object_detection_service),
) -> Dict[str, Any]:
    """
    Detect vehicles using OpenVINO optimized model
    """
    _validate_openvino_availability()
    _validate_file_and_thresholds(file, confidence_threshold, iou_threshold)
    _validate_service_components(service)

    logger.info(f"Processing OpenVINO detection for: {file.filename}")

    try:
        image_data = await file.read()
        result = await _process_openvino_detection(service, image_data, confidence_threshold, iou_threshold)

        logger.info(f"OpenVINO detection completed: {len(result.get('vehicle_detections', []))} vehicles found")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenVINO detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


def _validate_service_components(service):
    """Validate service and its components"""
    if service is None:
        raise HTTPException(status_code=500, detail="OpenVINO service is None")
    if service.detection_adapter is None:
        raise HTTPException(status_code=500, detail="OpenVINO detection_adapter is None")


async def _process_openvino_detection(
    service, image_data: bytes, confidence_threshold: float, iou_threshold: float
) -> Dict[str, Any]:
    """Process OpenVINO detection"""
    service.detection_adapter.set_confidence_threshold(confidence_threshold)
    service.detection_adapter.set_iou_threshold(iou_threshold)
    return await service.detect_vehicles_from_bytes(image_data)


@router.post("/openvino/detect/annotated")
async def detect_vehicles_annotated_openvino(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    service=Depends(get_openvino_vehicle_object_detection_service),
):
    """
    Detect vehicles and return annotated image using OpenVINO
    """
    _validate_openvino_availability()
    _validate_file_and_thresholds(file, confidence_threshold, iou_threshold)

    logger.info(f"Processing OpenVINO annotated detection for: {file.filename}")

    try:
        image_data = await file.read()
        annotated_image_bytes = await _process_openvino_annotated_detection(
            service, image_data, confidence_threshold, iou_threshold
        )

        logger.info(f"OpenVINO annotated detection completed for: {file.filename}")

        return StreamingResponse(
            io.BytesIO(annotated_image_bytes),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename=annotated_{file.filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenVINO annotated detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


def _validate_openvino_availability():
    """Validate OpenVINO availability"""
    if not OPENVINO_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenVINO adapter not available")


def _validate_file_and_thresholds(file: UploadFile, confidence_threshold: float, iou_threshold: float):
    """Validate file type and threshold parameters"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="confidence_threshold must be between 0.0 and 1.0")

    if not 0.0 <= iou_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="iou_threshold must be between 0.0 and 1.0")


async def _process_openvino_annotated_detection(
    service, image_data: bytes, confidence_threshold: float, iou_threshold: float
) -> bytes:
    """Process OpenVINO annotated detection"""
    service.detection_adapter.set_confidence_threshold(confidence_threshold)
    service.detection_adapter.set_iou_threshold(iou_threshold)
    return await service.detect_and_annotate_vehicles_from_bytes(image_data)


@router.get("/")
async def api_info() -> Dict[str, Any]:
    """
    Get API information

    Returns:
        Dictionary with API information
    """
    endpoints = {"health": "/api/v1/health", "ready": "/api/v1/ready"}

    # Add conditional endpoints
    if TORCH_AVAILABLE:
        endpoints.update(
            {
                "pytorch_classes": "/api/v1/pytorch/classes",
                "pytorch_detect": "/api/v1/pytorch/detect",
                "pytorch_detect_annotated": "/api/v1/pytorch/detect/annotated",
            }
        )

    if OPENVINO_AVAILABLE:
        endpoints.update(
            {
                "openvino_classes": "/api/v1/openvino/classes",
                "openvino_detect": "/api/v1/openvino/detect",
                "openvino_detect_annotated": "/api/v1/openvino/detect/annotated",
            }
        )

    return {
        "message": "Vehicle Detection API v1",
        "version": "1.0.0",
        "detection_type": "object_detection",
        "model_types": ["PyTorch YOLO", "OpenVINO YOLO"],
        "endpoints": endpoints,
        "features": [
            "Vehicle object detection",
            "Bounding box coordinates",
            "Multiple vehicle detection",
            "Confidence scores",
            "Multi-adapter support",
            "Annotated image output",
        ],
    }
