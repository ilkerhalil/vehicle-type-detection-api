"""
Vehicle Type Detection API
FastAPI application with Hexagonal Architecture and dependency injection
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.logger import setup_logger
from .routers.detect import router as detect_router

logger = setup_logger(__name__)

# Get cached settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Vehicle Type Detection API with Hexagonal Architecture...")

    # Check if required model files exist
    pytorch_model_path = settings.PROJECT_ROOT / "models" / "best.pt"
    openvino_model_path = settings.PROJECT_ROOT / "models" / "best_openvino_model" / "best.xml"

    if not pytorch_model_path.exists():
        logger.error(f"PyTorch model file not found: {pytorch_model_path}")
        raise FileNotFoundError(f"PyTorch model file not found: {pytorch_model_path}")

    if not openvino_model_path.exists():
        logger.warning(f"OpenVINO model file not found: {openvino_model_path}")

    # Initialize adapters
    try:
        # Test PyTorch adapter initialization
        from .adapters.dependencies import get_torch_yolo_detection_adapter

        get_torch_yolo_detection_adapter()
        logger.info("PyTorch YOLO detection adapter initialized successfully")

        # Try OpenVINO adapter
        try:
            from .adapters.dependencies import get_openvino_detection_adapter

            get_openvino_detection_adapter()
            logger.info("OpenVINO detection adapter initialized successfully")
        except Exception as openvino_error:
            logger.warning(f"OpenVINO adapter not available: {openvino_error}")

        logger.info("Vehicle Detection API v2 ready with available adapters")

    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Vehicle Type Detection API...")


# Create FastAPI app
app = FastAPI(
    title="Vehicle Type Detection API",
    description="API for detecting vehicle types using AI models (PyTorch, OpenVINO)",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detect_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vehicle Type Detection API",
        "version": "1.0.0",
        "status": "running",
        "architecture": "Hexagonal Architecture with FastAPI",
        "engines": ["PyTorch", "OpenVINO"],
        "endpoints": {"health": "/api/v1/health", "pytorch": "/api/v1/pytorch/", "openvino": "/api/v1/openvino/"},
    }


if __name__ == "__main__":
    import uvicorn

    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
