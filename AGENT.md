# Vehicle Type Detection API - AI Agent Documentation

## Overview

This document describes the AI-powered vehicle detection agent system implemented in this project. The agent provides intelligent vehicle type detection capabilities using multiple machine learning backends with a clean, hexagonal architecture.

## Agent Capabilities

### 🚗 Core Features

- **Multi-Model Support**: PyTorch, ONNX, and OpenVINO inference engines
- **Real-time Detection**: Fast vehicle type classification with confidence scores
- **Flexible Architecture**: Hexagonal/Clean architecture with dependency injection
- **Production Ready**: Docker support, comprehensive testing, and monitoring
- **API-First**: RESTful API with FastAPI framework

### 🧠 AI Models

#### Supported Backends

1. **PyTorch (v2.8.0+)**
   - Native PyTorch inference
   - CUDA/CPU support
   - Dynamic model loading

2. **ONNX Runtime**
   - Cross-platform optimization
   - CPU/GPU execution providers
   - Standardized model format

3. **OpenVINO (v2025.3.0+)**
   - Intel hardware optimization
   - Advanced performance tuning
   - Edge deployment ready

#### Model Types

- **YOLO-based Detection**: Vehicle bounding box detection
- **Classification**: Vehicle type identification
- **Multi-class Output**: Car, Truck, Bus, Motorcycle, Bicycle

### 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Core Services │    │   AI Adapters   │
│   Controllers   │────│   & Business    │────│   (PyTorch,     │
│   (Routers)     │    │   Logic         │    │   ONNX, OpenVINO)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                    ┌─────────────────┐
                    │   Ports &       │
                    │   Interfaces    │
                    │   (Hexagonal)   │
                    └─────────────────┘
```

### 📡 API Endpoints

#### Version 2 (PyTorch)
- `GET /api/v2/health` - Health check
- `POST /api/v2/detect` - Vehicle detection
- `POST /api/v2/detect/annotated` - Detection with annotations
- `GET /api/v2/model/info` - Model information

#### Version 3 (OpenVINO)
- `GET /api/v3/health` - Health check
- `POST /api/v3/detect` - Optimized detection
- `POST /api/v3/detect/annotated` - Detection with annotations
- `GET /api/v3/model/info` - Model information
- `POST /api/v3/model/config` - Update model configuration

### 🔧 Configuration

#### Environment Variables

Key configuration options available in `.env`:

```bash
# Model Settings
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45
OPENVINO_DEVICE=CPU
TORCH_DEVICE=auto

# Performance
MAX_WORKERS=4
THREAD_POOL_SIZE=10

# API Settings
HOST=0.0.0.0
PORT=8000
```

#### Model Files

- `models/best.pt` - PyTorch model
- `models/best.onnx` - ONNX model
- `models/best_openvino_model/` - OpenVINO IR files
- `models/labels.txt` - Class labels

### 🚀 Usage Examples

#### Python SDK Usage

```python
from vehicle_type_detection.src.adapters.dependencies import (
    get_openvino_vehicle_object_detection_service
)

# Get service
service = get_openvino_vehicle_object_detection_service()

# Detect vehicles
with open("image.jpg", "rb") as f:
    result = await service.detect_vehicles_from_bytes(f.read())

print(f"Found {len(result['detections'])} vehicles")
```

#### API Usage

```bash
# Health check
curl -X GET "http://localhost:8000/api/v3/health"

# Vehicle detection
curl -X POST "http://localhost:8000/api/v3/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@vehicle_image.jpg" \
  -F "confidence_threshold=0.6"
```

#### Docker Usage

```bash
# Build and run
make docker-build
make docker-run

# Or with docker-compose
docker-compose up -d
```

### 🧪 Testing

#### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Service integration testing
- **API Tests**: Endpoint testing
- **Model Tests**: AI model validation

#### Running Tests

```bash
# Install test dependencies
pip install -r dev-requirements.txt

# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m api

# Run with coverage
pytest --cov=vehicle_type_detection --cov-report=html
```

### 📊 Performance Metrics

#### Inference Benchmarks

| Backend  | CPU (ms) | GPU (ms) | Memory (MB) | Model Size |
|----------|----------|----------|-------------|------------|
| PyTorch  | ~150-200 | ~50-80   | ~800-1200   | ~14MB      |
| ONNX     | ~100-150 | ~40-60   | ~600-900    | ~14MB      |
| OpenVINO | ~80-120  | ~35-50   | ~400-700    | ~7MB       |

*Benchmarks on Intel i7-10700K, RTX 3080, 640x640 input*

#### Throughput

- **Single Image**: 5-15 FPS depending on backend
- **Batch Processing**: Up to 30 FPS with optimizations
- **Concurrent Requests**: 10-50 requests/second

### 🛠️ Development

#### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/ilkerhalil/vehicle-type-detection-api.git
cd vehicle-type-detection-api

# Install dependencies
pip install -r requirements.txt
pip install -r dev-requirements.txt

# Setup pre-commit hooks
pre-commit install

# Run development server
make run-dev
```

#### Code Quality

- **Formatting**: Black, isort
- **Linting**: Ruff, flake8, mypy
- **Security**: Bandit
- **Testing**: pytest, coverage
- **Documentation**: Sphinx

### 🔍 Monitoring & Debugging

#### Health Checks

The agent provides comprehensive health monitoring:

```json
{
  "status": "healthy",
  "version": "3.0.0",
  "adapters": {
    "openvino": {
      "available": true,
      "ready": true,
      "model_type": "YOLO (OpenVINO)"
    }
  }
}
```

#### Logging

Structured logging with configurable levels:

```python
import logging
logger = logging.getLogger("vehicle_detection")
logger.info("Processing image", extra={"image_size": "640x640"})
```

#### Metrics

- Request/response times
- Model inference latency
- Memory usage
- Error rates

### 🚧 Limitations & Known Issues

#### Current Limitations

1. **Input Size**: Optimized for 640x640 images
2. **Batch Size**: Single image processing only
3. **Model Format**: Requires specific YOLO format
4. **Memory**: High memory usage with PyTorch backend

#### Roadmap

- [ ] Batch processing support
- [ ] Additional model architectures
- [ ] Real-time video processing
- [ ] Edge deployment optimizations
- [ ] Model quantization
- [ ] Multi-GPU support

### 🤝 Contributing

#### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Run quality checks: `make lint test`
5. Submit pull request

#### Code Standards

- Follow PEP 8 style guide
- Write comprehensive tests
- Document public APIs
- Use type hints
- Maintain test coverage >80%

### 📚 Additional Resources

#### Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ONNX Documentation](https://onnx.ai/onnx/)

#### Model Training

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Custom Dataset Training](docs/training.md)
- [Model Optimization Guide](docs/optimization.md)

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🆘 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/ilkerhalil/vehicle-type-detection-api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ilkerhalil/vehicle-type-detection-api/discussions)
- **Email**: contact@example.com

---

**Last Updated**: September 2025
**Version**: 3.0.0
**Maintainer**: Vehicle Detection Team