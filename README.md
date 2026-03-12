# Vehicle Type Detection API

AI-powered vehicle type detection service built with modern FastAPI application. Supports **PyTorch** and **OpenVINO** engines using Hexagonal Architecture design pattern.

## 🚀 Features

- **🎯 Hexagonal Architecture**: Clean code structure with Clean Architecture
- **🤖 Multi AI Engine**: PyTorch (YOLOv8) and Intel OpenVINO support
- **⚡ Singleton Pattern**: Thread-safe model loading optimization
- **💉 Dependency Injection**: Clean dependency management with FastAPI Depends
- **⚙️ Cached Settings**: Performance optimization with Pydantic Settings
- **🎪 FastAPI 2.0**: Modern RESTful API framework
- **📋 5 Vehicle Types**: Car, Motorcycle, Truck, Bus, Bicycle
- **🐳 Docker Support**: Container-based deployment
- **📚 Swagger/ReDoc**: Automatic API documentation

## 🤖 Supported AI Engines

### PyTorch (Ultralytics YOLOv8)
- **Model**: `models/best.pt`
- **Backend**: PyTorch + Ultralytics
- **Advantage**: High accuracy, GPU support
- **Endpoints**:
  - `/api/v1/pytorch/detect` - JSON response
  - `/api/v1/pytorch/detect/annotated` - Annotated image

### OpenVINO (Intel Optimized)
- **Model**: `models/best_openvino_model/`
- **Backend**: Intel OpenVINO Runtime
- **Advantage**: CPU optimization, fast inference
- **Endpoints**:
  - `/api/v1/openvino/detect` - JSON response
  - `/api/v1/openvino/detect/annotated` - Annotated image

## 🛠️ Quick Start

### 1. Install Dependencies

```bash
make install
```

### 2. Run the API

```bash
make run-hexagonal
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# OpenVINO detection
curl -X POST "http://localhost:8000/api/v1/openvino/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg"

# PyTorch detection
curl -X POST "http://localhost:8000/api/v1/pytorch/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg"
```

### 4. Stop the API

```bash
make stop-hexagonal
```

API: http://localhost:8000 | Docs: http://localhost:8000/docs

## 📡 API Endpoints

### Main Endpoints
- **GET** `/` - API information and version details
- **GET** `/api/v1/health` - Status of all engines
- **GET** `/api/v1/ready` - List of ready engines

### PyTorch Engine
- **GET** `/api/v1/pytorch/classes` - Supported classes
- **POST** `/api/v1/pytorch/detect` - Vehicle detection (JSON)
- **POST** `/api/v1/pytorch/detect/annotated` - Annotated image

### OpenVINO Engine
- **GET** `/api/v1/openvino/classes` - Supported classes
- **POST** `/api/v1/openvino/detect` - Vehicle detection (JSON)
- **POST** `/api/v1/openvino/detect/annotated` - Annotated image

## 💻 Usage Examples

### 1. Health Check

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "engines": ["PyTorch", "OpenVINO"],
  "adapters": {
    "pytorch": {
      "available": true,
      "ready": true,
      "model_type": "PyTorch YOLO"
    },
    "openvino": {
      "available": true,
      "ready": true,
      "model_type": "OpenVINO YOLO"
    }
  }
}
```

### 2. Vehicle Detection with OpenVINO

```bash
curl -X POST "http://localhost:8000/api/v1/openvino/detect?confidence_threshold=0.5" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg"
```

### 3. Vehicle Detection with PyTorch

```bash
curl -X POST "http://localhost:8000/api/v1/pytorch/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg"
```

### 4. Get Annotated Image

```bash
# OpenVINO annotated
curl -X POST "http://localhost:8000/api/v1/openvino/detect/annotated" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg" \
     --output annotated_openvino.jpg

# PyTorch annotated
curl -X POST "http://localhost:8000/api/v1/pytorch/detect/annotated" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg" \
     --output annotated_pytorch.jpg
```

### 5. Usage with Python

```python
import requests

# Vehicle detection with OpenVINO
with open('vehicle_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/api/v1/openvino/detect',
        files=files,
        params={'confidence_threshold': 0.5}
    )
    result = response.json()
    print(f"Number of detected vehicles: {len(result['detections'])}")
```

## 🎯 Architecture Design

### Hexagonal Architecture

```
┌─────────────────────────────────────────────┐
│             FastAPI Routes                  │
│    /pytorch/detect  /openvino/detect        │
├─────────────────────────────────────────────┤
│           Services Layer                    │
│      VehicleObjectDetectionService          │
├─────────────────────────────────────────────┤
│    Ports (Interfaces)    │   Adapters      │
│  • VehicleDetectionPort  │ • PyTorch       │
│  • ImageProcessingPort   │ • OpenVINO      │
│                          │ • Image Adapter │
└─────────────────────────────────────────────┘
```

### Core Principles

- **🔌 Ports & Adapters**: Separate interfaces and implementations
- **💉 Dependency Injection**: Clean dependency management with FastAPI Depends
- **⚡ Singleton Pattern**: Thread-safe model optimization
- **⚙️ Configuration Caching**: Settings optimization with `@lru_cache`
- **🧪 Testability**: Mockable interfaces
- **🔄 Separation of Concerns**: Each layer has single responsibility

## 📊 Response Formats

### Vehicle Detection Response

```json
{
  "total_detections": 2,
  "vehicle_detections": [
    {
      "class_id": 0,
      "class_name": "Car",
      "confidence": 0.8883,
      "bbox": {
        "x1": 441.28,
        "y1": 81.21,
        "x2": 558.72,
        "y2": 230.75
      }
    },
    {
      "class_id": 1,
      "class_name": "Motorcycle",
      "confidence": 0.7654,
      "bbox": {
        "x1": 120.5,
        "y1": 150.2,
        "x2": 180.8,
        "y2": 220.1
      }
    }
  ],
  "image_info": {
    "width": 720,
    "height": 1280,
    "channels": 3
  },
  "model_info": {
    "engine": "OpenVINO",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
  }
}
```

## 🛠️ Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make install-dev` | Install development dependencies |
| `make run-hexagonal` | Run API (foreground) |
| `make run-hexagonal-bg` | Run API in background |
| `make stop-hexagonal` | Stop background API |
| `make test-quick` | Quick curl-based test |
| `make test-hexagonal` | Detailed Python-based test |

## 💡 Supported Features

### Image Formats
- JPEG, PNG, BMP, TIFF

### Vehicle Types (5 Classes)
- **Car**: Cars
- **Motorcycle**: Motorcycles
- **Truck**: Trucks
- **Bus**: Buses
- **Bicycle**: Bicycles

### Technical Requirements
- Python 3.8+
- PyTorch 2.0+
- OpenVINO 2025.3.0+
- OpenCV 4.0+
- FastAPI 2.0+
- Pydantic Settings 2.4.0+
- Uvicorn

## 🧪 Test Commands

### Automated Testing

```bash
# Quick curl-based test
make test-quick

# Detailed Python-based test
make test-hexagonal
```

### Manual Testing

```bash
# API information
curl http://localhost:8000/

# Health check
curl http://localhost:8000/api/v1/health

# Test with OpenVINO
curl -X POST "http://localhost:8000/api/v1/openvino/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg"
```

## 🐳 Running with Docker

### 1. Build Docker Image

```bash
docker build -t vehicle-detection-api .
```

### 2. Run Container

```bash
docker run -p 8000:8000 vehicle-detection-api
```

### 3. With Docker Compose

```bash
docker-compose up -d
```

## 📁 Project Structure

```
vehicle_type_detection_api/
├── 📄 Makefile                    # Automation commands
├── 🐳 Dockerfile                  # Container configuration
├── 🐳 docker-compose.yml          # Multi-container setup
├── 📦 requirements.txt            # Python dependencies
├── 📦 dev-requirements.txt        # Development dependencies
├── 📖 README.md                   # This file
├── 🤖 models/                     # AI model files
│   ├── best.pt                    # PyTorch YOLOv8 model
│   ├── best_openvino_model/       # OpenVINO IR model
│   │   ├── best.xml               # Model structure
│   │   ├── best.bin               # Model weights
│   │   └── metadata.yaml          # Model metadata
│   └── labels.txt                 # Class labels (5 classes)
├── 📷 samples/                    # Test images
└── 🏗️ vehicle-type-detection/    # Main application
    └── src/
        ├── 🚀 main.py             # FastAPI entry point
        ├── 🔌 adapters/           # Hexagonal Architecture
        │   ├── 📋 ports.py        # Interface definitions
        │   ├── 🎯 torch_yolo_adapter.py    # PyTorch implementation
        │   ├── 🎯 openvino_adapter.py      # OpenVINO implementation
        │   ├── 🖼️ image_adapter.py         # OpenCV image processing
        │   └── 💉 dependencies.py          # FastAPI Depends providers
        ├── 🏗️ services/           # Business logic layer
        │   ├── detection_service.py  # Detection orchestration
        │   └── model_service.py      # Model management
        ├── 🌐 routers/            # API endpoints
        │   └── detect.py          # v1 API routes (all engines)
        └── ⚙️ core/              # Configuration layer
            ├── config.py          # App configuration
            ├── injection.py       # Dependency injection setup
            └── logger.py          # Logging setup
```

## 🔧 Troubleshooting

### Model-Related Issues
- **PyTorch model not found**: Ensure `models/best.pt` file exists
- **OpenVINO model not found**: Ensure `models/best_openvino_model/` directory exists
- **Labels file not found**: Ensure `models/labels.txt` file exists
- **OpenVINO Runtime error**: Check that `openvino` package is correctly installed

### API-Related Issues
- **Port already in use**: Ensure port 8000 is not used by another application
- **Import error**: Reinstall dependencies with `make install`
- **Singleton error**: Restart API: `make stop-hexagonal && make run-hexagonal`

### Test-Related Issues
- **Test failed**: Ensure API is running: `curl http://localhost:8000/api/v1/health`
- **Connection refused**: Wait for API to start (5-10 seconds)
- **Model loading errors**: Check existence of related model files

## 🎯 Development Notes

### Architecture Advantages
- **Clean Code**: Clean code structure with Hexagonal Architecture
- **Multi-Engine Support**: PyTorch and OpenVINO support
- **Testability**: Mockable interfaces
- **Maintainability**: Low coupling between layers
- **Scalability**: New adapters can be easily added
- **Performance**: Optimization with singleton pattern and cached settings

### Dependency Injection
- Uses FastAPI Depends
- Thread-safe singleton implementation
- Performance boost with cached settings (`@lru_cache`)
- Environment variables support with Pydantic Settings
- Loose coupling through interfaces
- Easy substitutability for mock tests

### Engine Comparison

| Feature | PyTorch | OpenVINO |
|---------|---------|----------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **CPU Optimization** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **GPU Support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Installation Ease** | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 📄 License

This project is open source under the MIT license.

---

**🚀 Vehicle Type Detection API v2.0 - PyTorch + OpenVINO Supported Modern AI Service**