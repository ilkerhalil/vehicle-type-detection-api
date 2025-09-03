# Vehicle Type Detection API

Modern bir FastAPI uygulaması ile araç türü tespiti yapan AI destekli servis. **Hexagonal Architecture** tasarım deseni kullanarak **PyTorch** ve **OpenVINO** engine'lerini destekler.

## 🚀 Özellikler

- **🎯 Hexagonal Architecture**: Clean Architecture ile temiz kod yapısı
- **🤖 Çoklu AI Engine**: PyTorch (YOLOv8) ve Intel OpenVINO desteği
- **⚡ Singleton Pattern**: Thread-safe model yükleme optimizasyonu
- **💉 Dependency Injection**: FastAPI Depends ile temiz bağımlılık yönetimi
- **⚙️ Cached Settings**: Pydantic Settings ile performans optimizasyonu
- **🎪 FastAPI 2.0**: Modern RESTful API framework
- **📋 5 Araç Türü**: Car, Motorcycle, Truck, Bus, Bicycle
- **🐳 Docker Desteği**: Konteyner tabanlı deployment
- **📚 Swagger/ReDoc**: Otomatik API dokümantasyonu

## 🤖 Desteklenen AI Engines

### PyTorch (Ultralytics YOLOv8)
- **Model**: `models/best.pt`
- **Backend**: PyTorch + Ultralytics
- **Avantaj**: Yüksek doğruluk, GPU desteği
- **Endpoints**:
  - `/api/v2/pytorch/detect` - JSON sonuç
  - `/api/v2/pytorch/detect/annotated` - Annotated görsel

### OpenVINO (Intel Optimized)
- **Model**: `models/best_openvino_model/`
- **Backend**: Intel OpenVINO Runtime
- **Avantaj**: CPU optimizasyonu, hızlı inference
- **Endpoints**:
  - `/api/v2/openvino/detect` - JSON sonuç
  - `/api/v2/openvino/detect/annotated` - Annotated görsel

## 🛠️ Hızlı Başlangıç

### 1. Bağımlılıkları Yükle

```bash
make install
```

### 2. API'yi Çalıştır

```bash
make run-hexagonal
```

### 3. API'yi Test Et

```bash
# Health check
curl http://localhost:8000/api/v2/health

# OpenVINO ile tespit
curl -X POST "http://localhost:8000/api/v2/openvino/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg"

# PyTorch ile tespit
curl -X POST "http://localhost:8000/api/v2/pytorch/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg"
```

### 4. API'yi Durdur

```bash
make stop-hexagonal
```

API: http://localhost:8000 | Docs: http://localhost:8000/docs

## 📡 API Endpoints

### Ana Endpoints
- **GET** `/` - API bilgileri ve versiyon detayları
- **GET** `/api/v2/health` - Tüm engine'lerin durumu
- **GET** `/api/v2/ready` - Hazır engine'lerin listesi

### PyTorch Engine
- **GET** `/api/v2/pytorch/classes` - Desteklenen sınıflar
- **POST** `/api/v2/pytorch/detect` - Araç tespiti (JSON)
- **POST** `/api/v2/pytorch/detect/annotated` - Annotated görsel

### OpenVINO Engine
- **GET** `/api/v2/openvino/classes` - Desteklenen sınıflar
- **POST** `/api/v2/openvino/detect` - Araç tespiti (JSON)
- **POST** `/api/v2/openvino/detect/annotated` - Annotated görsel

## 💻 Kullanım Örnekleri

### 1. Health Check

```bash
curl http://localhost:8000/api/v2/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
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

### 2. OpenVINO ile Araç Tespiti

```bash
curl -X POST "http://localhost:8000/api/v2/openvino/detect?confidence_threshold=0.5" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg"
```

### 3. PyTorch ile Araç Tespiti

```bash
curl -X POST "http://localhost:8000/api/v2/pytorch/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg"
```

### 4. Annotated Görsel Alma

```bash
# OpenVINO annotated
curl -X POST "http://localhost:8000/api/v2/openvino/detect/annotated" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg" \
     --output annotated_openvino.jpg

# PyTorch annotated
curl -X POST "http://localhost:8000/api/v2/pytorch/detect/annotated" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg" \
     --output annotated_pytorch.jpg
```

### 5. Python ile Kullanım

```python
import requests

# OpenVINO ile araç tespiti
with open('vehicle_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/api/v2/openvino/detect',
        files=files,
        params={'confidence_threshold': 0.5}
    )
    result = response.json()
    print(f"Tespit edilen araç sayısı: {len(result['detections'])}")
```

## 🎯 Mimari Tasarım

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

### Temel Prensipler

- **🔌 Ports & Adapters**: Interface'ler ve implementasyonlar ayrı
- **💉 Dependency Injection**: FastAPI Depends ile temiz bağımlılık yönetimi
- **⚡ Singleton Pattern**: Thread-safe model optimizasyonu
- **⚙️ Configuration Caching**: `@lru_cache` ile settings optimizasyonu
- **🧪 Testability**: Mock'lanabilir interface'ler
- **🔄 Separation of Concerns**: Her katman tek sorumluluğa sahip

## 📊 Response Formatları

### Araç Tespiti Yanıtı

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

## 🛠️ Makefile Komutları

| Komut | Açıklama |
|-------|----------|
| `make install` | Bağımlılıkları yükle |
| `make install-dev` | Geliştirme bağımlılıkları yükle |
| `make run-hexagonal` | API'yi çalıştır (foreground) |
| `make run-hexagonal-bg` | API'yi arka planda çalıştır |
| `make stop-hexagonal` | Arka plan API'sini durdur |
| `make test-quick` | Hızlı curl tabanlı test |
| `make test-hexagonal` | Detaylı Python tabanlı test |

## 💡 Desteklenen Özellikler

### Görüntü Formatları
- JPEG, PNG, BMP, TIFF

### Araç Türleri (5 Sınıf)
- **Car**: Arabalar
- **Motorcycle**: Motosikletler
- **Truck**: Kamyonlar
- **Bus**: Otobüsler
- **Bicycle**: Bisikletler

### Teknik Gereksinimler
- Python 3.8+
- PyTorch 2.0+
- OpenVINO 2025.3.0+
- OpenCV 4.0+
- FastAPI 2.0+
- Pydantic Settings 2.4.0+
- Uvicorn

## 🧪 Test Komutları

### Otomatik Test

```bash
# Hızlı curl tabanlı test
make test-quick

# Detaylı Python tabanlı test
make test-hexagonal
```

### Manuel Test

```bash
# API bilgileri
curl http://localhost:8000/

# Sağlık kontrolü
curl http://localhost:8000/api/v2/health

# OpenVINO ile test
curl -X POST "http://localhost:8000/api/v2/openvino/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/27.jpg"
```

## 🐳 Docker ile Çalıştırma

### 1. Docker Image Oluşturma

```bash
docker build -t vehicle-detection-api .
```

### 2. Container Çalıştırma

```bash
docker run -p 8000:8000 vehicle-detection-api
```

### 3. Docker Compose ile

```bash
docker-compose up -d
```

## 📁 Proje Yapısı

```
vehicle-type-detection-api/
├── 📄 Makefile                    # Automation commands
├── 🐳 Dockerfile                  # Container configuration
├── 🐳 docker-compose.yml          # Multi-container setup
├── 📦 requirements.txt            # Python dependencies
├── 📦 dev-requirements.txt        # Development dependencies
├── 📖 README.md                   # Bu dosya
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
        │   └── detection_v2.py    # v2 API routes (all engines)
        └── ⚙️ core/              # Configuration layer
            ├── config.py          # App configuration
            ├── injection.py       # Dependency injection setup
            └── logger.py          # Logging setup
```

## 🔧 Sorun Giderme

### Model İlgili Sorunlar
- **PyTorch model bulunamıyor**: `models/best.pt` dosyasının mevcut olduğundan emin olun
- **OpenVINO model bulunamıyor**: `models/best_openvino_model/` klasörünün mevcut olduğundan emin olun
- **Labels dosyası bulunamıyor**: `models/labels.txt` dosyasının mevcut olduğundan emin olun
- **OpenVINO Runtime hatası**: `openvino` paketinin doğru yüklendiğini kontrol edin

### API İlgili Sorunlar
- **Port zaten kullanımda**: Port 8000'in başka uygulama tarafından kullanılmadığından emin olun
- **Import hatası**: `make install` ile bağımlılıkları yeniden yükleyin
- **Singleton hatası**: API'yi yeniden başlatın: `make stop-hexagonal && make run-hexagonal`

### Test İlgili Sorunlar
- **Test başarısız**: API'nin çalıştığından emin olun: `curl http://localhost:8000/api/v2/health`
- **Connection refused**: API'nin başlatılmasını bekleyin (5-10 saniye)
- **Model loading errors**: İlgili model dosyalarının varlığını kontrol edin

## 🎯 Geliştirme Notları

### Mimari Avantajları
- **Clean Code**: Hexagonal Architecture ile temiz kod yapısı
- **Multi-Engine Support**: PyTorch ve OpenVINO desteği
- **Testability**: Mock'lanabilir interface'ler
- **Maintainability**: Katmanlar arası düşük bağımlılık
- **Scalability**: Yeni adapter'lar kolayca eklenebilir
- **Performance**: Singleton pattern ve cached settings ile optimizasyon

### Dependency Injection
- FastAPI Depends kullanılır
- Thread-safe singleton implementasyonu
- Cached settings ile performans artışı (`@lru_cache`)
- Pydantic Settings ile environment variables desteği
- Interface'ler üzerinden gevşek bağlılık
- Mock testler için kolay değiştirilebilirlik

### Engine Karşılaştırması

| Özellik | PyTorch | OpenVINO |
|---------|---------|----------|
| **Doğruluk** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Hız** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **CPU Optimizasyonu** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **GPU Desteği** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Kurulum Kolaylığı** | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 📄 Lisans

Bu proje MIT lisansı altında açık kaynak kodludur.

---

**🚀 Vehicle Type Detection API v2.0 - PyTorch + OpenVINO Destekli Modern AI Servisi**
