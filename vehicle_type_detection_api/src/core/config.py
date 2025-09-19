"""
Configuration settings for the Vehicle Detection API
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration"""

    # API Settings
    API_TITLE: str = "Vehicle Type Detection API"
    API_DESCRIPTION: str = "API for detecting vehicle types using AI model with Hexagonal Architecture"
    API_VERSION: str = "2.0.0"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    RELOAD: bool = True

    # Model Settings
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    MODEL_PATH: Path = PROJECT_ROOT / "models" / "best.pt"  # PyTorch YOLO model for v2

    # Image Processing Settings
    DEFAULT_IMAGE_SIZE: tuple[int, int] = (224, 224)  # ResNet18 standard input size
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB max file size
    ALLOWED_IMAGE_TYPES: list[str] = ["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"]

    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # CORS Settings
    ALLOW_ORIGINS: list[str] = ["*"]  # Configure based on your needs
    ALLOW_CREDENTIALS: bool = True
    ALLOW_METHODS: list[str] = ["*"]
    ALLOW_HEADERS: list[str] = ["*"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",  # Ignore extra fields from environment
    }

    def validate_model_files(self) -> bool:
        """Validate that required model files exist"""
        return self.MODEL_PATH.exists() and self.LABELS_PATH.exists()

    def get_model_info(self) -> dict:
        """Get model file information"""
        return {
            "model_path": str(self.MODEL_PATH),
            "labels_path": str(self.LABELS_PATH),
            "model_exists": self.MODEL_PATH.exists(),
            "labels_exist": self.LABELS_PATH.exists(),
            "model_size": self.MODEL_PATH.stat().st_size if self.MODEL_PATH.exists() else 0,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings factory function.

    This function uses @lru_cache to ensure Settings is instantiated only once
    and cached for subsequent calls, providing both performance benefits and
    singleton-like behavior for configuration.

    Returns:
        Settings: Cached settings instance
    """
    return Settings()


# Global settings instance (cached)
settings = get_settings()
