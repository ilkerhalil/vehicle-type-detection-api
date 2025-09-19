"""
Setup script for Vehicle Type Detection API
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()
                if line.strip() and not line.startswith('#')]

# Read version from __init__.py
def get_version():
    """Get version from __init__.py"""
    init_file = this_directory / "__init__.py"
    with open(init_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="vehicle-type-detection-api",
    version=get_version(),
    author="Vehicle Detection Team",
    author_email="contact@example.com",
    description="API for detecting vehicle types using AI model with Hexagonal Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ilkerhalil/vehicle-type-detection-api",
    project_urls={
        "Bug Tracker": "https://github.com/ilkerhalil/vehicle-type-detection-api/issues",
        "Documentation": "https://github.com/ilkerhalil/vehicle-type-detection-api#readme",
        "Source Code": "https://github.com/ilkerhalil/vehicle-type-detection-api",
    },
    packages=find_packages(exclude=["tests*", "docs*"]),
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json"],
        "models": ["*.pt", "*.onnx", "*.xml", "*.bin", "*.mapping"],
        "samples": ["*.jpg", "*.jpeg", "*.png"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: FastAPI",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("dev-requirements.txt"),
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "httpx>=0.24.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vehicle-detection-api=vehicle_type_detection.src.main:main",
            "convert-to-openvino=convert_to_openvino:main",
        ],
    },
    keywords=[
        "vehicle detection",
        "computer vision",
        "AI",
        "machine learning",
        "fastapi",
        "pytorch",
        "openvino",
        "onnx",
        "yolo",
        "object detection"
    ],
    zip_safe=False,
)