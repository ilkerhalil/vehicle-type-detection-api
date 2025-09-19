# Vehicle Type Detection API - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete Python project structure
- OpenVINO adapter implementation
- Comprehensive testing setup
- Development tools configuration

## [3.0.0] - 2025-09-18

### Added
- OpenVINO adapter for optimized inference
- API v3 endpoints with OpenVINO support
- Comprehensive project setup files
- Environment configuration templates
- Testing framework setup
- Development requirements

### Changed
- Updated API version to 3.0.0
- Enhanced model support (PyTorch, ONNX, OpenVINO)
- Improved documentation

### Fixed
- Import path issues
- Model loading consistency

## [2.0.0] - Previous Version

### Added
- PyTorch adapter
- ONNX adapter
- Hexagonal architecture
- FastAPI implementation
- Docker support

### Features
- Vehicle type detection
- Multiple model format support
- Clean architecture design
- Dependency injection

## [1.0.0] - Initial Release

### Added
- Basic vehicle detection functionality
- Initial API implementation

---

## Release Notes

### Version 3.0.0
This version introduces OpenVINO support for optimized inference performance, particularly beneficial for production deployments on Intel hardware. The addition provides:

- **Performance**: Faster inference times with OpenVINO optimization
- **Flexibility**: Support for multiple AI inference backends
- **Production Ready**: Enhanced configuration and testing setup

### Migration Guide

#### From 2.x to 3.x
- Update import paths if using the package programmatically
- Review new environment variables in `.env.example`
- Install OpenVINO dependencies: `pip install openvino`

#### API Changes
- New v3 endpoints available at `/api/v3/`
- Backward compatibility maintained for v2 endpoints
- Enhanced model configuration options

### Dependencies
- Python 3.8+
- FastAPI 0.116.1+
- OpenVINO 2025.3.0+
- PyTorch 2.8.0+
- Ultralytics 8.3.192+

### Breaking Changes
None in this release - fully backward compatible.

### Security Updates
- Updated dependencies to latest versions
- Enhanced security configurations

### Performance Improvements
- OpenVINO optimization for Intel hardware
- Improved model loading times
- Better memory management