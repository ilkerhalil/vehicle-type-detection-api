.PHONY: install run test clean docker-build docker-run dev-tools

# Install dependencies using uv
install:
	uv sync

# Install development dependencies using uv
install-dev:
	uv sync --dev

# Install development tools (uv-based linting, formatting, type checking)
dev-tools:
	uv tool install black
	uv tool install isort
	uv tool install flake8
	uv tool install mypy
	uv tool install pytest
	uv tool install pytest-cov
	uv tool install complexipy

# Clean cache and temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	uv cache clean

# Format code using uv tools
format:
	uv run isort vehicle_type_detection_api/src vehicle_type_detection_api/tests
	uv run black vehicle_type_detection_api/src vehicle_type_detection_api/tests
# Debug environment with conda volume
start-debug: down-debug
	docker compose -f docker-compose-debug.yml up --build -d

down-debug:
	docker compose -f docker-compose-debug.yml down

# Debug logs
logs-debug:
	docker compose -f docker-compose-debug.yml logs -f

# Lint code using uv tools
lint:
	uv run flake8 vehicle_type_detection_api/src/ vehicle_type_detection_api/tests/

complexity:
	uv run complexipy vehicle_type_detection_api/src

# Run tests using uv
test:
	uv run pytest tests/ -v

# Run tests with coverage
test-coverage:
	uv run pytest tests/ -v --cov=vehicle_type_detection_api --cov-report=xml

# Show help
help:
	@echo "Available commands (using uv):"
	@echo "  install              - Install dependencies (uv sync)"
	@echo "  install-dev          - Install development dependencies (uv sync --dev)"
	@echo "  dev-tools            - Install development tools via uv tool"
	@echo "  clean                - Clean cache and temporary files"
	@echo "  docker-build         - Build Docker image"
	@echo "  docker-run           - Run Docker container"
	@echo "  format               - Format code using uv tools"
	@echo "  lint                 - Lint code using uv tools"
	@echo "  test                 - Run tests using uv"
	@echo "  test-coverage        - Run tests with coverage using uv"
	@echo "  complexity           - Run complexity analysis using uv"
	@echo "  help                 - Show this help"
