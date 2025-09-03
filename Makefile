.PHONY: install run test clean docker-build docker-run

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r dev-requirements.txt


# Clean cache and temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Format code
format:
	isort vehicle-type-detection/src vehicle-type-detection/tests
	black vehicle-type-detection/src vehicle-type-detection/tests

# Debug environment with conda volume
start-debug: down-debug
	docker compose -f docker-compose-debug.yml up --build -d

down-debug:
	docker compose -f docker-compose-debug.yml down

# Debug logs
logs-debug:
	docker compose -f docker-compose-debug.yml logs -f

# Lint code
lint:
	flake8 vehicle-type-detection/src/ vehicle-type-detection/tests/

complexity:
	complexipy vehicle-type-detection/src

# Show help
help:
	@echo "Available commands:"
	@echo "  install              - Install dependencies"
	@echo "  install-dev          - Install development dependencies"
	@echo "  start-debug          - Start debug environment with conda volume"
	@echo "  clean                - Clean cache and temporary files"
	@echo "  docker-build         - Build Docker image"
	@echo "  docker-run           - Run Docker container"
	@echo "  docker-compose-up    - Run with Docker Compose"
	@echo "  docker-compose-down  - Stop Docker Compose"
	@echo "  format               - Format code"
	@echo "  lint                 - Lint code"
	@echo "  help                 - Show this help"
