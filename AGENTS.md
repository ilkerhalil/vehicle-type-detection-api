# Agent Guidelines for Vehicle Type Detection API

This document provides guidelines for AI agents (like Claude Code) working on the Vehicle Type Detection API project.

## Project Overview

The Vehicle Type Detection API is a FastAPI application for vehicle detection using AI models (PyTorch YOLOv8 and Intel OpenVINO) with Hexagonal Architecture.

## Development Setup

### Dependency Management
This project uses **UV** as its package manager. All dependency installation and management should be done using UV commands:

```bash
# Install dependencies
uv sync

# Install development dependencies
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Run code formatting
uv run black vehicle_type_detection_api/src vehicle_type_detection_api/tests
uv run isort vehicle_type_detection_api/src vehicle_type_detection_api/tests

# Run linting
uv run flake8 vehicle_type_detection_api/src vehicle_type_detection_api/tests

# Run type checking
uv run mypy vehicle_type_detection_api/src
```

### Environment
- Python 3.14+
- Dependencies managed via `pyproject.toml` and `uv.lock`
- Development tools configured in `pyproject.toml` (Black, ISort, Flake8, MyPy, etc.)

## Key Project Conventions

### Code Style
- **Formatting**: Black with line length 120 (configured in pyproject.toml)
- **Import Sorting**: ISort with Black profile
- **Linting**: Flake8 with max line length 120
- **Type Checking**: MyPy with Python 3.14 target
- **Security**: Bandit for security linting

### Architecture
Follows Hexagonal Architecture (Ports & Adapters pattern):
- **Core**: Business logic and interfaces (ports)
- **Adapters**: Implementations of ports (PyTorch, OpenVINO, image processing)
- **Entry Points**: FastAPI routes
- **Dependency Injection**: FastAPI Depends with singleton pattern for model loading

### Testing
- Tests located in `vehicle_type_detection_api/tests/`
- Run tests with: `uv run pytest tests/ -v`
- Test coverage configured to generate XML reports
- Mock external dependencies when testing services

## GitHub Actions Workflows

All GitHub Actions workflows have been updated to use UV for consistency:

1. **CI Workflow** (`.github/workflows/ci.yml`):
   - Installs dependencies using `uv sync --dev`
   - Runs tests with `uv run pytest`
   - Runs linting/formatting/type checking with `uv run <tool>`

2. **Test Workflow** (`.github/workflows/test.yml`):
   - Same UV-based dependency management and test execution

3. **Pre-commit Workflow** (`.github/workflows/pre-commit.yml`):
   - Installs pre-commit and hooks via UV
   - Runs pre-commit checks with `uv run pre-commit`

## Common Tasks

### Running the API
```bash
# Using Makefile (which uses UV internally)
make run-hexagonal

# Or directly with UV
uv run vehicle_type_detection_api.src.main
```

### Making Changes
1. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make small, focused changes following trunk-based development principles

3. Test locally:
   ```bash
   uv run pytest tests/ -v
   uv run black --check vehicle_type_detection_api/src vehicle_type_detection_api/tests
   uv run isort --check-only vehicle_type_detection_api/src vehicle_type_detection_api/tests
   uv run flake8 vehicle_type_detection_api/src vehicle_type_detection_api/tests
   uv run mypy vehicle_type_detection_api/src
   ```

4. Commit using conventional commits:
   ```bash
   git commit -m "feat: add new vehicle detection endpoint"
   ```

5. Push and create a pull request

### Debugging
- Use the debug Docker compose setup:
  ```bash
  make start-debug
  make logs-debug
  make down-debug
  ```

## Important Notes

1. **Always use UV** for dependency management - never use pip directly for project dependencies
2. **Pre-commit hooks** are configured and should pass before committing
3. **Type checking** with MyPy is required - ensure no type errors
4. **Line length** is 120 characters for Black and ISort (not the default 88)
5. **OpenVINO** requires special attention - ensure dependencies are properly resolved
6. **Testing** should always pass before considering work complete

## Troubleshooting

### OpenVINO Import Issues
If encountering OpenVINO import errors:
1. Ensure UV is used for installation (not pip)
2. Check that `uv sync --dev` was run to install all dependencies
3. Verify that the OpenVINO version in pyproject.toml matches what's available

### Dependency Conflicts
If experiencing dependency conflicts:
1. Check that `uv.lock` is up to date
2. Run `uv lock` to update the lock file if needed
3. Use `uv sync --dev` to reinstall from lock file

### Pre-commit Hook Failures
If pre-commit hooks fail:
1. Run `uv run pre-commit run --all-files` to see the specific errors
2. Fix the reported issues (formatting, linting, type checking, etc.)
3. Re-run the hooks to verify they pass