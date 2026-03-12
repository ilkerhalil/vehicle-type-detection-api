# UV Workspace Setup

This document explains how to set up and use the UV workspace for the Vehicle Type Detection API project.

## Why UV?

We're using [Astral's UV](https://github.com/astral-sh/uv) as our Python package manager and resolver because it's:
- Extremely fast (typically 10-100x faster than pip)
- Reliable and deterministic
- Drop-in replacement for pip/virtualenv/poetry
- Supports modern Python packaging standards

## Prerequisites

1. Install UV: Follow the instructions at https://docs.astral.sh/uv/getting-started/installation/
2. Ensure you have Python 3.14+ installed (UV will manage this for you)

## Setup Instructions

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/ilkerhalil/vehicle-type-detection-api.git
cd vehicle-type-detection-api

# Install dependencies using UV
uv sync

# Install development dependencies
uv sync --dev

# Install development tools (linting, formatting, type checking)
uv tool install black isort flake8 mypy
```

### Activating the Environment

```bash
# Activate the virtual environment
source .venv/bin/activate

# Or run commands directly with UV
uv run python -m pytest tests/
uv run black --check src/
uv run isort --check-only src/
uv run flake8 src/
uv run mypy src/
```

## Available Commands

### Dependency Management
- `uv sync` - Install production dependencies
- `uv sync --dev` - Install production + development dependencies
- `uv add <package>` - Add a new dependency
- `uv add --dev <package>` - Add a new development dependency
- `uv remove <package>` - Remove a dependency
- `uv lock` - Update the lock file

### Running the Application
```bash
# Run the API server
uv run uvicorn vehicle_type_detection_api.src.main:app --reload

# Or use the convenience script
uv run vehicle-type-detection-api
```

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=vehicle_type_detection_api --cov-report=xml

# Run specific test
uv run pytest tests/test_adapters.py -v
```

### Code Quality
```bash
# Format code
uv run black src/ tests/

# Sort imports
uv run isort src/ tests/

# Lint code
uv run flake8 src/ tests/

# Type checking
uv run mypy src/ --ignore-missing-imports
```

### Docker
```bash
# Build Docker image
docker build -t vehicle-type-detection-api:dev .

# Run container
docker run -p 8000:8000 vehicle-type-detection-api:dev
```

## Troubleshooting

### Common Issues

1. **"No module named 'xyz'" errors for heavy dependencies (OpenVINO, PyTorch, etc.)**
   - These are expected in the basic development environment
   - Install them separately if needed: `uv add openvino torch torchvision`
   - Or use the full dev setup: `uv sync --dev`

2. **Import errors when running tests**
   - Make sure you're using the UV environment: `source .venv/bin/activate`
   - Or prefix commands with `uv run`: `uv run pytest tests/`

3. **Conflicts with system Python packages**
   - UV creates an isolated environment at `.venv/`
   - Always use `uv run` or activate the venv to avoid conflicts

## Configuration Files

- `pyproject.toml` - Main project configuration and dependencies
- `.python-version` - Specifies Python 3.14 (used by various tools)
- `requirements.lock` - Frozen dependency tree for reproducible builds
- `.github/workflows/` - GitHub Actions CI/CD pipelines using UV

## IDE Support

### VS Code
1. Install the Python extension
2. UV should be automatically detected
3. Select the interpreter from `.venv/bin/python`

### PyCharm
1. Go to Settings → Project → Python Interpreter
2. Add → Existing environment → `.venv/bin/python`

## CI/CD Integration

Our GitHub Actions workflows in `.github/workflows/` are configured to use UV:
- Dependency installation via `uv sync`
- Testing via `uv run pytest`
- Code quality checks via `uv run black`, `uv run isort`, etc.
- All jobs run on Python 3.14 as specified in the workflows

---

*Happy coding with UV! 🚀*