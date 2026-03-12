# Contributing to Vehicle Type Detection API

Thank you for your interest in contributing to our project! We use trunk-based development to ensure rapid integration and high-quality releases.

## Getting Started

1. Fork the repository
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/vehicle-type-detection-api.git
   cd vehicle-type-detection-api
   ```

3. Set up the development environment:
   ```bash
   make install
   ```

## Development Workflow

We follow [trunk-based development](TRUNK_BASED_DEVELOPMENT.md) practices. Here's how to contribute:

### Making Changes

1. **Start with fresh main**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create a topic branch for your work** (optional for trivial changes):
   ```bash
   git checkout -b topic/brief-description-of-change
   ```

3. **Make small, focused changes**:
   - Keep changes to what can be completed in <1 day
   - Write tests for new functionality
   - Ensure existing tests still pass

4. **Test locally**:
   ```bash
   # Run the test suite
   make test-hexagonal

   # Or run specific tests
   python -m pytest tests/ -v
   ```

5. **Commit frequently** using conventional commits:
   ```bash
   git commit -m "feat: add vehicle classification confidence scoring"
   ```

6. **Push your branch**:
   ```bash
   git push origin topic/brief-description-of-change
   ```

### Submitting Changes

1. Open a Pull Request against the `main` branch
2. Keep your PR focused and small (<50 lines ideal)
3. Include a clear description of what and why
4. Add testing instructions if not obvious
5. Request review from maintainers

### Review Process

1. Reviews should happen within hours, not days
2. Address all feedback promptly
3. Once approved, maintainers will squash and merge
4. Your branch will be deleted automatically after merge

## Code Style

We use several automated tools to maintain code quality:

- **Formatting**: Black (line length 88)
- **Import Sorting**: ISort (profile: black)
- **Linting**: Flake8 + Ruff
- **Type Checking**: MyPy
- **Security**: Bandit

You can run all checks locally:
```bash
# Install pre-commit hooks
pre-commit install

# Or run manually
pre-commit run --all-files
```

## Commit Message Format

Please follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect meaning (white-space, formatting, etc.)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

**Examples**:
- `feat: add confidence threshold parameter to detection endpoints`
- `fix: correct relative imports in model_service.py`
- `docs: update README with OpenVINO usage examples`
- `refactor: extract model loading logic to separate method`
- `test: add unit tests for image preprocessing functions`
- `chore: update dependencies to latest versions`

## Reporting Issues

When reporting bugs or suggesting features, please include:

1. Clear description of the issue or feature request
2. Steps to reproduce (for bugs)
3. Expected vs actual behavior
4. Environment details (Python version, OS, etc.)
5. Any relevant logs or error messages
6. For feature requests: why this would be valuable to users

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to reach out to maintainers or check existing issues and discussions for guidance.