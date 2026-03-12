# Trunk-Based Development Workflow

This document outlines the trunk-based development workflow for the Vehicle Type Detection API project.

## Overview

Trunk-based development is a software development practice where developers merge small, frequent updates to a single "trunk" (the main branch) rather than creating long-lived feature branches. This approach promotes continuous integration, reduces merge conflicts, and enables faster feedback cycles.

## Core Principles

1. **Single Source of Truth**: The `main` branch is always deployable
2. **Small Batches**: Changes are small and frequent (typically <1 day of work)
3. **Fast Integration**: Developers integrate with trunk at least once per day
4. **Automated Testing**: Comprehensive CI pipeline validates every change
5. **Branch by Abstraction**: For larger changes, use feature flags rather than long-lived branches

## Workflow Practices

### Daily Development Flow

1. **Start with Fresh Main**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create Short-Lived Topic Branch (Optional)**
   For very small changes, you can commit directly to main after review.
   For slightly larger changes that need preliminary review:
   ```bash
   git checkout -b topic/brief-description
   # Make small changes (<1 day of work)
   git commit -m "feat: implement small feature"
   git push origin topic/brief-description
   ```

3. **Open Pull Request**
   - Keep PRs small and focused
   - Target branch: `main`
   - Include clear description and testing instructions
   - Request review from 1 team member

4. **Review and Merge Quickly**
   - Review should happen within hours, not days
   - Address feedback promptly
   - Once approved, squash and merge to maintain clean history
   - Delete topic branch after merge

5. **Continuous Integration**
   - Every push to main triggers CI pipeline
   - Main branch must always pass all tests
   - If CI fails, fix immediately (priority over new work)

### When to Use Direct Commits to Main

For trivial changes that don't require review:
- Documentation fixes
- Typos and minor text changes
- Dependency version bumps
- Obvious bug fixes with clear test evidence

Even for direct commits:
- Ensure CI passes before considering work complete
- Follow conventional commit format
- Keep changes small and focused

### Handling Larger Features

For changes that will take more than 1 day:

1. **Feature Flags/Toggles**: Implement the feature behind a flag
2. **Branch by Abstraction**: Build the new implementation alongside the old one
3. **Strangler Pattern**: Gradually replace old functionality with new
4. **Short-lived Integration Branches**: If absolutely necessary, create integration branches that live for <2 days

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring
- `perf`: Performance improvements
- `test`: Adding or correcting tests
- `chore`: Maintenance tasks

Examples:
- `feat: add OpenVINO support for vehicle detection`
- `fix: correct relative imports in model_service.py`
- `docs: update README with usage examples`
- `refactor: extract model loading logic to separate method`
- `test: add unit tests for image preprocessing`
- `chore: update dependencies to latest versions`

## Benefits of Trunk-Based Development

1. **Reduced Merge Conflicts**: Small, frequent integrations prevent divergence
2. **Faster Feedback**: Issues are discovered and fixed quickly
3. **Continuous Deployment**: Main branch is always ready for release
4. **Better Collaboration**: Everyone sees changes as they happen
5. **Simplified Release Process**: No complex release branching needed
6. **Improved Code Quality**: Continuous integration catches issues early

## Project-Specific Guidelines

### For Vehicle Type Detection API

1. **Model Changes**: Treat model updates as infrastructure changes
   - Test thoroughly with both PyTorch and OpenVINO adapters
   - Ensure backward compatibility where possible
   - Use feature flags for major model architecture changes

2. **API Changes**:
   - Maintain backward compatibility for at least one release cycle
   - Add new endpoints rather than modifying existing ones when possible
   - Deprecate old endpoints with clear timelines

3. **Configuration Changes**:
   - Validate configuration changes don't break existing deployments
   - Provide clear migration paths for configuration updates
   - Use environment variables for environment-specific settings

4. **Database/Storage Changes**:
   - Not applicable for current version (uses in-memory/storage adapters)
   - If added, use migration scripts with backward compatibility

## CI/CD Integration

Our GitHub Actions workflows support trunk-based development by:

1. **Fast Feedback**: CI runs on every push to any branch
2. **Main Branch Protection**:
   - Requires status checks to pass before merging
   - Requires pull request reviews
   - Prevents force pushes
3. **Quality Gates**:
   - All tests must pass
   - Code coverage thresholds (if implemented)
   - Linting and type checking must pass
4. **Deployment Readiness**:
   - Main branch is always in a deployable state
   - Docker images can be built from any passing commit on main

## Best Practices

### Do:
- Make changes small and focused
- Commit early and often
- Run tests locally before pushing
- Keep PRs under 50 lines when possible
- Review others' PRs promptly
- Delete topic branches after merging
- Use `git fetch --prune` regularly to clean up remote tracking branches

### Don't:
- Let PRs linger for days without review
- Make large, sweeping changes in a single PR
- Ignore CI failures on main branch
- Commit debug or temporary code to main
- Create long-lived feature branches without justification

## Transitioning to Trunk-Based Development

If coming from a feature-branch workflow:

1. Start with small changes to get comfortable with the flow
2. Practice making changes that can be reviewed in <1 hour
3. Learn to use feature flags for experimental work
4. Trust the CI pipeline to catch issues
5. Embrace the discipline of small batches

## References

- [Trunk Based Development](https://trunkbaseddevelopment.com/)
- [Google's Engineering Practices](https://testing.googleblog.com/2016/08/test-driven-development-at-google.html)
- [Trunk Based Development vs GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-vs-trunk-based-development)

---

*Adopted for Vehicle Type Detection API - Main branch is always releasable*