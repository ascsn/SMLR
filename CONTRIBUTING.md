# Contributing to SMLR

Thank you for your interest in contributing to SMLR! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SMLR.git
   cd SMLR
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ascsn/SMLR.git
   ```

## Development Setup

We recommend using `uv` for fast, reproducible development environments:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create development environment
uv sync --group dev --group docs

# Activate the virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

Alternatively, using standard Python tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev,docs]
```

## How to Contribute

### Reporting Bugs

- **Check existing issues** to avoid duplicates
- **Use the bug report template** (if available)
- **Include**:
  - Python version and OS
  - Minimal reproducible example
  - Expected vs. actual behavior
  - Full error traceback

### Suggesting Features

- **Open an issue** with the `enhancement` label
- **Describe the use case** clearly
- **Provide examples** of how the feature would be used

### Contributing Code

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, readable code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**:
   ```bash
   pytest
   ```

4. **Check code formatting**:
   ```bash
   ruff check src/ tests/
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```
   
   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions/changes
   - `refactor:` for code refactoring
   - `perf:` for performance improvements

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

## Coding Standards

### Python Style

- Follow **PEP 8** conventions
- Use **type hints** for function signatures
- Maximum line length: **100 characters**
- Use **descriptive variable names**

### Code Organization

- Keep functions focused and single-purpose
- Use docstrings for all public functions/classes
- Organize imports: standard library → third-party → local

### Example Docstring Format

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief one-line summary.

    More detailed description if needed. Explain the purpose,
    behavior, and any important notes.

    Parameters
    ----------
    param1 : Type1
        Description of param1.
    param2 : Type2
        Description of param2.

    Returns
    -------
    ReturnType
        Description of return value.

    Examples
    --------
    >>> function_name(value1, value2)
    expected_result
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=smlr --cov-report=html

# Run specific test file
pytest tests/test_emulator.py

# Run specific test
pytest tests/test_emulator.py::test_emulator_interpolates_spectrum
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common setup
- Aim for >90% code coverage
- Test edge cases and error conditions

### Test Requirements

- Tests must pass before merging
- New features require new tests
- Bug fixes should include regression tests
- Use deterministic random seeds for reproducibility

## Documentation

### Building Documentation Locally

The documentation includes automatically generated example metrics and plots. When you build the docs, 
all example scripts are run to capture current performance metrics.

```bash
# Quick method: Serve docs with live reload (runs examples automatically)
uv run mkdocs serve

# Manual method: Generate metrics and build separately
python scripts/generate_example_metrics.py
uv run mkdocs build

# Convenience script: One command to do everything
./scripts/build_docs.sh
```

**Note:** Building docs takes several minutes as it runs all example scripts. The metrics are automatically
captured and inserted into the documentation using Jinja templates.

### How Documentation Auto-Generation Works

1. **Pre-build hook** (`docs/hooks.py`) runs before MkDocs builds
2. **Metrics script** (`scripts/generate_example_metrics.py`) executes all examples
3. **Performance data** is captured and saved to `docs/example_metrics.json`
4. **Jinja templates** in the markdown files use `{{ metrics.*.* }}` syntax
5. **Final docs** show current, accurate performance metrics

### Adding New Examples

To add a new example that auto-generates metrics:

1. Create the example script in `examples/`
2. Add metric extraction function in `scripts/generate_example_metrics.py`
3. Update documentation markdown with Jinja template syntax:
   ```markdown
   {% if metrics and metrics.your_example %}
   | Metric | Value |
   |--------|-------|
   | Error | {{ metrics.your_example.error | percent }} |
   {% endif %}
   ```

### Documentation Standards

- Update `docs/` when adding features
- Include **code examples** in documentation
- Keep API reference synchronized with code
- Use **clear, concise language**
- Add **diagrams** for complex concepts (when helpful)

### Docstring Requirements

All public API functions and classes must have docstrings that include:
- Brief description
- Parameter descriptions with types
- Return value description
- Examples (when helpful)
- Raises section (if exceptions are raised)

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up-to-date with `main`

### PR Description

Include in your PR description:
- **What** changed and **why**
- Link to related issues (e.g., "Closes #123")
- **How to test** the changes
- Screenshots/examples (if applicable)

### Review Process

1. Automated tests will run on your PR
2. Maintainers will review your code
3. Address any requested changes
4. Once approved, a maintainer will merge

### Updating Your PR

```bash
# Sync with upstream
git fetch upstream
git rebase upstream/main

# Make additional changes
git add .
git commit -m "address review comments"
git push --force-with-lease origin feature/your-feature-name
```

## Project Structure

```
SMLR/
├── src/smlr/          # Main package code
│   ├── data.py        # Data loading and datasets
│   ├── lorentz.py     # Lorentzian mixture fitting
│   ├── emulator.py    # Emulator training and prediction
│   ├── metrics.py     # Evaluation metrics
│   ├── plotting.py    # Visualization utilities
│   └── demo/          # Demo scripts
├── tests/             # Test suite
├── docs/              # Documentation source
├── examples/          # Example scripts
└── pyproject.toml     # Project configuration
```

## Development Tips

- **Use type hints** - they help catch bugs and improve IDE support
- **Write tests first** - TDD helps clarify requirements
- **Keep PRs focused** - smaller PRs are easier to review
- **Ask questions** - if something is unclear, just ask!

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md
- Project documentation
- GitHub contributors page

## Questions?

- Open an issue for general questions
- Tag issues with `question` label
- Check existing documentation first

Thank you for contributing to SMLR! 🎉
