# Contributing to WSI Tissue Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/dmp2/wsi-tissue-pipeline.git
cd wsi-tissue-pipeline

# Create a virtual environment (or use conda)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Code Style

This project uses **ruff** for linting and **black** for formatting, both configured in `pyproject.toml` with a line length of 100 characters.

```bash
# Lint
ruff check src/ tests/

# Format
black src/ tests/
```

## Running Tests

```bash
pytest tests/
```

## Making Changes

1. Fork the repository
2. Create a feature branch from `main`: `git checkout -b feature/your-feature`
3. Make your changes
4. Run linting and tests to ensure nothing is broken
5. Commit with a clear message describing the change
6. Push to your fork and open a Pull Request

## Reporting Issues

When reporting bugs, please include:

- A description of the problem
- Steps to reproduce it
- Your Python version and OS
- Any relevant error messages or logs

Feature requests are also welcome as GitHub Issues.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
