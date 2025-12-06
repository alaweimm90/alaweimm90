# Contributing to ORCHEX

Thank you for your interest in contributing to ORCHEX! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ORCHEX.git
   cd ORCHEX
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   # Or manually:
   pip install -r requirements-dev.txt
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Copy environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Code Style

We use:
- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking (gradually increasing strictness)

Run formatting and linting:
```bash
make format  # Auto-fix issues
make lint    # Check for issues
```

## Testing

- Write tests for all new features
- Maintain test coverage above 60% (target: 85%)
- Run tests: `make test` or `pytest`

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(api): add new evaluation endpoint
fix(cache): resolve memory leak in LRU cache
docs(readme): update installation instructions
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all tests pass and linting is clean
4. Update documentation if needed
5. Submit a pull request with a clear description

## Project Structure

```
ORCHEX/
├── src/ORCHEX/          # Main package code
├── tests/              # Test files
├── examples/           # Example inputs
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Questions?

Open an issue or contact the maintainers.
