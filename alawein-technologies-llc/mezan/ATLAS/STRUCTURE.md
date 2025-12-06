# ORCHEX Project Structure

This document describes the standard repository structure for the ORCHEX Hypothesis Evaluation Platform.

## Directory Layout

```
ORCHEX/
├── src/ORCHEX/              # Main Python package
│   ├── __init__.py         # Package initialization and exports
│   ├── atlas_api_server.py # Flask REST API server
│   ├── quality_gates.py    # Pre-run validation system
│   ├── pii_redactor.py     # PII detection and redaction
│   ├── tracing_logger.py   # OpenTelemetry-compatible tracing
│   ├── shadow_eval.py      # Shadow evaluation system
│   ├── advanced_systems.py # Meta-evaluation, ML routing, etc.
│   ├── results_store.py    # Results storage and retrieval
│   └── validate_*.py       # Input/artifact validation
│
├── tests/                  # Test suite
│   └── golden/            # Golden test examples
│       └── golden_tests.py
│
├── examples/               # Example input files
│   ├── hypothesis.input.json
│   ├── nightmare_example.json
│   ├── chaos_example.json
│   └── ...
│
├── docs/                   # Documentation
│   ├── guides/            # User and developer guides
│   ├── analysis/          # Strategic analysis documents
│   ├── legal/             # Legal compliance docs
│   └── engineering-framework/ # Engineering standards
│
├── scripts/               # Utility scripts (if any)
│
├── .github/               # GitHub configuration
│   └── workflows/         # CI/CD workflows
│       └── ci.yml
│
├── README.md              # Main project documentation
├── START_HERE.md          # Getting started guide
├── SETUP.md               # Setup instructions
├── CONTRIBUTING.md        # Contribution guidelines
├── LICENSE                # MIT License
├── Makefile               # Development commands
├── pyproject.toml         # Python project configuration
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Multi-service Docker setup
└── .env.example           # Environment variable template
```

## Key Files

### Configuration Files

- **`pyproject.toml`**: Python project metadata, build configuration, and tool settings (Black, Ruff, MyPy, pytest)
- **`requirements.txt`**: Production dependencies
- **`requirements-dev.txt`**: Development dependencies (includes production deps)
- **`.env.example`**: Template for environment variables
- **`docker-compose.yml`**: Multi-service setup (API, Redis, Prometheus, Grafana)

### Development Files

- **`Makefile`**: Common development commands (`make test`, `make lint`, etc.)
- **`.pre-commit-config.yaml`**: Pre-commit hooks for code quality
- **`.editorconfig`**: Editor configuration for consistent formatting
- **`.gitignore`**: Git ignore patterns

### Documentation

- **`README.md`**: Project overview and main documentation
- **`START_HERE.md`**: Navigation guide for new users
- **`SETUP.md`**: Quick setup instructions
- **`CONTRIBUTING.md`**: Guidelines for contributors
- **`docs/`**: Comprehensive documentation organized by topic

## Package Structure

The main package is located in `src/ORCHEX/` following Python best practices:

- **`src/` layout**: Prevents accidental imports from project root
- **Package imports**: All imports use `from ORCHEX.module import ...`
- **`__init__.py`**: Exports main classes and functions for easy importing

## Running the Project

### Local Development

```bash
# Install dependencies
make install-dev

# Run API server
make run

# Run tests
make test

# Format code
make format
```

### Docker

```bash
# Start all services
make docker-up

# Or build and run manually
docker-compose up -d
```

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- **`ci.yml`**: Runs linting, type checking, and tests on every push/PR

## Standards

- **Python**: 3.9+ (tested on 3.9, 3.10, 3.11, 3.12)
- **Code Style**: Black (100 char line length)
- **Linting**: Ruff
- **Type Checking**: MyPy (gradually increasing strictness)
- **Testing**: pytest with coverage (target: 85%)
- **Commits**: Conventional Commits format
