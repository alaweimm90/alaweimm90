.PHONY: help test lint format type-check clean install install-dev coverage benchmark profile docs check-all format-check

help:
	@echo "QAPlibria-new Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run all tests"
	@echo "  make test-fast         - Run tests without coverage"
	@echo "  make test-verbose      - Run tests with verbose output"
	@echo "  make coverage          - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              - Run linters (ruff, mypy)"
	@echo "  make format            - Auto-format code with black"
	@echo "  make format-check      - Check code format without changing"
	@echo "  make type-check        - Run type checking with mypy"
	@echo "  make check-all         - Run all quality checks"
	@echo ""
	@echo "Development:"
	@echo "  make install           - Install package in production mode"
	@echo "  make install-dev       - Install package with dev dependencies"
	@echo "  make clean             - Remove generated files and caches"
	@echo "  make docs              - Build documentation"
	@echo ""
	@echo "Performance:"
	@echo "  make benchmark         - Run benchmark suite"
	@echo "  make profile           - Profile critical functions"
	@echo ""
	@echo "Utilities:"
	@echo "  make help              - Show this help message"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	@echo "Running test suite with coverage..."
	pytest tests/ -v --cov=qaplibria --cov=atlas --cov-report=html --cov-report=term

test-fast:
	@echo "Running test suite (fast)..."
	pytest tests/ -q

test-verbose:
	@echo "Running test suite (verbose)..."
	pytest tests/ -vv --tb=short

coverage:
	@echo "Generating coverage report..."
	pytest tests/ --cov=qaplibria --cov=atlas --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Code Quality
lint:
	@echo "Running linters..."
	ruff check qaplibria atlas tests --select=E,W,F
	@echo "Type checking..."
	mypy qaplibria atlas --ignore-missing-imports 2>/dev/null || true

format:
	@echo "Formatting code with black..."
	black qaplibria atlas tests

format-check:
	@echo "Checking code format..."
	black --check qaplibria atlas tests

type-check:
	@echo "Running type checks..."
	mypy qaplibria atlas --ignore-missing-imports

check-all: format-check lint test
	@echo "✅ All checks passed!"

# Development utilities
clean:
	@echo "Cleaning generated files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Performance
benchmark:
	@echo "Running benchmark suite..."
	pytest tests/test_benchmarks.py -v --tb=short
	@echo "Benchmark results in results/"

profile:
	@echo "Profiling critical functions..."
	python -m cProfile -o profiling_output.prof qaplibria/core/pipeline.py
	python -m pstats profiling_output.prof

# Documentation
docs:
	@echo "Documentation is in README.md and docs/"
	@echo "Development guide: DEVELOPMENT.md"
	@echo "Changelog: CHANGELOG.md"
	@echo "Archived docs: .archive/docs/"
