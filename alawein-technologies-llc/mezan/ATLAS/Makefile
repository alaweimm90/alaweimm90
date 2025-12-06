.PHONY: install install-dev test lint format clean run docker-build docker-up docker-down help

help:
	@echo "ATLAS Development Commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters (ruff, mypy)"
	@echo "  make format        - Format code (black, ruff)"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make run           - Run API server locally"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start Docker Compose services"
	@echo "  make docker-down   - Stop Docker Compose services"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest --cov=atlas --cov-report=term --cov-report=html

lint:
	ruff check src/ tests/
	mypy src/atlas/ || true

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m atlas.atlas_api_server

docker-build:
	docker build -t atlas-api:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
