# Production-ready Dockerfile for QAPlibria-new
# Base image: Python 3.11 slim
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[dev]"

# Run linting and tests (optional, can be commented out for faster builds)
RUN make lint && \
    make test

# Set ownership and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose port for API server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
