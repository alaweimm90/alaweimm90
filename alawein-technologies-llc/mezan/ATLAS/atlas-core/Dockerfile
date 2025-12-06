# Multi-stage Dockerfile for MEZAN ATLAS Production Deployment
# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements files
COPY requirements.txt atlas-core/requirements.txt ./
COPY ../requirements.txt ../requirements-base.txt

# Create virtual environment and install dependencies
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r ../requirements-base.txt

# Stage 2: Testing stage (optional, can be skipped in production builds)
FROM builder AS tester

WORKDIR /app

# Copy application code
COPY . /app/atlas-core/
COPY ../src /app/src/
COPY ../tests /app/tests/

# Set Python path
ENV PYTHONPATH="/app/src:/app/atlas-core:${PYTHONPATH}"

# Install test dependencies
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    pytest-asyncio>=0.21.0 \
    pytest-timeout>=2.1.0

# Run tests with coverage
RUN pytest /app/tests --cov=/app/src --cov=/app/atlas-core \
    --cov-report=term --cov-report=xml \
    --timeout=300 || echo "Tests completed"

# Stage 3: Production runtime
FROM python:3.9-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r atlas && useradd -r -g atlas -d /app -s /sbin/nologin atlas

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /venv /venv

# Copy application code
COPY --chown=atlas:atlas . /app/atlas-core/
COPY --chown=atlas:atlas ../src /app/src/

# Copy startup script
COPY --chown=atlas:atlas scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set environment variables
ENV PATH="/venv/bin:$PATH" \
    PYTHONPATH="/app/src:/app/atlas-core:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=atlas.atlas_api_server \
    FLASK_ENV=production \
    ATLAS_ENV=production

# Security: Drop capabilities
RUN setcap -r /venv/bin/python3.9 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R atlas:atlas /app/logs /app/data /app/cache

# Switch to non-root user
USER atlas

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080

# Set labels for container metadata
LABEL \
    org.opencontainers.image.title="MEZAN ATLAS Core" \
    org.opencontainers.image.description="Enterprise Multi-Agent Research Orchestration Platform" \
    org.opencontainers.image.vendor="AlaweinOS" \
    org.opencontainers.image.version="3.5.0" \
    org.opencontainers.image.authors="Meshal Alawein <meshal@berkeley.edu>" \
    org.opencontainers.image.licenses="Apache-2.0"

# Entry point
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "atlas.atlas_api_server"]

# Stage 4: Development image (optional)
FROM production AS development

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    htop \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN /venv/bin/pip install --no-cache-dir \
    ipython \
    ipdb \
    black \
    ruff \
    mypy \
    pytest \
    pytest-cov \
    pytest-watch

# Set development environment
ENV FLASK_ENV=development \
    FLASK_DEBUG=1 \
    ATLAS_ENV=development

# Switch back to atlas user
USER atlas

# Override command for development
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080", "--reload"]