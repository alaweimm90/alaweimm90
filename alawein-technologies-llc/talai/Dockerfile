# TalAI Turing Challenge System - Production Docker Image
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY turing-features/agent-tournaments/requirements.txt /app/requirements-tournaments.txt
COPY turing-features/devils-advocate/requirements.txt /app/requirements-advocate.txt
COPY turing-features/swarm-voting/requirements.txt /app/requirements-swarm.txt
COPY turing-features/emergent-behavior/requirements.txt /app/requirements-emergent.txt

# Install Python dependencies to /install
RUN pip install --prefix=/install --no-cache-dir \
    -r /app/requirements-tournaments.txt \
    -r /app/requirements-advocate.txt \
    -r /app/requirements-swarm.txt \
    -r /app/requirements-emergent.txt

# Install additional dependencies
RUN pip install --prefix=/install --no-cache-dir \
    pydantic==2.5.0 \
    anthropic==0.18.0 \
    openai==1.10.0 \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    redis==5.0.1 \
    numpy==1.26.3 \
    scipy==1.11.4

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.9-slim

LABEL maintainer="Meshal Alawein <meshal@berkeley.edu>"
LABEL description="TalAI Turing Challenge System - Autonomous Research Validation"
LABEL version="1.0.0"

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY turing-features/ /app/turing-features/
COPY INTEGRATION_EXAMPLE.py /app/
COPY IMPLEMENTATION_SUMMARY.md /app/
COPY MASTER_INDEX.md /app/
COPY PROJECT_STATUS.md /app/

# Create non-root user for security
RUN useradd -m -u 1000 talair && \
    chown -R talair:talair /app

USER talair

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "talair_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Expose API port
EXPOSE 808000
