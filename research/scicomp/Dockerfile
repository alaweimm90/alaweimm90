# SciComp Docker Image
FROM python:3.11-slim

# Metadata
LABEL maintainer="UC Berkeley SciComp Team <scicomp@berkeley.edu>"
LABEL version="1.0.1"
LABEL description="SciComp - Cross-Platform Scientific Computing Suite"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BERKELEY_SCICOMP_DOCKER=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libnetcdf-dev \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash berkeley

# Set working directory
WORKDIR /app

# Copy requirements first (for better layer caching)
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install jupyter jupyterlab ipywidgets

# Copy the entire framework
COPY . .

# Install Berkeley SciComp Framework
RUN pip install -e .

# Create directories for data and notebooks
RUN mkdir -p /app/data /app/notebooks /app/output && \
    chown -R berkeley:berkeley /app

# Switch to non-root user
USER berkeley

# Set up Jupyter
RUN jupyter notebook --generate-config && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Expose ports for Jupyter Lab and Dask
EXPOSE 8888 8787

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import Python; print('Berkeley SciComp Framework OK')" || exit 1

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]