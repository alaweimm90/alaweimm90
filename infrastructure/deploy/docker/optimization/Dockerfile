# ATLAS Optimization Service
FROM atlas-base:latest

# Install Python for analysis tools
USER root
RUN apk add --no-cache python3 py3-pip git && \
    pip3 install --no-cache-dir \
    radon \
    pylint \
    black \
    isort \
    mypy \
    bandit

# Switch back to atlas user
USER atlas

# Copy optimization service code
COPY tools/atlas/analysis/ ./analysis/
COPY tools/atlas/refactoring/ ./refactoring/
COPY tools/atlas/services/ ./services/
COPY tools/atlas/types/ ./types/
COPY tools/atlas/utils/ ./utils/

# Install additional Node.js dependencies for optimization
RUN npm install --save \
    @babel/parser \
    @babel/traverse \
    @typescript-eslint/parser \
    eslint \
    prettier \
    prom-client \
    winston \
    joi \
    simple-git

# Create service directory
RUN mkdir -p /app/optimization

# Copy service files
COPY tools/atlas/analysis/index.js /app/optimization/
COPY tools/atlas/analysis/analyzer.js /app/optimization/
COPY tools/atlas/refactoring/index.js /app/optimization/
COPY tools/atlas/refactoring/engine.js /app/optimization/
COPY tools/atlas/refactoring/scheduler.js /app/optimization/

# Set environment variables
ENV SERVICE_NAME=optimization
ENV SERVICE_PORT=3003
ENV REDIS_URL=redis://redis:6379
ENV METRICS_PORT=9093

# Analysis configuration
ENV ANALYSIS_DEPTH=3
ENV REFACTORING_ENABLED=true
ENV SCHEDULER_INTERVAL=3600000

# Expose ports
EXPOSE 3003 9093

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3003/health || exit 1

# Start optimization service
CMD ["node", "optimization/index.js"]</instructions>