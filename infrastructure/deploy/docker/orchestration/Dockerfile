# ATLAS Orchestration Service
FROM atlas-base:latest

# Copy orchestration service code
COPY tools/atlas/orchestration/ ./orchestration/
COPY tools/atlas/services/ ./services/
COPY tools/atlas/types/ ./types/
COPY tools/atlas/utils/ ./utils/

# Install additional dependencies for orchestration
RUN npm install --save \
    express \
    redis \
    @kubernetes/client-node \
    prom-client \
    winston \
    joi

# Create service directory
RUN mkdir -p /app/orchestration

# Copy service files
COPY tools/atlas/orchestration/index.js /app/orchestration/
COPY tools/atlas/orchestration/router.js /app/orchestration/
COPY tools/atlas/orchestration/registry.js /app/orchestration/
COPY tools/atlas/orchestration/load-balancer.js /app/orchestration/
COPY tools/atlas/orchestration/fallback-manager.js /app/orchestration/

# Set environment variables
ENV SERVICE_NAME=orchestration
ENV SERVICE_PORT=3001
ENV REDIS_URL=redis://redis:6379
ENV METRICS_PORT=9091

# Expose ports
EXPOSE 3001 9091

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

# Start orchestration service
CMD ["node", "orchestration/index.js"]</instructions>