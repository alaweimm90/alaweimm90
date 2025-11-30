# ATLAS Storage Service
FROM atlas-base:latest

# Install PostgreSQL client and Redis for data operations
USER root
RUN apk add --no-cache postgresql-client redis && \
    pip3 install --no-cache-dir psycopg2-binary redis

# Switch back to atlas user
USER atlas

# Copy storage service code
COPY tools/atlas/services/ ./services/
COPY tools/atlas/types/ ./types/
COPY tools/atlas/utils/ ./utils/

# Install additional dependencies for storage
RUN npm install --save \
    pg \
    redis \
    @elastic/elasticsearch \
    prom-client \
    winston \
    joi \
    knex \
    objection

# Create service directory
RUN mkdir -p /app/storage

# Copy service files
COPY tools/atlas/services/storage/index.js /app/storage/
COPY tools/atlas/services/storage/metrics-db.js /app/storage/
COPY tools/atlas/services/storage/state-store.js /app/storage/
COPY tools/atlas/services/storage/task-history.js /app/storage/
COPY tools/atlas/services/storage/config-store.js /app/storage/

# Database migration files
COPY tools/atlas/services/storage/migrations/ /app/storage/migrations/

# Set environment variables
ENV SERVICE_NAME=storage
ENV SERVICE_PORT=3004
ENV METRICS_PORT=9094

# Database configuration
ENV POSTGRES_HOST=atlas-postgres
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB=atlas
ENV POSTGRES_USER=atlas
ENV POSTGRES_PASSWORD=""

# Redis configuration
ENV REDIS_URL=redis://redis:6379

# Elasticsearch configuration
ENV ELASTICSEARCH_URL=http://elasticsearch:9200

# Storage configuration
ENV METRICS_RETENTION_DAYS=90
ENV HISTORY_RETENTION_DAYS=365
ENV STATE_BACKUP_INTERVAL=3600000

# Expose ports
EXPOSE 3004 9094

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3004/health || exit 1

# Start storage service
CMD ["node", "storage/index.js"]</instructions>