# ATLAS Multi-Agent Orchestration Platform
# Docker container for API server and CLI

# =============================================================================
# Build Stage
# =============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Install build dependencies for native modules (better-sqlite3)
RUN apk add --no-cache python3 make g++

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev for building)
RUN npm ci

# Copy source files
COPY tsconfig.json ./
COPY tools/ ./tools/

# =============================================================================
# Production Stage
# =============================================================================
FROM node:20-alpine AS production

WORKDIR /app

# Install runtime dependencies for native modules
RUN apk add --no-cache python3 make g++

# Create non-root user for security
RUN addgroup -g 1001 atlas && \
    adduser -u 1001 -G atlas -s /bin/sh -D atlas

# Copy package files
COPY package*.json ./

# Install production dependencies only
RUN npm ci --omit=dev && \
    npm cache clean --force

# Copy built application from builder
COPY --from=builder /app/tools ./tools
COPY --from=builder /app/tsconfig.json ./

# Create data directories with proper permissions
RUN mkdir -p /app/.atlas/data && \
    chown -R atlas:atlas /app

# Switch to non-root user
USER atlas

# Environment variables
ENV NODE_ENV=production
ENV ATLAS_DATA_DIR=/app/.atlas
ENV ATLAS_PORT=3000

# Expose API port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# Default command: start the API server
CMD ["npx", "tsx", "tools/atlas/api/cli.ts"]
