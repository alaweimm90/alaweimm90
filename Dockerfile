# Meta-Orchestration Master API Server
# Multi-stage build for minimal production image

# ============================================================================
# Stage 1: Build
# ============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache python3 make g++

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install all dependencies (including devDependencies for build)
RUN npm ci

# Copy source files
COPY tools/ ./tools/
COPY .ai/ ./.ai/
COPY .metaHub/ ./.metaHub/

# Build TypeScript (if needed)
RUN npm run type-check || true

# ============================================================================
# Stage 2: Production
# ============================================================================
FROM node:20-alpine AS production

LABEL maintainer="Meta-Orchestration Master"
LABEL description="AI Governance REST API Server"
LABEL version="1.0.0"

WORKDIR /app

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Copy package files
COPY package*.json ./

# Install production dependencies only
RUN npm ci --omit=dev && \
    npm cache clean --force

# Copy built application
COPY --from=builder /app/tools/ ./tools/
COPY --from=builder /app/.ai/ ./.ai/
COPY --from=builder /app/.metaHub/ ./.metaHub/

# Create necessary directories
RUN mkdir -p .ai/cache .ai/config .ai/reports && \
    chown -R nodejs:nodejs /app

# Switch to non-root user
USER nodejs

# Environment variables
ENV NODE_ENV=production
ENV AI_API_PORT=3200
ENV RATE_LIMIT_MAX=100
ENV API_AUTH_REQUIRED=true

# Expose API port
EXPOSE 3200

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3200/health || exit 1

# Start the API server
CMD ["npx", "tsx", "tools/ai/api/server.ts"]

