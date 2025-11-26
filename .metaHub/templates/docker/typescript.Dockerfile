# syntax=docker/dockerfile:1
ARG NODE_VERSION=20-slim

# Build stage
FROM node:${NODE_VERSION} AS builder

WORKDIR /app

# Install dependencies first (for better caching)
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy source and build
COPY . .
RUN npm run build 2>/dev/null || echo "No build script found"

# Production stage
FROM node:${NODE_VERSION}

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy built artifacts from builder
COPY --from=builder --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:appgroup /app/dist ./dist
COPY --from=builder --chown=appuser:appgroup /app/package*.json ./

# Switch to non-root user
USER appuser

# Set environment variables
ENV NODE_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "console.log('healthy')" || exit 1

# Expose port (customize as needed)
# EXPOSE 3000

# Define the command to run your application
# CMD ["node", "dist/index.js"]
