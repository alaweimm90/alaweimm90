# SuperTool Multi-Stage Dockerfile
# Optimized for size and security

# Stage 1: Build dependencies
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY cli/package*.json ./

# Install dependencies (including dev dependencies for build)
RUN npm ci --only=production && \
    npm cache clean --force

# Stage 2: Runtime image
FROM node:20-alpine AS runtime

# Add metadata
LABEL maintainer="Meshal Alawein" \
      description="SuperTool - DevOps automation CLI" \
      version="1.0.0"

# Create non-root user for security
RUN addgroup -g 1001 -S supertool && \
    adduser -S supertool -u 1001 -G supertool

# Set working directory
WORKDIR /app

# Copy dependencies from builder
COPY --from=builder --chown=supertool:supertool /app/node_modules ./node_modules

# Copy application files
COPY --chown=supertool:supertool cli/bin ./bin
COPY --chown=supertool:supertool cli/lib ./lib
COPY --chown=supertool:supertool cli/package.json ./

# Switch to non-root user
USER supertool

# Set environment variables
ENV NODE_ENV=production \
    PATH="/app/bin:${PATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "console.log('healthy')" || exit 1

# Default command
ENTRYPOINT ["node", "bin/cli.mjs"]
CMD ["--help"]
