# ATLAS Base Docker Image
FROM node:20-alpine

# Install security updates and required packages
RUN apk update && apk upgrade && \
    apk add --no-cache \
    curl \
    wget \
    ca-certificates \
    dumb-init \
    su-exec \
    tzdata && \
    rm -rf /var/cache/apk/*

# Create non-root user
RUN addgroup -g 1001 -S atlas && \
    adduser -S atlas -u 1001

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && \
    npm cache clean --force

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R atlas:atlas /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Security: Don't run as root
USER atlas

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Default command (should be overridden by child images)
CMD ["node", "index.js"]</instructions>