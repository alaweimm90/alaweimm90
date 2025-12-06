# REPZ Platform - Production Dockerfile
# Multi-stage build optimized for production deployment

# Build stage
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git \
    ca-certificates

# Copy package files
COPY package*.json ./
COPY turbo.json ./
COPY packages/*/package.json ./packages/*/

# Install dependencies
RUN npm ci --only=production --ignore-scripts

# Copy source code
COPY . .

# Build the application
ENV NODE_ENV=production
ENV VITE_BUILD_TARGET=production

RUN npm run build:production

# Production stage
FROM node:20-alpine AS production

# Install production dependencies
RUN apk add --no-cache \
    tini \
    dumb-init \
    ca-certificates \
    curl \
    && addgroup -g 1001 -S nodejs \
    && adduser -S nextjs -u 1001

# Set working directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/package*.json ./

# Copy production configuration files
COPY --chown=nextjs:nodejs docker/config/ ./config/
COPY --chown=nextjs:nodejs docker/scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp \
    && chown -R nextjs:nodejs /app/logs /app/tmp

# Health check script
COPY --chown=nextjs:nodejs docker/healthcheck.js ./

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000
ENV HOST=0.0.0.0

# Security: Run as non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD node healthcheck.js

# Use tini as entrypoint for proper signal handling
ENTRYPOINT ["tini", "--"]

# Start the application
CMD ["node", "dist/server.js"]

# Development stage (for docker-compose)
FROM node:20-alpine AS development

WORKDIR /app

# Install all dependencies (including dev)
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git \
    ca-certificates

# Copy package files
COPY package*.json ./
COPY turbo.json ./

# Install all dependencies
RUN npm ci

# Copy source code
COPY . .

# Expose port
EXPOSE 3000
EXPOSE 6006

# Start development server
CMD ["npm", "run", "dev"]

# Testing stage
FROM node:20-alpine AS testing

WORKDIR /app

# Install dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git \
    ca-certificates \
    chromium \
    nss \
    freetype \
    freetype-dev \
    harfbuzz \
    ca-certificates \
    ttf-freefont

# Set Chrome path for testing
ENV CHROME_BIN=/usr/bin/chromium-browser
ENV CHROME_PATH=/usr/bin/chromium-browser

# Copy package files
COPY package*.json ./
COPY turbo.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Run tests
CMD ["npm", "run", "test:run"]

# Nginx stage (for serving static assets)
FROM nginx:alpine AS nginx

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY docker/nginx/nginx.conf /etc/nginx/nginx.conf
COPY docker/nginx/default.conf /etc/nginx/conf.d/default.conf

# Security headers
COPY docker/nginx/security-headers.conf /etc/nginx/conf.d/security-headers.conf

# Set up permissions for non-root execution
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    touch /var/run/nginx.pid && chown nginx:nginx /var/run/nginx.pid

# Switch to non-root user
USER nginx

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Expose non-privileged port
EXPOSE 8080

# Start nginx
CMD ["nginx", "-g", "daemon off;"]