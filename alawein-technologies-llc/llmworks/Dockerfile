# Multi-stage build for production optimization
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files for dependency installation
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY . .

# Build arguments
ARG NODE_ENV=production
ARG VITE_ENVIRONMENT=production
ARG VITE_API_URL
ARG VITE_CDN_URL
ARG VITE_WS_URL
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY
ARG VITE_SENTRY_DSN

# Set environment variables
ENV NODE_ENV=$NODE_ENV
ENV VITE_ENVIRONMENT=$VITE_ENVIRONMENT
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_CDN_URL=$VITE_CDN_URL
ENV VITE_WS_URL=$VITE_WS_URL
ENV VITE_SUPABASE_URL=$VITE_SUPABASE_URL
ENV VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY
ENV VITE_SENTRY_DSN=$VITE_SENTRY_DSN

# Build the application
RUN npm run build

# Production stage with nginx
FROM nginx:alpine AS production

# Install security updates
RUN apk update && apk upgrade && apk add --no-cache \
    curl \
    ca-certificates

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built application from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy service worker to root
COPY --from=builder /app/public/service-worker.js /usr/share/nginx/html/
COPY --from=builder /app/public/manifest.json /usr/share/nginx/html/

# Create health check endpoint
RUN echo '<!DOCTYPE html><html><head><title>Health Check</title></head><body><h1>OK</h1><p>LLM Works is running</p></body></html>' > /usr/share/nginx/html/health

# Set up proper permissions for non-root execution
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    touch /var/run/nginx.pid && chown nginx:nginx /var/run/nginx.pid

# Switch to non-root user
USER nginx

# Expose non-privileged ports
EXPOSE 8080 8443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]