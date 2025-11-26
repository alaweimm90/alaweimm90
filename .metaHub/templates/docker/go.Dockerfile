# syntax=docker/dockerfile:1
ARG GO_VERSION=1.22

# Build stage
FROM golang:${GO_VERSION}-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git

# Copy go mod files first (for better caching)
COPY go.mod go.sum* ./
RUN go mod download

# Copy source and build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /app/main .

# Production stage - minimal scratch image
FROM alpine:3.19

WORKDIR /app

# Create non-root user for security
RUN addgroup -g 1000 appgroup && \
    adduser -u 1000 -G appgroup -s /bin/sh -D appuser

# Copy binary from builder
COPY --from=builder --chown=appuser:appgroup /app/main ./main

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ./main --health || exit 1

# Expose port (customize as needed)
# EXPOSE 8080

# Define the command to run your application
# CMD ["./main"]
