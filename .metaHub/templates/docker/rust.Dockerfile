# syntax=docker/dockerfile:1
# Golden Path Rust Dockerfile - Multi-stage, Security-Hardened
ARG RUST_VERSION=1.82

# =============================================================================
# Build Stage - Compile Rust binary
# =============================================================================
FROM rust:${RUST_VERSION}-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends pkg-config libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a dummy project to cache dependencies
RUN cargo new --bin app
WORKDIR /app/app

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock* ./

# Build dependencies only (cached layer)
RUN cargo build --release && \
    rm src/*.rs target/release/deps/app*

# Copy actual source code
COPY src ./src

# Build the actual application
RUN cargo build --release --locked

# =============================================================================
# Production Stage - Minimal runtime image
# =============================================================================
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies and create non-root user
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libssl3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy binary from builder
COPY --from=builder --chown=appuser:appgroup /app/app/target/release/app ./app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ./app --health || exit 1

# Expose port (customize as needed, use ports >= 1024)
# EXPOSE 8080

# Define the command to run your application
CMD ["./app"]
