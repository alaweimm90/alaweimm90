# MEZAN ORCHEX Docker Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Docker Architecture](#docker-architecture)
3. [Building Images](#building-images)
4. [Running Containers](#running-containers)
5. [Docker Compose](#docker-compose)
6. [Networking](#networking)
7. [Volumes & Storage](#volumes--storage)
8. [Security](#security)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

## Introduction

MEZAN ORCHEX uses Docker for containerization, providing consistent deployments across different environments. This guide covers Docker-specific aspects of running ORCHEX.

### Benefits of Dockerization

- **Consistency**: Same environment from development to production
- **Isolation**: Services run in isolated containers
- **Scalability**: Easy horizontal scaling
- **Portability**: Run anywhere Docker is supported
- **Version Control**: Tagged images for rollback capability

## Docker Architecture

### Multi-Stage Build

Our Dockerfile uses multi-stage builds for optimization:

```dockerfile
# Stage 1: Builder - Install dependencies
FROM python:3.9-slim AS builder

# Stage 2: Tester - Run tests
FROM builder AS tester

# Stage 3: Production - Minimal runtime
FROM python:3.9-slim AS production

# Stage 4: Development - With dev tools
FROM production AS development
```

### Image Layers

```
mezan-ORCHEX:latest
├── Base: python:3.9-slim (45MB)
├── System dependencies (15MB)
├── Python packages (250MB)
├── Application code (10MB)
└── Configuration (1MB)
Total: ~321MB
```

## Building Images

### Basic Build

```bash
# Build default production image
docker build -t mezan-ORCHEX:latest .

# Build specific target
docker build --target development -t mezan-ORCHEX:dev .

# Build with build args
docker build \
    --build-arg VERSION=3.5.0 \
    --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
    -t mezan-ORCHEX:3.5.0 .
```

### Advanced Build Options

```bash
# Multi-platform build
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t mezan-ORCHEX:latest \
    --push .

# Build with cache mount
docker build \
    --cache-from type=registry,ref=myregistry/mezan-ORCHEX:cache \
    --cache-to type=registry,ref=myregistry/mezan-ORCHEX:cache,mode=max \
    -t mezan-ORCHEX:latest .

# Build with secrets
docker build \
    --secret id=api_key,src=.api_key \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t mezan-ORCHEX:latest .
```

### Build Script Usage

```bash
# Production build
./scripts/build.sh --target production --version v3.5.0

# Development build with no cache
./scripts/build.sh --target development --no-cache

# Multi-platform build and push
./scripts/build.sh --multi-platform --push --registry myregistry.io
```

## Running Containers

### Basic Run Commands

```bash
# Run with default settings
docker run -d --name ORCHEX-api -p 8080:8080 mezan-ORCHEX:latest

# Run with environment variables
docker run -d \
    --name ORCHEX-api \
    -p 8080:8080 \
    -e ANTHROPIC_API_KEY=your_key \
    -e LOG_LEVEL=DEBUG \
    mezan-ORCHEX:latest

# Run with volume mounts
docker run -d \
    --name ORCHEX-api \
    -p 8080:8080 \
    -v $(pwd)/config:/app/config:ro \
    -v ORCHEX-data:/app/data \
    mezan-ORCHEX:latest
```

### Advanced Run Options

```bash
# Run with resource limits
docker run -d \
    --name ORCHEX-api \
    --memory="2g" \
    --memory-swap="4g" \
    --cpus="2" \
    --pids-limit=1000 \
    -p 8080:8080 \
    mezan-ORCHEX:latest

# Run with health check
docker run -d \
    --name ORCHEX-api \
    --health-cmd="curl -f http://localhost:8080/health || exit 1" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    --health-start-period=60s \
    -p 8080:8080 \
    mezan-ORCHEX:latest

# Run with security options
docker run -d \
    --name ORCHEX-api \
    --security-opt=no-new-privileges \
    --cap-drop=ALL \
    --cap-add=NET_BIND_SERVICE \
    --read-only \
    --tmpfs /tmp \
    -p 8080:8080 \
    mezan-ORCHEX:latest
```

## Docker Compose

### Development Environment

```yaml
# docker-compose.dev.yml
version: '3.9'

services:
  ORCHEX-api-dev:
    build:
      context: .
      target: development
    volumes:
      - ./src:/app/src:rw  # Hot reload
      - ./tests:/app/tests:rw
    environment:
      FLASK_ENV: development
      FLASK_DEBUG: 1
    ports:
      - "8080:8080"  # API
      - "5678:5678"  # Debugger
    command: python -m flask run --reload
```

### Production Environment

```yaml
# docker-compose.yml
version: '3.9'

services:
  ORCHEX-api:
    image: mezan-ORCHEX:${VERSION:-latest}
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Compose Commands

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d ORCHEX-api

# Scale service
docker-compose up -d --scale ORCHEX-api=3

# View logs
docker-compose logs -f ORCHEX-api

# Execute command in running container
docker-compose exec ORCHEX-api python -m ORCHEX.migrate

# Stop services
docker-compose stop

# Remove everything
docker-compose down -v --remove-orphans
```

## Networking

### Network Architecture

```
┌─────────────────────────────────────────┐
│         ORCHEX-network (bridge)          │
│         172.28.0.0/16                   │
├─────────────────────────────────────────┤
│ ORCHEX-api     │ 172.28.0.2              │
│ redis         │ 172.28.0.3              │
│ postgres      │ 172.28.0.4              │
│ nginx         │ 172.28.0.5              │
│ prometheus    │ 172.28.0.6              │
│ grafana       │ 172.28.0.7              │
└─────────────────────────────────────────┘
```

### Custom Networks

```bash
# Create custom network
docker network create \
    --driver bridge \
    --subnet 172.28.0.0/16 \
    --gateway 172.28.0.1 \
    --opt com.docker.network.bridge.name=atlas0 \
    ORCHEX-network

# Connect container to network
docker network connect ORCHEX-network ORCHEX-api

# Inspect network
docker network inspect ORCHEX-network

# List networks
docker network ls
```

### Service Discovery

```yaml
# Internal DNS resolution
services:
  ORCHEX-api:
    hostname: ORCHEX-api
    networks:
      - ORCHEX-network

  redis:
    hostname: redis
    networks:
      - ORCHEX-network
```

```bash
# Test connectivity
docker-compose exec ORCHEX-api ping redis
docker-compose exec ORCHEX-api nslookup redis
```

## Volumes & Storage

### Volume Types

```yaml
# Named volumes (recommended for data)
volumes:
  ORCHEX-data:
    driver: local
  ORCHEX-logs:
    driver: local

# Bind mounts (for development)
volumes:
  - ./src:/app/src:rw
  - ./config:/app/config:ro

# tmpfs mounts (for temporary data)
tmpfs:
  - /tmp
  - /run
```

### Volume Management

```bash
# Create volume
docker volume create ORCHEX-data

# List volumes
docker volume ls

# Inspect volume
docker volume inspect ORCHEX-data

# Remove unused volumes
docker volume prune

# Backup volume
docker run --rm \
    -v ORCHEX-data:/source:ro \
    -v $(pwd)/backup:/backup \
    alpine tar czf /backup/ORCHEX-data.tar.gz -C /source .

# Restore volume
docker run --rm \
    -v ORCHEX-data:/target \
    -v $(pwd)/backup:/backup:ro \
    alpine tar xzf /backup/ORCHEX-data.tar.gz -C /target
```

### Storage Drivers

```bash
# Check storage driver
docker info | grep "Storage Driver"

# Configure storage driver (daemon.json)
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ]
}
```

## Security

### Image Security

```bash
# Scan image for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image mezan-ORCHEX:latest

# Sign images
export DOCKER_CONTENT_TRUST=1
docker trust sign mezan-ORCHEX:latest

# Verify image signature
docker trust inspect mezan-ORCHEX:latest
```

### Runtime Security

```yaml
# Security options in docker-compose.yml
services:
  ORCHEX-api:
    security_opt:
      - no-new-privileges:true
      - seccomp:unconfined
      - apparmor:docker-default
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    user: "1000:1000"
```

### Secrets Management

```bash
# Create secret
echo "my-secret-value" | docker secret create api_key -

# Use in service
docker service create \
    --name ORCHEX-api \
    --secret api_key \
    mezan-ORCHEX:latest

# Access in container
cat /run/secrets/api_key
```

## Performance Optimization

### Image Optimization

```dockerfile
# Minimize layers
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*

# Use .dockerignore
echo "*.pyc" >> .dockerignore
echo "__pycache__" >> .dockerignore
echo ".git" >> .dockerignore

# Multi-stage builds
FROM python:3.9-slim AS builder
# Build steps
FROM python:3.9-slim AS production
COPY --from=builder /app /app
```

### Container Optimization

```bash
# Limit container resources
docker run -d \
    --memory="1g" \
    --memory-reservation="750m" \
    --cpus="1.5" \
    --blkio-weight=500 \
    mezan-ORCHEX:latest

# Use init process
docker run -d --init mezan-ORCHEX:latest

# Optimize logging
docker run -d \
    --log-driver=json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    mezan-ORCHEX:latest
```

### Build Cache Optimization

```bash
# Use BuildKit
export DOCKER_BUILDKIT=1

# Cache mount
# syntax=docker/dockerfile:1
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# External cache
docker build \
    --cache-from type=registry,ref=myregistry/cache \
    --cache-to type=registry,ref=myregistry/cache \
    -t mezan-ORCHEX:latest .
```

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs ORCHEX-api

# Debug with shell
docker run -it --entrypoint /bin/sh mezan-ORCHEX:latest

# Check exit code
docker inspect ORCHEX-api --format='{{.State.ExitCode}}'
```

#### Permission Issues

```bash
# Fix ownership
docker run --rm -v ORCHEX-data:/data alpine \
    chown -R 1000:1000 /data

# Run as root (debugging only)
docker run -it --user root mezan-ORCHEX:latest /bin/bash
```

#### Networking Issues

```bash
# Test DNS
docker run --rm alpine nslookup google.com

# Check port binding
netstat -tulpn | grep 8080

# Inspect network
docker network inspect bridge
```

#### Performance Issues

```bash
# Check resource usage
docker stats

# Inspect cgroup limits
docker inspect ORCHEX-api | jq '.[0].HostConfig.Memory'

# Profile container
docker run -d --name ORCHEX-api \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    mezan-ORCHEX:latest

docker exec ORCHEX-api py-spy top --pid 1
```

### Debugging Commands

```bash
# Interactive shell
docker exec -it ORCHEX-api /bin/bash

# Run command
docker exec ORCHEX-api python -c "import sys; print(sys.path)"

# Copy files from container
docker cp ORCHEX-api:/app/logs/error.log ./error.log

# System information
docker exec ORCHEX-api cat /proc/cpuinfo
docker exec ORCHEX-api free -h
docker exec ORCHEX-api df -h

# Network debugging
docker exec ORCHEX-api netstat -tulpn
docker exec ORCHEX-api ss -tulpn
docker exec ORCHEX-api ip addr

# Process debugging
docker exec ORCHEX-api ps aux
docker exec ORCHEX-api top
```

### Container Logs

```bash
# Follow logs
docker logs -f ORCHEX-api

# Last 100 lines
docker logs --tail 100 ORCHEX-api

# Since timestamp
docker logs --since 2024-01-01T00:00:00 ORCHEX-api

# Filter logs
docker logs ORCHEX-api 2>&1 | grep ERROR

# Export logs
docker logs ORCHEX-api > ORCHEX-api.log 2>&1
```

### Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune

# Complete cleanup
docker system prune -a --volumes

# Check disk usage
docker system df
```

## Best Practices

### Image Building

1. **Use specific base image tags**: `python:3.9.18-slim`, not `python:latest`
2. **Minimize layers**: Combine RUN commands
3. **Order layers**: Least changing first
4. **Use .dockerignore**: Exclude unnecessary files
5. **Multi-stage builds**: Separate build and runtime
6. **Non-root user**: Run containers as non-root
7. **Health checks**: Define HEALTHCHECK in Dockerfile
8. **Labels**: Add metadata with LABEL
9. **Security scanning**: Scan images before deployment
10. **Sign images**: Use Docker Content Trust

### Container Runtime

1. **Resource limits**: Always set memory and CPU limits
2. **Read-only filesystem**: Use --read-only when possible
3. **Drop capabilities**: Use --cap-drop=ALL
4. **No new privileges**: Use --security-opt=no-new-privileges
5. **Init process**: Use --init for proper signal handling
6. **Logging**: Configure appropriate log drivers
7. **Secrets**: Never hardcode, use secrets management
8. **Health monitoring**: Implement health checks
9. **Graceful shutdown**: Handle SIGTERM properly
10. **One process per container**: Follow single responsibility

### Docker Compose

1. **Version pinning**: Specify service versions
2. **Environment files**: Use .env files for configuration
3. **Named volumes**: Prefer over bind mounts in production
4. **Networks**: Use custom networks for isolation
5. **Dependencies**: Use depends_on with health checks
6. **Resource limits**: Define in deploy section
7. **Restart policies**: Configure appropriately
8. **Extension fields**: Use YAML anchors for DRY
9. **Override files**: Separate dev and prod configs
10. **Secrets**: Use Docker secrets in Swarm mode

---

**Last Updated:** November 2024
**Version:** 3.5.0
**Author:** MEZAN Development Team