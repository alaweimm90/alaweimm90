# SuperTool Docker Guide

Complete guide for using SuperTool with Docker and container orchestration.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Docker Images](#docker-images)
- [Building Images](#building-images)
- [Running Containers](#running-containers)
- [Docker Compose](#docker-compose)
- [Security Scanning](#security-scanning)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Build the Image

```bash
# Using Makefile (recommended)
make docker-build

# Or using Docker directly
docker build -f devops/docker/Dockerfile -t supertool:latest .
```

### 2. Run SuperTool

```bash
# Show help
docker run --rm supertool:latest --help

# Run specific command
docker run --rm supertool:latest discover

# Mount volumes for file access
docker run --rm -v $(pwd)/outputs:/outputs supertool:latest build
```

### 3. Development Environment

```bash
# Start development container
make docker-dev

# Access container shell
docker exec -it supertool-dev /bin/sh
```

---

## Docker Images

### Production Image (`Dockerfile`)

**Features:**

- Multi-stage build for minimal size
- Alpine Linux base (~50MB)
- Non-root user for security
- Only production dependencies
- Health check included

**Size:** ~80-100MB (vs ~500MB without optimization)

**Use for:**

- Production deployments
- CI/CD pipelines
- Container registries
- Kubernetes clusters

### Development Image (`Dockerfile.dev`)

**Features:**

- Includes dev dependencies
- Development tools (git, bash, vim, nodemon)
- Interactive shell
- Hot reload support

**Size:** ~150-200MB

**Use for:**

- Local development
- Testing
- Debugging
- Interactive work

---

## Building Images

### Production Build

```bash
# Standard build
docker build -f devops/docker/Dockerfile -t supertool:latest .

# With build args
docker build \
  -f devops/docker/Dockerfile \
  -t supertool:1.0.0 \
  --build-arg NODE_VERSION=20 \
  .

# Multi-platform build
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f devops/docker/Dockerfile \
  -t supertool:latest \
  .
```

### Development Build

```bash
docker build -f devops/docker/Dockerfile.dev -t supertool:dev .
```

### Build with Cache

```bash
# Use BuildKit cache
docker buildx build \
  --cache-from type=registry,ref=myregistry/supertool:cache \
  --cache-to type=registry,ref=myregistry/supertool:cache \
  -t supertool:latest \
  .
```

---

## Running Containers

### Basic Usage

```bash
# Run with default command (--help)
docker run --rm supertool:latest

# Run specific command
docker run --rm supertool:latest --version
docker run --rm supertool:latest discover
docker run --rm supertool:latest validate
```

### With Volume Mounts

```bash
# Mount current directory
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  supertool:latest discover

# Mount outputs directory
docker run --rm \
  -v $(pwd)/outputs:/outputs \
  supertool:latest build --output /outputs
```

### Environment Variables

```bash
# Set environment variables
docker run --rm \
  -e NODE_ENV=production \
  -e DEBUG=* \
  supertool:latest discover

# Load from .env file
docker run --rm \
  --env-file .env \
  supertool:latest discover
```

### Interactive Shell

```bash
# Production image
docker run --rm -it supertool:latest /bin/sh

# Development image
docker run --rm -it supertool:dev /bin/sh
```

---

## Docker Compose

### Services Available

1. **supertool** - Production service
2. **supertool-dev** - Development service with hot reload

### Starting Services

```bash
# Start all services
docker-compose -f devops/docker/docker-compose.yml up -d

# Start specific service
docker-compose -f devops/docker/docker-compose.yml up -d supertool-dev

# View logs
docker-compose -f devops/docker/docker-compose.yml logs -f

# Stop services
docker-compose -f devops/docker/docker-compose.yml down
```

### Using Makefile Commands

```bash
# Start development environment
make docker-dev

# Start all services
make docker-up

# View logs
make docker-logs

# Stop all services
make docker-down
```

### Development Workflow

```bash
# 1. Start dev container
make docker-dev

# 2. Access shell
docker exec -it supertool-dev /bin/sh

# 3. Inside container:
node bin/cli.mjs --help
npm test
npm run build

# 4. Stop when done
make docker-down
```

---

## Security Scanning

### Using Trivy

#### Automated Scanning

```bash
# Using Makefile
make docker-scan

# Using script directly
bash devops/security/trivy-scan.sh supertool:latest
```

#### Manual Scanning

```bash
# Scan Docker image
trivy image supertool:latest

# Scan with specific severity
trivy image --severity CRITICAL,HIGH supertool:latest

# Generate SARIF report for GitHub
trivy image --format sarif --output trivy-results.sarif supertool:latest

# Scan filesystem
trivy fs ./cli

# Scan configuration files
trivy config ./devops
```

#### Generate SBOM

```bash
# CycloneDX format
trivy image --format cyclonedx --output sbom.json supertool:latest

# SPDX format
trivy image --format spdx-json --output sbom-spdx.json supertool:latest
```

### CI/CD Integration

The GitHub Actions workflow automatically:

- Builds Docker image
- Scans with Trivy for vulnerabilities
- Uploads SARIF to GitHub Security
- Fails on CRITICAL vulnerabilities
- Generates SBOM artifacts

---

## Production Deployment

### Container Registry

#### GitHub Container Registry

```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag
docker tag supertool:latest ghcr.io/username/supertool:latest
docker tag supertool:latest ghcr.io/username/supertool:1.0.0

# Push
docker push ghcr.io/username/supertool:latest
docker push ghcr.io/username/supertool:1.0.0
```

#### Docker Hub

```bash
# Login
docker login

# Tag
docker tag supertool:latest username/supertool:latest
docker tag supertool:latest username/supertool:1.0.0

# Push
docker push username/supertool:latest
docker push username/supertool:1.0.0
```

### Best Practices

#### Image Tagging

```bash
# Always use semantic versioning
docker tag supertool:latest supertool:1.0.0
docker tag supertool:latest supertool:1.0
docker tag supertool:latest supertool:1

# Include git commit
docker tag supertool:latest supertool:1.0.0-abc1234

# Environment-specific tags
docker tag supertool:latest supertool:1.0.0-prod
docker tag supertool:latest supertool:1.0.0-staging
```

#### Resource Limits

```bash
# Run with memory limit
docker run --rm \
  --memory=512m \
  --cpus=1 \
  supertool:latest

# In docker-compose.yml
services:
  supertool:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

#### Health Checks

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' supertool

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' supertool
```

---

## Kubernetes Deployment

### Basic Deployment

```yaml
# Coming in Phase 5: Kubernetes Native
# See: devops/kubernetes/ for manifests
```

### Using Helm

```bash
# Coming in Phase 5: Kubernetes Native
# See: DEVOPS_ROADMAP.md Phase 5
```

---

## Troubleshooting

### Common Issues

#### 1. Build Fails - Dependencies Missing

```bash
# Problem: npm install fails
# Solution: Clear npm cache and rebuild
docker build --no-cache -f devops/docker/Dockerfile -t supertool:latest .
```

#### 2. Permission Denied

```bash
# Problem: Cannot write to mounted volume
# Solution: Run with user mapping
docker run --rm \
  -v $(pwd)/outputs:/outputs \
  --user $(id -u):$(id -g) \
  supertool:latest
```

#### 3. Container Exits Immediately

```bash
# Debug: Run with shell
docker run --rm -it --entrypoint /bin/sh supertool:latest

# Check logs
docker logs <container-id>
```

#### 4. Image Too Large

```bash
# Check image size
docker images supertool:latest

# Analyze layers
docker history supertool:latest

# Use dive to inspect
dive supertool:latest
```

### Debugging

```bash
# Run with debug output
docker run --rm -e DEBUG=* supertool:latest

# Access running container
docker exec -it <container-id> /bin/sh

# Inspect container
docker inspect <container-id>

# View resource usage
docker stats <container-id>
```

### Performance

```bash
# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker build -f devops/docker/Dockerfile -t supertool:latest .

# Use layer caching
docker build --cache-from supertool:latest -t supertool:latest .

# Multi-stage build already optimized in Dockerfile
```

---

## Environment Variables

### Available Variables

- `NODE_ENV` - Node environment (production/development)
- `DEBUG` - Enable debug logging (\*)
- `LOG_LEVEL` - Logging level (info/debug/error)

### Setting Variables

```bash
# In docker run
docker run --rm -e NODE_ENV=production supertool:latest

# In docker-compose.yml
services:
  supertool:
    environment:
      - NODE_ENV=production
      - DEBUG=*

# From .env file
docker run --rm --env-file .env supertool:latest
```

---

## Advanced Usage

### Multi-Container Setup

```yaml
version: '3.9'
services:
  supertool:
    image: supertool:latest
    depends_on:
      - database
      - cache
    environment:
      - DATABASE_URL=postgres://db:5432/supertool
      - REDIS_URL=redis://cache:6379
    networks:
      - supertool-net

  database:
    image: postgres:15-alpine
    networks:
      - supertool-net

  cache:
    image: redis:7-alpine
    networks:
      - supertool-net

networks:
  supertool-net:
```

### Building from CI/CD

```yaml
# .github/workflows/docker-build.yml
- name: Build Docker Image
  uses: docker/build-push-action@v5
  with:
    context: .
    file: devops/docker/Dockerfile
    push: true
    tags: |
      ghcr.io/username/supertool:latest
      ghcr.io/username/supertool:${{ github.sha }}
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

---

## Next Steps

1. **Phase 2**: Infrastructure as Code
   - See `DEVOPS_ROADMAP.md` for Terraform/Ansible integration

2. **Phase 3**: GitOps
   - ArgoCD/FluxCD deployment
   - Kubernetes manifests

3. **Phase 5**: Kubernetes Native
   - Helm charts
   - Full orchestration support

---

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Trivy Security Scanner](https://aquasecurity.github.io/trivy/)
- [Docker BuildKit](https://docs.docker.com/build/buildkit/)

---

**Status**: Phase 1 Complete (Containerization)
**Next**: Phase 2 (Infrastructure as Code)
**Updated**: November 27, 2025
