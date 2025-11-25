# Docker Stack Quick Start

Multi-organization containerized development environment with 5 production services.

## Prerequisites

- Docker Desktop installed and running
- At least 8GB RAM available
- Ports 3000, 3001, 8080, 8081, 8888 available

## Quick Start

### 1. Configure Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your actual values
# (Optional - services will start with defaults)
```

### 2. Start All Services

```bash
# Production mode
docker compose up -d

# Development mode (with hot reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### 3. Verify Services

```bash
# Check status
docker compose ps

# View logs
docker compose logs -f

# Check health
docker compose ps --filter "health=healthy"
```

### 4. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| SimCore | http://localhost:3000 | React TypeScript app |
| repz | http://localhost:8080 | REPZ platform |
| benchbarrier | http://localhost:8081 | Benchmark UI |
| mag-logic | http://localhost:8888 | Jupyter notebook (Python) |
| Attributa | http://localhost:3001 | NLP platform |

## Common Commands

```bash
# Start specific services
docker compose up simcore repz

# Stop all services
docker compose down

# Rebuild after changes
docker compose up --build

# View logs for specific service
docker compose logs -f simcore

# Execute command in container
docker compose exec simcore sh

# Restart a service
docker compose restart simcore

# Remove volumes (warning: deletes data!)
docker compose down -v
```

## Development Workflow

### Making Code Changes

Development mode mounts your source code as volumes, enabling hot reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up simcore
```

Changes to files in `.config/organizations/AlaweinOS/SimCore/src` will automatically reload.

### Building Production Images

```bash
# Build all images
docker compose build

# Build specific image
docker compose build simcore

# Build without cache
docker compose build --no-cache
```

### Debugging

```bash
# View logs
docker compose logs -f simcore

# Execute shell in container
docker compose exec simcore sh

# Check resource usage
docker stats

# Inspect container
docker compose exec simcore env
```

## Network Architecture

```
frontend-network:
  - simcore (3000)
  - repz (8080)
  - benchbarrier (8081)
  - attributa (3001)

backend-network:
  - repz (8080)
  - benchbarrier (8081)
  - mag-logic (8888)
  - attributa (3001)

science-network:
  - mag-logic (8888)
```

Services on the same network can communicate using service names:
```javascript
// From simcore container
fetch('http://repz:8080/api/data')
```

## Volume Management

```bash
# List volumes
docker volume ls | grep GitHub

# Inspect volume
docker volume inspect simcore-data

# Backup volume
docker run --rm -v simcore-data:/data -v $(pwd):/backup alpine tar czf /backup/simcore-backup.tar.gz /data

# Restore volume
docker run --rm -v simcore-data:/data -v $(pwd):/backup alpine tar xzf /backup/simcore-backup.tar.gz -C /
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 3000
netstat -ano | findstr :3000

# Kill process (replace PID)
taskkill /F /PID <PID>
```

### Container Won't Start

```bash
# Check logs
docker compose logs simcore

# Rebuild without cache
docker compose build --no-cache simcore

# Start in foreground to see errors
docker compose up simcore
```

### Health Check Failing

```bash
# Check health check command
docker compose exec simcore sh
# Inside container:
wget -O- http://localhost:3000

# View health check logs
docker inspect --format='{{json .State.Health}}' simcore | jq
```

### Out of Disk Space

```bash
# Clean up old images
docker system prune -a

# Remove unused volumes
docker volume prune

# Check disk usage
docker system df
```

## Resource Limits

### Production Limits

| Service | CPU | Memory |
|---------|-----|--------|
| simcore | 1.0 | 512MB |
| repz | 1.0 | 512MB |
| benchbarrier | 1.0 | 512MB |
| mag-logic | 2.0 | 2GB |
| attributa | 1.5 | 768MB |

**Total**: 6.5 CPU, 4.3GB RAM

### Adjust Limits

Edit `docker-compose.yml`:

```yaml
services:
  simcore:
    deploy:
      resources:
        limits:
          cpus: '2.0'  # Increase CPU
          memory: 1G   # Increase memory
```

## CI/CD Integration

### GitHub Actions

Copy the CI/CD template to your project:

```bash
mkdir -p .github/workflows
cp .metaHub/templates/ci-cd/docker-ci.yml .github/workflows/docker-ci.yml
```

### Manual Build & Push

```bash
# Tag image
docker tag simcore:latest ghcr.io/username/simcore:latest

# Push to registry
docker push ghcr.io/username/simcore:latest
```

## Performance Optimization

### Build Cache

Docker Compose automatically uses BuildKit cache:

```bash
# Build with cache
docker compose build

# Force rebuild
docker compose build --no-cache
```

### Multi-Stage Builds

All Dockerfiles use multi-stage builds:
- Builder stage: Install deps, build app
- Production stage: Copy only artifacts

### Layer Caching

Order instructions by change frequency:
1. System dependencies (rarely change)
2. Package dependencies (change occasionally)
3. Source code (changes frequently)

## Security

### Scan Images

```bash
# Using Docker Scout
docker scout cves simcore:latest

# Using Trivy
trivy image simcore:latest
```

### Update Base Images

```bash
# Pull latest base images
docker compose pull

# Rebuild with new bases
docker compose build --pull
```

### Check for Secrets

```bash
# Never commit secrets!
git secrets --scan
```

## Documentation

- **CI/CD Pipeline**: [.metaHub/templates/ci-cd/README.md](.metaHub/templates/ci-cd/README.md)
- **Phase 2 Progress**: [.metaHub/docs/containerization/PHASE2_PROGRESS.md](.metaHub/docs/containerization/PHASE2_PROGRESS.md)
- **Metrics Comparison**: [.metaHub/docs/containerization/METRICS_COMPARISON.md](.metaHub/docs/containerization/METRICS_COMPARISON.md)
- **Build Superprompt**: [.metaHub/docs/DOCKER_BUILD_SUPERPROMPT.md](.metaHub/docs/DOCKER_BUILD_SUPERPROMPT.md)

## Support

For issues or questions:

1. Check logs: `docker compose logs -f`
2. Review documentation in `.metaHub/docs/`
3. Consult [Docker documentation](https://docs.docker.com/)
4. Check GitHub issues in repository

---

**Created**: 2025-11-24
**Phase**: 2 of 15 (Week 2)
**Status**: Production Ready
**Containerization Rate**: 61.3% (49/80 projects)
