# Docker CI/CD Pipeline Templates

Reusable GitHub Actions workflows for building, testing, and deploying Docker containers with comprehensive security scanning and quality checks.

## Overview

This template provides a complete CI/CD pipeline with 6 stages:

1. **Validation** - Lint Dockerfiles with Hadolint
2. **Build** - Multi-stage Docker builds with caching
3. **Security** - Vulnerability scanning with Trivy and Snyk
4. **Quality** - Image size checks and best practices
5. **Deploy** - Environment-based deployments
6. **Notification** - Status notifications

## Quick Start

### 1. Copy Template to Your Project

```bash
# From repository root
mkdir -p .github/workflows
cp .metaHub/templates/ci-cd/docker-ci.yml .github/workflows/docker-ci.yml
```

### 2. Customize for Your Project

Edit `.github/workflows/docker-ci.yml` and update:

```yaml
env:
  REGISTRY: ghcr.io  # or docker.io, gcr.io, etc.
  IMAGE_NAME: ${{ github.repository }}
  NODE_VERSION: '20'  # or '18', '22', etc.
  PYTHON_VERSION: '3.11'  # if using Python
```

### 3. Configure Secrets

Add the following secrets to your GitHub repository:

- `GITHUB_TOKEN` - Automatically provided by GitHub Actions
- `SNYK_TOKEN` - (Optional) Get from [snyk.io](https://snyk.io)

**Settings → Secrets and variables → Actions → New repository secret**

### 4. Trigger Pipeline

Push to main branch or create a pull request:

```bash
git add .github/workflows/docker-ci.yml
git commit -m "feat: add Docker CI/CD pipeline"
git push origin main
```

## Pipeline Stages

### Stage 1: Validation

Validates Dockerfile syntax and best practices using Hadolint.

**What it checks:**
- FROM instruction best practices
- Layer optimization
- Security vulnerabilities
- Deprecated instructions
- Missing health checks

**Failure threshold:** Warning level

### Stage 2: Build

Builds Docker image with multi-stage optimization and layer caching.

**Features:**
- Docker Buildx for multi-platform builds
- GitHub Actions cache integration
- Automatic tagging (branch, PR, semver, SHA)
- Container startup test

**Outputs:**
- `image-digest` - SHA256 digest of built image
- `image-tag` - Full image tag with registry path

### Stage 3: Security Scanning

Scans container for vulnerabilities using industry-standard tools.

**Tools:**
- **Trivy** - Comprehensive vulnerability scanner (free)
- **Snyk** - Advanced security analysis (requires token)

**Severity levels:**
- CRITICAL - Must fix immediately
- HIGH - Should fix before production
- MEDIUM - Fix when possible
- LOW - Optional fixes

**Results uploaded to:**
- GitHub Security tab
- SARIF reports in workflow artifacts

### Stage 4: Quality Checks

Ensures Docker image meets quality standards.

**Checks:**
- Image size (warns if > 1GB)
- Multi-stage build usage
- Health check configuration
- Layer count optimization

### Stage 5: Deployment

Deploys to staging or production environments.

**Environment options:**
- `staging` - Default for all pushes
- `production` - Manual trigger or tag-based

**Deployment targets (customize as needed):**
- Kubernetes (kubectl)
- Docker Compose
- AWS ECS
- Azure Container Instances
- Google Cloud Run

**Built-in features:**
- Environment-specific configuration
- Rollback on failure
- Health check verification

### Stage 6: Notifications

Sends status notifications on pipeline completion.

**Integration options (add as needed):**
- Slack
- Discord
- Microsoft Teams
- Email
- Custom webhooks

## Customization Examples

### Node.js Application

```yaml
env:
  NODE_VERSION: '20'

jobs:
  build:
    steps:
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          build-args: |
            NODE_VERSION=${{ env.NODE_VERSION }}
```

### Python Application

```yaml
env:
  PYTHON_VERSION: '3.11'

jobs:
  build:
    steps:
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          build-args: |
            PYTHON_VERSION=${{ env.PYTHON_VERSION }}
```

### Multiple Dockerfiles

```yaml
strategy:
  matrix:
    service:
      - name: frontend
        dockerfile: Dockerfile.frontend
      - name: backend
        dockerfile: Dockerfile.backend

steps:
  - name: Build ${{ matrix.service.name }}
    uses: docker/build-push-action@v5
    with:
      context: .
      file: ${{ matrix.service.dockerfile }}
```

### Custom Deployment

Replace the deploy job with your specific deployment commands:

```yaml
deploy:
  steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/my-app \
          app=${{ needs.build.outputs.image-tag }} \
          --record

    - name: Wait for rollout
      run: |
        kubectl rollout status deployment/my-app \
          --timeout=5m
```

## Environment Configuration

### Staging Environment

Create `.github/workflows/staging.env`:

```env
ENVIRONMENT=staging
API_URL=https://api.staging.example.com
DATABASE_URL=${{ secrets.STAGING_DB_URL }}
```

### Production Environment

Create `.github/workflows/production.env`:

```env
ENVIRONMENT=production
API_URL=https://api.example.com
DATABASE_URL=${{ secrets.PRODUCTION_DB_URL }}
```

## Workflow Triggers

### Automatic Triggers

```yaml
on:
  push:
    branches: [main, develop]
    paths:
      - 'src/**'
      - 'Dockerfile'
```

### Manual Triggers

Run from GitHub Actions tab with custom parameters:

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options: [staging, production]
```

### Tag-Based Release

```yaml
on:
  push:
    tags:
      - 'v*.*.*'
```

## Security Best Practices

### 1. Never Commit Secrets

Use GitHub Secrets for sensitive data:

```yaml
env:
  API_KEY: ${{ secrets.API_KEY }}
```

### 2. Use Minimal Base Images

```dockerfile
FROM node:20-alpine  # Not node:20 (smaller)
FROM python:3.11-slim  # Not python:3.11 (smaller)
```

### 3. Scan Before Deployment

Never deploy if security scan fails:

```yaml
deploy:
  needs: [build, security]  # Requires security to pass
```

### 4. Use Read-Only Containers

```dockerfile
USER node  # Don't run as root
RUN chmod -R 555 /app  # Read-only files
```

### 5. Implement Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1
```

## Monitoring & Debugging

### View Pipeline Status

GitHub → Actions tab → Select workflow run

### Download Artifacts

1. Go to workflow run
2. Scroll to "Artifacts" section
3. Download SARIF reports, logs, etc.

### View Security Findings

GitHub → Security tab → Code scanning alerts

### Debug Failed Builds

```yaml
- name: Debug Docker build
  if: failure()
  run: |
    docker images
    docker ps -a
    docker logs $(docker ps -aq | head -1)
```

## Performance Optimization

### Build Cache

Uses GitHub Actions cache automatically:

```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```

### Layer Optimization

Order Dockerfile instructions by change frequency:

```dockerfile
# 1. System dependencies (rarely change)
RUN apt-get update && apt-get install -y ...

# 2. Application dependencies (change occasionally)
COPY package*.json ./
RUN npm ci

# 3. Source code (changes frequently)
COPY . .
```

### Parallel Jobs

Jobs run in parallel by default:

```
validate → build → [security, quality] → deploy
```

## Troubleshooting

### Build Fails on Hadolint

Check Dockerfile against [best practices](https://github.com/hadolint/hadolint#rules).

Common issues:
- Missing `USER` instruction
- Using `latest` tag
- Not pinning package versions

### Trivy Shows Vulnerabilities

1. Update base image: `FROM node:20-alpine` → `FROM node:20.11-alpine`
2. Update dependencies: `npm update` or `pip install --upgrade`
3. Check [CVE database](https://cve.mitre.org) for severity

### Deployment Fails

1. Check environment secrets are configured
2. Verify deployment target is accessible
3. Review deployment logs in workflow run

### Image Too Large

1. Use multi-stage builds
2. Use Alpine base images
3. Remove build dependencies in production stage
4. Use `.dockerignore` to exclude files

## Related Resources

- [Dockerfile Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Hadolint Rules](https://github.com/hadolint/hadolint#rules)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)

## Support

For issues or questions about this template:

1. Check existing issues in repository
2. Review GitHub Actions logs
3. Consult Docker and GitHub Actions documentation
4. Open a new issue with:
   - Workflow YAML
   - Error messages
   - Dockerfile content
   - Steps to reproduce
