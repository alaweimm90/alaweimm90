# HELIOS v0.1.0 Production Deployment Guide

**Version**: 0.1.0
**Status**: Production Ready
**Last Updated**: 2025-11-19

---

## Quick Start (5 minutes)

### Option 1: pip (Recommended for Users)
```bash
# Basic installation
pip install helios

# With specific domains
pip install helios[quantum,optimization]

# With all domains + dev tools
pip install helios[all,dev]

# Verify installation
python -c "import helios; print(f'HELIOS {helios.__version__} ready')"
```

### Option 2: Docker (Recommended for Deployment)
```bash
# Pull pre-built image
docker pull ghcr.io/alaweinoss/helios:v0.1.0

# Run interactive container
docker run -it ghcr.io/alaweinoss/helios:v0.1.0 /bin/bash

# Run CLI command
docker run ghcr.io/alaweinoss/helios:v0.1.0 helios domain list

# Run with local volume for persistent data
docker run -it -v $(pwd)/data:/app/data ghcr.io/alaweinoss/helios:v0.1.0
```

### Option 3: Source (For Developers)
```bash
# Clone repository
git clone https://github.com/alaweinoss/helios.git
cd helios

# Setup development environment
bash helios/scripts/setup.sh

# Verify
python -c "import helios; print(helios.__version__)"
```

---

## Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

**Key Variables:**
```bash
# LLM Provider (choose one)
LLM_PROVIDER=openai                    # openai, anthropic, google
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Logging
LOG_LEVEL=INFO                         # DEBUG, INFO, WARNING, ERROR
LOG_FILE=helios.log

# Performance
NUM_WORKERS=4                          # CPU workers
USE_GPU=true                           # Enable GPU acceleration
GPU_MEMORY_LIMIT=8                     # GB
```

---

## Local Deployment (Development)

### Single Machine Setup
```bash
# 1. Install dependencies
pip install -e ".[all,dev]"

# 2. Verify installation
bash helios/scripts/test.sh

# 3. Run examples
python helios/examples/basic_usage.py

# 4. Start CLI
helios --help
helios generate "machine learning"
```

### Docker Compose (Multi-Container)
```bash
# Start all services
cd helios/docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- `helios-core`: Main HELIOS service
- `helios-api`: FastAPI endpoint (optional)
- `helios-worker`: Background task processor
- `postgres`: Persistent storage

---

## Server Deployment

### Kubernetes (Recommended for Production)

```bash
# 1. Build and push Docker image
docker build -t myregistry/helios:v0.1.0 .
docker push myregistry/helios:v0.1.0

# 2. Create Kubernetes namespace
kubectl create namespace helios

# 3. Create ConfigMap from .env
kubectl create configmap helios-config --from-file=.env -n helios

# 4. Deploy using Helm (recommended) or kubectl
kubectl apply -f helios/k8s/deployment.yaml -n helios

# 5. Verify deployment
kubectl get pods -n helios
kubectl logs -f deployment/helios-core -n helios
```

**Example Kubernetes Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: helios-core
  namespace: helios
spec:
  replicas: 3
  selector:
    matchLabels:
      app: helios-core
  template:
    metadata:
      labels:
        app: helios-core
    spec:
      containers:
      - name: helios
        image: myregistry/helios:v0.1.0
        ports:
        - containerPort: 8000
        env:
        - name: LLM_PROVIDER
          valueFrom:
            configMapKeyRef:
              name: helios-config
              key: LLM_PROVIDER
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### AWS/GCP/Azure Deployment

**AWS ECS:**
```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name helios

# 2. Push image
docker tag helios:v0.1.0 <account-id>.dkr.ecr.<region>.amazonaws.com/helios:v0.1.0
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/helios:v0.1.0

# 3. Create ECS task definition (template provided)
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 4. Create ECS service
aws ecs create-service --cluster helios-prod --service-name helios-core --task-definition helios:1 --desired-count 3
```

**GCP Cloud Run:**
```bash
# 1. Push to Google Container Registry
docker tag helios:v0.1.0 gcr.io/<project-id>/helios:v0.1.0
docker push gcr.io/<project-id>/helios:v0.1.0

# 2. Deploy
gcloud run deploy helios \
  --image gcr.io/<project-id>/helios:v0.1.0 \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2
```

---

## Scaling Configuration

### Horizontal Scaling (Multiple Instances)

**Load Balancer Setup:**
```nginx
upstream helios_backend {
    server helios1:8000;
    server helios2:8000;
    server helios3:8000;
}

server {
    listen 80;
    server_name helios.example.com;

    location / {
        proxy_pass http://helios_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Auto-Scaling (Kubernetes):**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: helios-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: helios-core
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Scaling (Resource Limits)
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

---

## Monitoring & Logging

### Health Check
```bash
# Local
curl http://localhost:8000/health

# Kubernetes
kubectl exec -it deployment/helios-core -n helios -- \
  python -c "from helios import __version__; print(f'HELIOS {__version__} healthy')"
```

### Logs

```bash
# Local
tail -f helios.log

# Docker
docker logs -f <container-id>

# Kubernetes
kubectl logs -f deployment/helios-core -n helios
```

### Metrics

**Prometheus Integration:**
```python
# In your code
from prometheus_client import Counter, Histogram
import time

hypothesis_counter = Counter('helios_hypotheses_generated', 'Total hypotheses generated')
validation_histogram = Histogram('helios_validation_duration_seconds', 'Validation duration')

# Use in code
hypothesis_counter.inc()
with validation_histogram.time():
    result = validator.validate(hypothesis)
```

---

## Backup & Recovery

### Persistent Data
```bash
# Backup Hall of Failures database
docker exec helios-postgres pg_dump helios > backup.sql

# Backup configuration
cp .env env.backup

# Backup results
tar -czf results-backup.tar.gz data/
```

### Recovery
```bash
# Restore database
docker exec -i helios-postgres psql helios < backup.sql

# Restore configuration
cp env.backup .env

# Restore results
tar -xzf results-backup.tar.gz
```

---

## Performance Tuning

### CPU Optimization
```bash
# Set CPU affinity
export HELIOS_CPU_AFFINITY=0,1,2,3

# Use process pools
export NUM_WORKERS=4
```

### Memory Optimization
```bash
# Limit cache size
export CACHE_SIZE_GB=2

# Enable memory profiling
export HELIOS_PROFILE_MEMORY=true
```

### GPU Acceleration
```bash
# Enable CUDA
export USE_GPU=true
export CUDA_VISIBLE_DEVICES=0,1

# Set memory limit
export GPU_MEMORY_LIMIT=8
```

---

## Troubleshooting

### Installation Issues
```bash
# Clean install
pip uninstall helios -y
pip install --no-cache-dir helios[all,dev]

# Check versions
pip show helios
python -c "import helios; print(helios.__version__)"
```

### Runtime Issues
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check dependencies
python -c "from helios import check_dependencies; check_dependencies()"

# Run diagnostics
python helios/core/diagnostics.py
```

### Docker Issues
```bash
# Rebuild image
docker build --no-cache -t helios:v0.1.0 .

# Check image
docker images helios:v0.1.0

# Run with verbose output
docker run -it --env LOG_LEVEL=DEBUG helios:v0.1.0
```

---

## Security Considerations

### API Key Management
```bash
# Use environment variables (NOT in code)
export OPENAI_API_KEY='sk-...'

# Use .env file (never commit to git)
# .gitignore includes .env automatically
```

### Network Security
```bash
# Restrict to local network only
export HELIOS_BIND=127.0.0.1:8000

# Behind reverse proxy with HTTPS
# Use nginx or HAProxy with SSL termination
```

### Data Isolation
```bash
# Separate data volumes per deployment
docker volume create helios-data-prod
docker volume create helios-data-staging

# Use read-only mounts where possible
docker run -v helios-data-prod:/app/data:ro helios:v0.1.0
```

---

## Support & Documentation

### Getting Help
- **Documentation**: See [GETTING_STARTED.md](helios/docs/GETTING_STARTED.md)
- **API Reference**: See [API.md](helios/docs/API.md)
- **Architecture**: See [ARCHITECTURE.md](helios/docs/ARCHITECTURE.md)
- **Issues**: GitHub Issues page
- **Discussions**: GitHub Discussions

### Performance Benchmarks
- See [PERFORMANCE_METRICS.md](docs/PERFORMANCE_METRICS.md)
- Compare implementations: `bash helios/scripts/benchmark.sh`

---

**Next Steps:**
1. Choose deployment option (local, Docker, Kubernetes)
2. Configure environment variables
3. Run health check
4. Start generating hypotheses!

For questions, see GitHub Issues or CONTRIBUTING.md

---

**Version**: 0.1.0 MVP
**Status**: ✅ Production Ready
