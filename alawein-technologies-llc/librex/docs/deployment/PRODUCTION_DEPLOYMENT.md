# Production Deployment Guide

Complete guide for deploying the enhanced Librex.QAP-new server in production environments.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Configuration](#environment-configuration)
3. [Local Production Setup](#local-production-setup)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Performance Tuning](#performance-tuning)
8. [Security Hardening](#security-hardening)
9. [Backup & Recovery](#backup--recovery)
10. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4 GB
- **Storage**: 10 GB
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.9+

### Recommended Requirements
- **CPU**: 4-8 cores
- **RAM**: 8-16 GB
- **Storage**: 50 GB SSD
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11

### Network
- **Ports**: 8000 (API), 9090 (Prometheus, optional)
- **Bandwidth**: 100 Mbps+
- **SSL/TLS**: Required for production

---

## Environment Configuration

### 1. Create Environment File

Create `.env.production`:

```bash
# Server Configuration
QAP_ENV=production
QAP_HOST=0.0.0.0
QAP_PORT=8000
QAP_WORKERS=4
QAP_LOG_LEVEL=info

# Security
QAP_CORS_ORIGINS=https://yourdomain.com,https://dashboard.yourdomain.com
QAP_ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Rate Limiting
QAP_RATE_LIMIT_CAPACITY=100
QAP_RATE_LIMIT_REFILL_RATE=1.0

# Cache Configuration
QAP_CACHE_MAX_SIZE=1000
QAP_CACHE_DEFAULT_TTL=300

# Circuit Breaker
QAP_CIRCUIT_BREAKER_THRESHOLD=5
QAP_CIRCUIT_BREAKER_TIMEOUT=60

# Monitoring
QAP_METRICS_ENABLED=true
QAP_PROMETHEUS_PORT=9090

# Performance
QAP_REQUEST_TIMEOUT=300
QAP_MAX_REQUEST_SIZE=10485760  # 10MB
```

### 2. Load Environment Variables

Add to server startup:

```python
import os
from dotenv import load_dotenv

load_dotenv('.env.production')

# Use in configuration
WORKERS = int(os.getenv('QAP_WORKERS', 4))
CORS_ORIGINS = os.getenv('QAP_CORS_ORIGINS', '*').split(',')
```

---

## Local Production Setup

### 1. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install system tools
sudo apt install -y nginx supervisor
```

### 2. Create Application User

```bash
# Create dedicated user
sudo useradd -m -s /bin/bash Librex.QAP

# Switch to user
sudo su - Librex.QAP
```

### 3. Setup Application

```bash
# Create directories
mkdir -p /home/Librex.QAP/app
mkdir -p /home/Librex.QAP/logs
cd /home/Librex.QAP/app

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements_enhanced.txt
pip install gunicorn  # Production server
```

### 4. Configure Gunicorn

Create `gunicorn_config.py`:

```python
"""Gunicorn configuration for production."""

import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 300
keepalive = 5

# Logging
accesslog = "/home/Librex.QAP/logs/access.log"
errorlog = "/home/Librex.QAP/logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "Librex.QAP-api"

# Server mechanics
daemon = False
pidfile = "/home/Librex.QAP/app/gunicorn.pid"
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if terminating SSL at application)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
```

### 5. Configure Supervisor

Create `/etc/supervisor/conf.d/Librex.QAP.conf`:

```ini
[program:Librex.QAP]
command=/home/Librex.QAP/app/venv/bin/gunicorn -c /home/Librex.QAP/app/gunicorn_config.py server_enhanced:app
directory=/home/Librex.QAP/app
user=Librex.QAP
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/home/Librex.QAP/logs/supervisor_error.log
stdout_logfile=/home/Librex.QAP/logs/supervisor_out.log
environment=PATH="/home/Librex.QAP/app/venv/bin"
```

Start service:

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start Librex.QAP
sudo supervisorctl status Librex.QAP
```

### 6. Configure Nginx

Create `/etc/nginx/sites-available/Librex.QAP`:

```nginx
upstream Librex.QAP_backend {
    server 127.0.0.1:8000 fail_timeout=0;
}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Client body size limit
    client_max_body_size 10M;

    # Timeouts
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;

    # Logging
    access_log /var/log/nginx/Librex.QAP_access.log;
    error_log /var/log/nginx/Librex.QAP_error.log;

    location / {
        proxy_pass http://Librex.QAP_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static files (if serving dashboard)
    location /dashboard {
        alias /home/Librex.QAP/dashboard;
        try_files $uri $uri/ /index.html;
    }

    # Health check endpoint (bypass proxy for nginx monitoring)
    location /health {
        access_log off;
        proxy_pass http://Librex.QAP_backend;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/Librex.QAP /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Docker Deployment

### 1. Create Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN groupadd -r Librex.QAP && useradd -r -g Librex.QAP Librex.QAP

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_enhanced.txt .
RUN pip install --no-cache-dir -r requirements_enhanced.txt \
    && pip install --no-cache-dir gunicorn

# Copy application
COPY server_enhanced.py .
COPY gunicorn_config.py .

# Create necessary directories
RUN mkdir -p /app/logs && chown -R Librex.QAP:Librex.QAP /app

# Switch to app user
USER Librex.QAP

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["gunicorn", "-c", "gunicorn_config.py", "server_enhanced:app"]
```

### 2. Create Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    image: Librex.QAP-enhanced:latest
    container_name: Librex.QAP-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - QAP_ENV=production
      - QAP_WORKERS=4
      - QAP_LOG_LEVEL=info
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    networks:
      - Librex.QAP-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    container_name: Librex.QAP-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - api
    networks:
      - Librex.QAP-network

  prometheus:
    image: prom/prometheus:latest
    container_name: Librex.QAP-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - Librex.QAP-network

  grafana:
    image: grafana/grafana:latest
    container_name: Librex.QAP-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=changeme
    depends_on:
      - prometheus
    networks:
      - Librex.QAP-network

networks:
  Librex.QAP-network:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
```

### 3. Build and Run

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Check status
docker-compose ps

# Stop services
docker-compose down
```

### 4. Docker Commands

```bash
# Build image
docker build -t Librex.QAP-enhanced:latest .

# Run container
docker run -d \
  --name Librex.QAP-api \
  -p 8000:8000 \
  -e QAP_WORKERS=4 \
  -v $(pwd)/logs:/app/logs \
  Librex.QAP-enhanced:latest

# View logs
docker logs -f Librex.QAP-api

# Execute commands in container
docker exec -it Librex.QAP-api bash

# Stop container
docker stop Librex.QAP-api

# Remove container
docker rm Librex.QAP-api
```

---

## Kubernetes Deployment

### 1. Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: Librex.QAP
```

### 2. Create ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: Librex.QAP-config
  namespace: Librex.QAP
data:
  QAP_ENV: "production"
  QAP_WORKERS: "4"
  QAP_LOG_LEVEL: "info"
  QAP_RATE_LIMIT_CAPACITY: "100"
  QAP_CACHE_MAX_SIZE: "1000"
```

### 3. Create Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: Librex.QAP-api
  namespace: Librex.QAP
  labels:
    app: Librex.QAP
    component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: Librex.QAP
      component: api
  template:
    metadata:
      labels:
        app: Librex.QAP
        component: api
    spec:
      containers:
      - name: api
        image: Librex.QAP-enhanced:latest
        imagePullPolicy: Always
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        envFrom:
        - configMapRef:
            name: Librex.QAP-config
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
```

### 4. Create Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: Librex.QAP-service
  namespace: Librex.QAP
  labels:
    app: Librex.QAP
spec:
  type: LoadBalancer
  selector:
    app: Librex.QAP
    component: api
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
```

### 5. Create Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: Librex.QAP-ingress
  namespace: Librex.QAP
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: Librex.QAP-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: Librex.QAP-service
            port:
              number: 80
```

### 6. Create HPA (Horizontal Pod Autoscaler)

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: Librex.QAP-hpa
  namespace: Librex.QAP
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: Librex.QAP-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

### 7. Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f namespace.yaml

# Create ConfigMap
kubectl apply -f configmap.yaml

# Create Deployment
kubectl apply -f deployment.yaml

# Create Service
kubectl apply -f service.yaml

# Create Ingress
kubectl apply -f ingress.yaml

# Create HPA
kubectl apply -f hpa.yaml

# Check status
kubectl get all -n Librex.QAP

# View logs
kubectl logs -f deployment/Librex.QAP-api -n Librex.QAP

# Scale deployment
kubectl scale deployment Librex.QAP-api --replicas=5 -n Librex.QAP

# Delete all resources
kubectl delete namespace Librex.QAP
```

---

## Monitoring Setup

### 1. Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'Librex.QAP-prod'

scrape_configs:
  - job_name: 'Librex.QAP-api'
    scrape_interval: 15s
    metrics_path: '/metrics/prometheus'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          service: 'Librex.QAP-api'
          environment: 'production'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

### 2. Grafana Dashboards

Import dashboard JSON or create panels:

**Request Rate Panel**:
```promql
rate(qap_requests_total[5m])
```

**Error Rate Panel**:
```promql
rate(qap_errors_total[5m]) / rate(qap_requests_total[5m]) * 100
```

**Response Time (p95)**:
```promql
qap_request_duration_seconds{quantile="0.95"}
```

**Active Requests**:
```promql
qap_active_requests
```

---

## Performance Tuning

### 1. Worker Configuration

```python
# Calculate optimal workers
import multiprocessing

# CPU-bound workloads
workers = multiprocessing.cpu_count() * 2 + 1

# I/O-bound workloads
workers = multiprocessing.cpu_count() * 4
```

### 2. Cache Tuning

```python
# Production cache settings
state.cache = Cache(
    max_size=10000,     # Increase cache size
    default_ttl=600     # Increase TTL
)
```

### 3. Rate Limit Tuning

```python
# Adjust based on capacity
state.rate_limiter = RateLimiter(
    capacity=200,       # Higher capacity for production
    refill_rate=2.0     # Faster refill
)
```

### 4. Database Connection Pooling (if using)

```python
# SQLAlchemy example
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_recycle=3600
)
```

---

## Security Hardening

### 1. Configure CORS for Production

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://dashboard.yourdomain.com"
    ],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### 2. Add Authentication Middleware

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

# Use as dependency
@app.post("/solve", dependencies=[Depends(verify_api_key)])
async def solve_problem(request: SolveRequest):
    ...
```

### 3. Enable HTTPS Only

```python
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app.add_middleware(HTTPSRedirectMiddleware)
```

---

## Backup & Recovery

### 1. Backup Request History

```bash
# Export data
curl http://localhost:8000/analytics/export/json > backup_$(date +%Y%m%d).json

# Schedule daily backups
0 2 * * * /path/to/backup_script.sh
```

### 2. Database Backups (if using)

```bash
# PostgreSQL example
pg_dump -U Librex.QAP Librex.QAP_db > backup_$(date +%Y%m%d).sql

# Restore
psql -U Librex.QAP Librex.QAP_db < backup_20251119.sql
```

---

## Troubleshooting

### High Memory Usage

```bash
# Check memory
curl http://localhost:8000/admin/status | jq '.memory'

# Reduce cache size
# Edit server_enhanced.py: Cache(max_size=500)

# Restart service
sudo supervisorctl restart Librex.QAP
```

### Circuit Breaker Stuck Open

```bash
# Check status
curl http://localhost:8000/admin/status | jq '.circuit_breaker'

# Reset
curl -X POST http://localhost:8000/admin/circuit-breaker/reset
```

### High Error Rate

```bash
# Check metrics
curl http://localhost:8000/metrics | jq '.error_rate'

# Check logs
tail -f /home/Librex.QAP/logs/error.log

# Review recent errors
curl http://localhost:8000/analytics/history | jq '.history[] | select(.error != null)'
```

---

**Production deployment complete!** Monitor your deployment at `/metrics` and `/health` endpoints.
