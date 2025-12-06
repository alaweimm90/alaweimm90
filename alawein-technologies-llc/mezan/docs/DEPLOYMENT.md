# MEZAN ORCHEX Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Deployment Methods](#deployment-methods)
5. [Configuration](#configuration)
6. [Security](#security)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Overview

MEZAN ORCHEX is a production-grade multi-agent orchestration platform designed for enterprise-scale deployments. This guide covers deployment strategies, configurations, and operational procedures.

### Key Features

- **Container-based deployment** using Docker
- **Horizontal scaling** with load balancing
- **High availability** with redundancy
- **Comprehensive monitoring** with Prometheus & Grafana
- **Automated backups** and disaster recovery
- **Security-hardened** configurations

## Prerequisites

### System Requirements

#### Minimum (Development)

- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- OS: Ubuntu 20.04+ / RHEL 8+ / macOS 12+
- Docker: 20.10+
- Docker Compose: 2.0+

#### Recommended (Production)

- CPU: 8+ cores
- RAM: 32GB+
- Storage: 500GB+ SSD (RAID recommended)
- OS: Ubuntu 22.04 LTS / RHEL 9
- Docker: Latest stable
- Docker Compose: Latest v2
- Kubernetes: 1.25+ (optional)

### Software Dependencies

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install supporting tools
sudo apt-get update
sudo apt-get install -y \
    git \
    curl \
    jq \
    make \
    python3-pip \
    postgresql-client \
    redis-tools

# Install AWS CLI (for S3 backups)
pip3 install awscli

# Install monitoring tools (optional)
sudo apt-get install -y htop iotop nethogs
```

### Network Requirements

- **Inbound Ports:**
  - 80/443: HTTP/HTTPS (Web traffic)
  - 8080: ORCHEX API
  - 3000: Grafana Dashboard
  - 9090: Prometheus Metrics

- **Outbound Connections:**
  - Anthropic API: api.anthropic.com
  - OpenAI API: api.openai.com
  - Docker Hub: hub.docker.com
  - S3: *.amazonaws.com (if using S3)

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Load Balancer                        │
│                        (Nginx)                           │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
    ┌────▼─────┐              ┌─────▼────┐
    │  ORCHEX   │              │  ORCHEX   │
    │  API #1  │              │  API #2  │
    └────┬─────┘              └─────┬────┘
         │                           │
         └─────────────┬─────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
    ┌────▼─────┐              ┌─────▼────┐
    │  Redis   │              │PostgreSQL│
    │  Cache   │              │    DB    │
    └──────────┘              └──────────┘
         │                           │
         └─────────────┬─────────────┘
                       │
    ┌──────────────────┴──────────────────┐
    │                                      │
┌───▼────┐    ┌────────────┐    ┌─────────▼──┐
│Prometheus│  │   Grafana   │    │   Backup   │
│ Metrics │   │  Dashboard  │    │   Service  │
└─────────┘   └─────────────┘    └────────────┘
```

### Service Definitions

| Service | Purpose | Port | Replicas |
|---------|---------|------|----------|
| ORCHEX-api | Main API server | 8080 | 2-10 |
| redis | Cache & message queue | 6379 | 1 |
| postgres | Persistent storage | 5432 | 1 |
| nginx | Load balancer | 80/443 | 1 |
| prometheus | Metrics collection | 9090 | 1 |
| grafana | Visualization | 3000 | 1 |

## Deployment Methods

### 1. Quick Start (Development)

```bash
# Clone repository
git clone https://github.com/AlaweinOS/MEZAN.git
cd MEZAN/ORCHEX/ORCHEX-core

# Copy environment template
cp .env.example .env

# Edit configuration
vi .env

# Start services
docker-compose -f docker-compose.dev.yml up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f ORCHEX-api-dev
```

### 2. Production Deployment

#### Step 1: Prepare Environment

```bash
# Create deployment directory
sudo mkdir -p /opt/ORCHEX
sudo chown $USER:$USER /opt/ORCHEX
cd /opt/ORCHEX

# Clone repository
git clone https://github.com/AlaweinOS/MEZAN.git .
cd ORCHEX/ORCHEX-core

# Create production environment file
cat > .env.production <<EOF
# API Keys (REQUIRED)
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key

# Database
POSTGRES_USER=ORCHEX
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=ORCHEX

# Redis
REDIS_PASSWORD=$(openssl rand -base64 32)

# Security
JWT_SECRET_KEY=$(openssl rand -base64 64)
SECRET_KEY=$(openssl rand -base64 32)

# S3 Backup (Optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=ORCHEX-backups

# Monitoring
SLACK_WEBHOOK=https://hooks.slack.com/services/...
EMAIL_TO=ops@yourdomain.com
EOF

# Secure the file
chmod 600 .env.production
```

#### Step 2: Build Images

```bash
# Build production images
./scripts/build.sh --target production --version v3.5.0

# Verify images
docker images | grep ORCHEX
```

#### Step 3: Deploy Services

```bash
# Deploy with production configuration
export ENVIRONMENT=production
./scripts/deploy.sh production

# Monitor deployment
watch docker-compose ps

# Check health
curl http://localhost:8080/health
```

#### Step 4: Configure SSL/TLS

```bash
# Generate SSL certificate (using Let's Encrypt)
sudo apt-get install certbot
sudo certbot certonly --standalone -d ORCHEX.yourdomain.com

# Update nginx configuration
cat > config/nginx.conf <<'EOF'
server {
    listen 80;
    server_name ORCHEX.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ORCHEX.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/ORCHEX.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ORCHEX.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://atlas_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Restart nginx
docker-compose restart nginx
```

### 3. Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace ORCHEX

# Create secrets
kubectl create secret generic ORCHEX-secrets \
    --from-env-file=.env.production \
    -n ORCHEX

# Apply manifests
kubectl apply -f k8s/ -n ORCHEX

# Check deployment
kubectl get pods -n ORCHEX
kubectl get svc -n ORCHEX

# Port forward for testing
kubectl port-forward -n ORCHEX svc/ORCHEX-api 8080:8080
```

### 4. Cloud Deployments

#### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
docker tag mezan-ORCHEX:latest $ECR_REGISTRY/mezan-ORCHEX:latest
docker push $ECR_REGISTRY/mezan-ORCHEX:latest

# Deploy with CloudFormation
aws cloudformation create-stack \
    --stack-name ORCHEX-production \
    --template-body file://cloudformation/ORCHEX-ecs.yml \
    --parameters file://cloudformation/parameters.json
```

#### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/$PROJECT_ID/mezan-ORCHEX

# Deploy to Cloud Run
gcloud run deploy ORCHEX-api \
    --image gcr.io/$PROJECT_ID/mezan-ORCHEX \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | Yes | - |
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `POSTGRES_USER` | PostgreSQL username | No | ORCHEX |
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes | - |
| `REDIS_PASSWORD` | Redis password | No | - |
| `JWT_SECRET_KEY` | JWT signing key | Yes | - |
| `LOG_LEVEL` | Logging level | No | INFO |
| `WORKERS` | Number of workers | No | 4 |
| `MAX_REPLICAS` | Max service replicas | No | 10 |

### Configuration Files

- **production.yaml**: Main production configuration
- **prometheus.yml**: Metrics collection configuration
- **grafana/**: Dashboard provisioning
- **nginx.conf**: Load balancer configuration

### Performance Tuning

```yaml
# Optimize for high throughput
performance:
  workers: 8
  threads: 4
  connection_pool: 100
  cache_ttl: 3600
  batch_size: 100

# Optimize for low latency
performance:
  workers: 4
  threads: 2
  connection_pool: 50
  cache_ttl: 300
  batch_size: 10
```

## Security

### Security Checklist

- [ ] Change all default passwords
- [ ] Enable SSL/TLS encryption
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Set up intrusion detection
- [ ] Implement rate limiting
- [ ] Configure CORS properly
- [ ] Enable security headers
- [ ] Regular security updates
- [ ] Backup encryption

### Hardening Steps

```bash
# 1. Secure Docker daemon
cat > /etc/docker/daemon.json <<EOF
{
    "icc": false,
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "5"
    },
    "userland-proxy": false,
    "no-new-privileges": true
}
EOF

# 2. Set up firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# 3. Enable audit logging
sudo apt-get install auditd
sudo systemctl enable auditd
sudo systemctl start auditd

# 4. Configure fail2ban
sudo apt-get install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

## Monitoring

### Metrics

Access Prometheus metrics at: http://localhost:9090

Key metrics to monitor:
- `atlas_api_requests_total`: Total API requests
- `atlas_api_request_duration_seconds`: Request latency
- `atlas_agent_execution_time`: Agent processing time
- `atlas_cache_hit_ratio`: Cache effectiveness
- `atlas_db_connections_active`: Database connections

### Dashboards

Access Grafana at: http://localhost:3000 (admin/admin)

Pre-configured dashboards:
- System Overview
- API Performance
- Agent Analytics
- Resource Usage
- Error Tracking

### Alerts

Configure alerts in `config/alerts.yml`:

```yaml
groups:
  - name: atlas_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(atlas_api_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected

      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes > 2147483648
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: Container memory usage is high
```

### Logging

```bash
# View logs
docker-compose logs -f ORCHEX-api

# Aggregate logs with Loki (optional)
docker run -d --name loki -p 3100:3100 grafana/loki

# Configure log shipping
cat > promtail-config.yml <<EOF
clients:
  - url: http://localhost:3100/loki/api/v1/push
scrape_configs:
  - job_name: ORCHEX
    static_configs:
      - targets:
          - localhost
        labels:
          job: ORCHEX
          __path__: /app/logs/*.log
EOF
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check logs
docker-compose logs ORCHEX-api

# Common fixes:
# - Check environment variables
# - Verify database connectivity
# - Check port availability
```

#### 2. Database Connection Issues

```bash
# Test database connection
docker-compose exec postgres pg_isready

# Check credentials
docker-compose exec postgres psql -U ORCHEX -c "SELECT 1"

# Reset database
docker-compose down -v
docker-compose up -d postgres
docker-compose exec postgres psql -U ORCHEX -f /docker-entrypoint-initdb.d/init.sql
```

#### 3. High Memory Usage

```bash
# Check container stats
docker stats

# Limit memory usage
docker-compose down
# Edit docker-compose.yml to add memory limits
docker-compose up -d
```

#### 4. API Performance Issues

```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null http://localhost:8080/health

# Scale up services
./scripts/scale.sh ORCHEX-api 3 scale

# Clear cache
docker-compose exec redis redis-cli FLUSHALL
```

### Diagnostic Commands

```bash
# Health check
curl http://localhost:8080/health

# Service status
docker-compose ps

# Resource usage
docker stats --no-stream

# Network connectivity
docker-compose exec ORCHEX-api ping -c 3 redis
docker-compose exec ORCHEX-api ping -c 3 postgres

# Database status
docker-compose exec postgres pg_stat_activity

# Redis status
docker-compose exec redis redis-cli INFO

# Check logs
docker-compose logs --tail=100 ORCHEX-api
```

## Best Practices

### 1. Deployment

- Always use version tags, never `latest` in production
- Implement blue-green deployments for zero-downtime updates
- Use health checks and readiness probes
- Maintain separate configurations for each environment
- Document all configuration changes

### 2. Security

- Rotate API keys and passwords regularly
- Use secrets management systems (Vault, AWS Secrets Manager)
- Enable audit logging for compliance
- Implement network segmentation
- Regular security scanning with Trivy/Clair

### 3. Monitoring

- Set up comprehensive alerting
- Monitor business metrics, not just system metrics
- Implement distributed tracing for debugging
- Regular performance profiling
- Maintain SLO/SLA dashboards

### 4. Backup & Recovery

- Automated daily backups
- Test restore procedures regularly
- Maintain off-site backup copies
- Document recovery procedures
- Practice disaster recovery drills

### 5. Scaling

- Start with vertical scaling, then horizontal
- Implement auto-scaling based on metrics
- Use connection pooling for databases
- Cache aggressively but invalidate properly
- Monitor and optimize slow queries

### 6. Operations

```bash
# Daily tasks
- Check monitoring dashboards
- Review error logs
- Verify backup completion
- Check security alerts

# Weekly tasks
- Review performance metrics
- Update dependencies
- Test failover procedures
- Review capacity planning

# Monthly tasks
- Security patching
- Disaster recovery testing
- Performance optimization review
- Cost optimization review
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/AlaweinOS/MEZAN/issues
- Documentation: https://docs.mezan.io
- Email: support@mezan.io

---

**Last Updated:** November 2024
**Version:** 3.5.0
**Maintainer:** MEZAN Development Team