# Foundry Infrastructure Documentation

## Overview

This document provides comprehensive documentation for the Foundry platform infrastructure, including Docker configurations, CI/CD pipelines, deployment strategies, monitoring, and operational procedures.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Environment](#development-environment)
3. [Production Environment](#production-environment)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [Deployment Procedures](#deployment-procedures)
6. [Monitoring & Logging](#monitoring--logging)
7. [Security](#security)
8. [Disaster Recovery](#disaster-recovery)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Applications

The Foundry platform consists of three main applications:

1. **Ghost Researcher** (Port 3000) - Paranormal activity investigation platform
2. **Scientific Tinder** (Port 3001) - Scientific idea matching platform
3. **Chaos Engine** (Port 3002) - Experimental idea combination engine

### Infrastructure Components

- **Frontend**: Next.js 14 applications with TypeScript
- **Backend API**: Node.js REST API (Port 8000)
- **Database**: PostgreSQL 15 with read replicas
- **Cache**: Redis 7 cluster
- **Load Balancer**: Nginx
- **CDN**: CloudFront
- **Container Orchestration**: Docker Swarm / ECS
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)

### Network Architecture

```
Internet
    ↓
CloudFront CDN
    ↓
WAF
    ↓
Load Balancer (Nginx)
    ↓
├── Ghost Researcher (3000)
├── Scientific Tinder (3001)
├── Chaos Engine (3002)
└── API Backend (8000)
    ↓
├── PostgreSQL Primary
├── PostgreSQL Replica
└── Redis Cluster
```

---

## Development Environment

### Prerequisites

- Docker 24.0+
- Docker Compose 2.20+
- Node.js 20 LTS
- npm 10+
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/Foundry.git
cd Foundry

# Copy environment variables
cp .env.example .env

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Access applications
# Ghost Researcher: http://localhost:3000
# Scientific Tinder: http://localhost:3001
# Chaos Engine: http://localhost:3002
# API: http://localhost:8000
# Adminer: http://localhost:8080
# Mailhog: http://localhost:8025
```

### Development Database

```bash
# Access PostgreSQL
docker exec -it postgres-dev psql -U postgres -d Foundry

# Run migrations
docker-compose exec api npm run migrate

# Seed database
docker-compose exec api npm run seed
```

### Hot Reloading

All applications support hot reloading in development mode. Changes to source files will automatically trigger rebuilds.

---

## Production Environment

### Infrastructure Requirements

- **Compute**:
  - 3x t3.medium instances for applications
  - 2x t3.large instances for database
  - 1x t3.medium for monitoring
- **Storage**:
  - 100GB SSD for database
  - 50GB for application logs
- **Network**:
  - Multi-AZ deployment
  - Private subnets for backend services
  - Public subnets for load balancers

### Production Deployment

```bash
# Deploy to production
./infrastructure/scripts/deploy.sh production v1.0.0

# Verify deployment
docker ps
curl https://api.Foundry.com/health
```

### SSL Certificates

SSL certificates are managed through Let's Encrypt with automatic renewal:

```bash
# Generate certificates
certbot certonly --webroot -w /var/www/certbot \
  -d Foundry.com \
  -d ghost-researcher.Foundry.com \
  -d scientific-tinder.Foundry.com \
  -d chaos-engine.Foundry.com \
  -d api.Foundry.com

# Auto-renewal cron job
0 0 * * 0 certbot renew --quiet
```

---

## CI/CD Pipeline

### Workflow Overview

1. **Pull Request**
   - Code quality checks (ESLint, Prettier, TypeScript)
   - Unit tests (Jest)
   - Component tests
   - E2E tests (Playwright)
   - Security scanning (npm audit, Snyk, OWASP)
   - Accessibility tests
   - Build validation
   - Bundle size analysis

2. **Merge to Main**
   - Deploy to staging
   - Run full E2E test suite
   - Performance benchmarks
   - Security scan on staging

3. **Release Tag**
   - Build production images
   - Push to registry
   - Deploy to production
   - Health checks
   - Smoke tests

### GitHub Actions Configuration

Workflows are located in `.github/workflows/`:
- `pull-request.yml` - PR validation
- `deploy.yml` - Deployment pipeline

### Build Performance Targets

- Build time: <5 minutes
- Test execution: <10 minutes
- Deployment: <10 minutes
- Rollback: <2 minutes

---

## Deployment Procedures

### Standard Deployment

```bash
# Check prerequisites
./infrastructure/scripts/deploy.sh staging check

# Deploy to staging
./infrastructure/scripts/deploy.sh staging v1.2.3

# Run smoke tests
npm run test:smoke:staging

# Deploy to production
./infrastructure/scripts/deploy.sh production v1.2.3
```

### Blue-Green Deployment

```bash
# Deploy new version to green environment
docker-compose -f docker-compose.green.yml up -d

# Switch traffic
./infrastructure/scripts/switch_traffic.sh green

# Verify
curl https://Foundry.com/api/version

# Remove blue environment
docker-compose -f docker-compose.blue.yml down
```

### Rollback Procedure

```bash
# Fast rollback (switch images)
./infrastructure/scripts/rollback.sh production previous fast

# Safe rollback (blue-green switch)
./infrastructure/scripts/rollback.sh production previous safe

# With database restore
./infrastructure/scripts/rollback.sh production previous safe --restore-db
```

### Health Checks

All services expose health endpoints:

```bash
# Check individual services
curl http://localhost:3000/api/health
curl http://localhost:3001/api/health
curl http://localhost:3002/api/health
curl http://localhost:8000/health

# Check all services
./infrastructure/scripts/health_check.sh
```

---

## Monitoring & Logging

### Metrics Collection

**Prometheus** collects metrics from all services:

```bash
# Access Prometheus
http://localhost:9090

# Example queries
rate(http_requests_total[5m])
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
up{job="api-backend"}
```

### Visualization

**Grafana** dashboards available at `http://localhost:3003`:

- System Overview
- Application Performance
- Database Metrics
- Cache Performance
- Business Metrics
- Security Dashboard

Default credentials: `admin/admin`

### Log Aggregation

**ELK Stack** for centralized logging:

```bash
# Access Kibana
http://localhost:5601

# View logs
GET /Foundry-*/_search
{
  "query": {
    "match": {
      "level": "error"
    }
  }
}
```

### Alerting

Alerts are configured in `infrastructure/monitoring/alert-rules.yml`:

- High error rate (>5%)
- Service down
- High memory usage (>90%)
- High CPU usage (>80%)
- Database replication lag
- SSL certificate expiry

### Custom Metrics

```javascript
// Application metrics example
const prometheus = require('prom-client');

const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status']
});

// Track request
const end = httpRequestDuration.startTimer();
// ... handle request
end({ method: 'GET', route: '/api/users', status: 200 });
```

---

## Security

### Security Measures

1. **Network Security**
   - Private VPC with isolated subnets
   - Security groups with minimal permissions
   - WAF rules for common attacks
   - DDoS protection

2. **Application Security**
   - JWT authentication
   - Rate limiting
   - Input validation
   - SQL injection prevention
   - XSS protection
   - CSRF tokens

3. **Container Security**
   - Non-root users in containers
   - Read-only root filesystems
   - Security scanning with Trivy
   - Minimal base images (Alpine)

4. **Secrets Management**
   - Environment variables for configuration
   - AWS Secrets Manager for production
   - Encrypted at rest
   - Regular rotation

### Security Scanning

```bash
# Scan Docker images
docker scan ghost-researcher:latest

# Dependency scanning
npm audit
snyk test

# OWASP dependency check
dependency-check --project Foundry --scan .
```

### Compliance

- GDPR compliant data handling
- Regular security audits
- Penetration testing quarterly
- SOC 2 Type II certification (in progress)

---

## Disaster Recovery

### Backup Strategy

1. **Database Backups**
   - Full backup: Daily at 2 AM UTC
   - Incremental: Every 6 hours
   - Transaction logs: Continuous
   - Retention: 30 days
   - Off-site storage: AWS S3

2. **Application State**
   - Configuration backups: Daily
   - Docker images: Tagged and stored in registry
   - Environment variables: Encrypted backups

### Recovery Procedures

#### Database Recovery

```bash
# List available backups
aws s3 ls s3://Foundry-backups/postgres/

# Restore from backup
./infrastructure/scripts/restore_database.sh 20240315

# Verify restoration
docker exec postgres-primary psql -U postgres -c "SELECT COUNT(*) FROM shared.users;"
```

#### Full System Recovery

```bash
# 1. Provision infrastructure
terraform apply -var-file=production.tfvars

# 2. Restore database
./infrastructure/scripts/restore_database.sh latest

# 3. Deploy applications
./infrastructure/scripts/deploy.sh production last-known-good

# 4. Verify system
./infrastructure/scripts/verify_recovery.sh
```

### RTO and RPO Targets

- **RTO** (Recovery Time Objective): 2 hours
- **RPO** (Recovery Point Objective): 1 hour

---

## Performance Optimization

### Application Level

1. **Frontend Optimization**
   - Code splitting
   - Lazy loading
   - Image optimization (WebP, AVIF)
   - Service workers for offline support
   - CDN for static assets

2. **Backend Optimization**
   - Database query optimization
   - Connection pooling
   - Response caching
   - Pagination
   - Async processing for heavy tasks

### Infrastructure Level

1. **Scaling Strategy**
   - Horizontal scaling for applications
   - Read replicas for database
   - Redis cluster for caching
   - Auto-scaling based on metrics

2. **Performance Monitoring**
   ```bash
   # Monitor response times
   curl -w "@curl-format.txt" -o /dev/null -s https://api.Foundry.com/health

   # Load testing
   artillery run load-test.yml

   # Performance profiling
   clinic doctor -- node server.js
   ```

### Performance Targets

- Page Load Time: <2s
- API Response Time: <200ms (p95)
- Lighthouse Score: >90
- Core Web Vitals:
  - LCP: <2.5s
  - FID: <100ms
  - CLS: <0.1

---

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker logs <container_name>

# Inspect container
docker inspect <container_name>

# Check resource usage
docker stats
```

#### 2. Database Connection Issues

```bash
# Test connection
docker exec api nc -zv postgres 5432

# Check credentials
docker exec api env | grep DATABASE_URL

# Verify database is running
docker exec postgres-primary pg_isready
```

#### 3. High Memory Usage

```bash
# Find memory consumers
docker stats --no-stream

# Check Node.js memory
docker exec <container> node -e "console.log(process.memoryUsage())"

# Analyze heap dump
docker exec <container> node --inspect=0.0.0.0:9229
```

#### 4. Slow Performance

```bash
# Check CPU usage
top
htop

# Monitor network
iftop
nethogs

# Database slow queries
docker exec postgres-primary psql -c "SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"
```

### Debug Mode

Enable debug logging:

```bash
# Set debug environment
export DEBUG=*
export LOG_LEVEL=debug

# Restart services
docker-compose restart
```

### Support Channels

- **Slack**: #Foundry-support
- **Email**: devops@Foundry.com
- **On-call**: PagerDuty
- **Documentation**: https://docs.Foundry.com

---

## Appendix

### Useful Commands

```bash
# View all running containers
docker ps

# Follow logs for all services
docker-compose logs -f

# Execute command in container
docker exec -it <container> /bin/sh

# Clean up Docker resources
docker system prune -af

# Check disk usage
df -h
du -sh /var/lib/docker

# Network diagnostics
netstat -tulpn
ss -tulpn

# Process monitoring
ps aux | grep node
pstree -p

# Database operations
pg_dump -U postgres Foundry > backup.sql
psql -U postgres Foundry < backup.sql
```

### Environment Variables

Key environment variables used across the platform:

| Variable | Description | Default |
|----------|-------------|---------|
| NODE_ENV | Environment mode | development |
| DATABASE_URL | PostgreSQL connection string | - |
| REDIS_URL | Redis connection string | - |
| JWT_SECRET | JWT signing secret | - |
| CORS_ORIGINS | Allowed CORS origins | - |
| LOG_LEVEL | Logging verbosity | info |
| PORT | Application port | varies |

### File Structure

```
infrastructure/
├── database/
│   ├── init.sql
│   └── migrations/
├── docker/
│   ├── Dockerfile.*
│   └── docker-compose.*.yml
├── monitoring/
│   ├── prometheus.yml
│   ├── alert-rules.yml
│   └── grafana/
├── nginx/
│   └── nginx.conf
├── scripts/
│   ├── deploy.sh
│   ├── rollback.sh
│   └── health_check.sh
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   └── modules/
└── README.md
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-03-15 | Initial infrastructure setup |
| 1.1.0 | 2024-03-20 | Added monitoring stack |
| 1.2.0 | 2024-03-25 | Implemented auto-scaling |
| 1.3.0 | 2024-03-30 | Enhanced security measures |

---

**Last Updated**: March 30, 2024
**Maintained By**: DevOps Team
**License**: MIT