# Production Deployment Best Practices for Librex.QAP-new

Comprehensive guide for deploying and maintaining Librex.QAP-new in production environments.

---

## Pre-Deployment Checklist

### Code Quality
- [ ] All tests passing (`make check-all`)
- [ ] Code coverage ≥ 40% overall, ≥ 91% critical paths
- [ ] No linting errors (`make lint`)
- [ ] Type checking passing (`mypy Librex.QAP/ ORCHEX/`)
- [ ] Documentation updated for all public APIs
- [ ] CHANGELOG.md updated with changes
- [ ] No debug statements or print() calls left in code

### Security
- [ ] All dependencies scanned for vulnerabilities
- [ ] Secrets NOT committed to git
- [ ] API authentication implemented
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Input validation on all endpoints
- [ ] SQL injection / command injection prevention reviewed
- [ ] SSL/TLS certificates configured

### Infrastructure
- [ ] Database backups automated
- [ ] Monitoring and alerting configured
- [ ] Log aggregation set up
- [ ] Performance tested under load
- [ ] Disaster recovery plan documented
- [ ] Scaling policies defined
- [ ] Resource quotas set
- [ ] Cost estimates reviewed

### Documentation
- [ ] Deployment guide written
- [ ] Runbook for common tasks created
- [ ] API documentation updated
- [ ] Troubleshooting guide provided
- [ ] Team trained on deployment procedures
- [ ] Access procedures documented

---

## Deployment Process

### 1. Pre-Deployment Testing

```bash
# Run full test suite
make check-all

# Test with Docker
docker build -t Librex.QAP-new:test .
docker run --rm Librex.QAP-new:test make test

# Load testing (optional)
pip install locust
locust -f tests/load_tests.py --host=http://localhost:8000
```

### 2. Gradual Rollout Strategy

**Phase 1: Canary Deployment (5% traffic)**
```bash
# Deploy to one instance
docker run -d --name Librex.QAP-canary Librex.QAP-new:latest

# Monitor metrics
watch -n 1 'curl http://localhost:8000/health'

# Check logs
docker logs -f Librex.QAP-canary
```

**Phase 2: Blue-Green Deployment (50% traffic)**
```bash
# Keep old version running (blue)
docker ps | grep Librex.QAP-blue

# Deploy new version (green)
docker run -d --name Librex.QAP-green Librex.QAP-new:latest

# Route traffic incrementally
# After validation, switch full traffic to green
```

**Phase 3: Full Deployment (100% traffic)**
```bash
# Decommission old version after validation period
docker stop Librex.QAP-blue
docker rm Librex.QAP-blue

# Rename green to production
docker rename Librex.QAP-green Librex.QAP-prod
```

### 3. Database Migration (if applicable)

```bash
# Create backup before migration
pg_dump production_db > backup_$(date +%Y%m%d).sql

# Run migrations
python -m alembic upgrade head

# Verify data integrity
python scripts/verify_db_integrity.py

# Run post-migration tests
pytest tests/integration/test_db_migrations.py
```

---

## Monitoring & Observability

### 1. Health Checks

Implement health check endpoints:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    """Basic health check."""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/ready")
def readiness_check():
    """Check if service is ready to handle requests."""
    try:
        # Check database connection
        db.execute("SELECT 1")

        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            return {"status": "not_ready", "reason": "High memory usage"}

        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest
    return generate_latest()
```

### 2. Key Metrics to Monitor

```python
import time
from prometheus_client import Counter, Histogram, Gauge

# API Metrics
api_requests = Counter(
    'Librex.QAP_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

api_latency = Histogram(
    'Librex.QAP_api_latency_seconds',
    'API request latency',
    ['endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

# Optimization Metrics
optimization_duration = Histogram(
    'Librex.QAP_optimization_duration_seconds',
    'Optimization task duration',
    ['method', 'problem_size']
)

optimization_quality = Gauge(
    'Librex.QAP_optimization_quality',
    'Best solution quality found',
    ['method']
)

# System Metrics
queue_length = Gauge(
    'Librex.QAP_queue_length',
    'Number of pending optimization tasks'
)

error_rate = Counter(
    'Librex.QAP_errors_total',
    'Total errors',
    ['error_type']
)
```

### 3. Alerting Rules

```yaml
# prometheus-rules.yml
groups:
  - name: Librex.QAP
    rules:
      - alert: HighErrorRate
        expr: rate(Librex.QAP_errors_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"

      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(Librex.QAP_api_latency_seconds_bucket[5m])) > 2.0
        for: 10m
        annotations:
          summary: "API latency is high"

      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / node_memory_MemTotal_bytes > 0.9
        for: 5m
        annotations:
          summary: "High memory usage"

      - alert: QueueBacklog
        expr: Librex.QAP_queue_length > 100
        for: 10m
        annotations:
          summary: "Task queue backlog building up"
```

---

## Logging Strategy

### 1. Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add custom fields
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id

        return json.dumps(log_data)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Example usage
logger.info("Optimization started", extra={
    'user_id': 'user123',
    'problem_size': 20,
    'method': 'fft_laplace'
})
```

### 2. Log Aggregation

```bash
# Using ELK Stack (Elasticsearch, Logstash, Kibana)
docker-compose -f elk-stack.yml up -d

# Ship logs to CloudWatch
pip install watchtower

import watchtower
logging.basicConfig(
    handlers=[watchtower.CloudWatchLogHandler()],
    level=logging.INFO
)
```

---

## Security Hardening

### 1. API Security

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address
import jwt

app = FastAPI()
security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)

# Rate limiting
@app.post("/solve")
@limiter.limit("10/minute")
async def solve(request: Request, problem: dict, credentials = Depends(security)):
    """Solve optimization problem with rate limiting."""

    # Validate token
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET,
            algorithms=["HS256"]
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Validate input
    if not isinstance(problem, dict):
        raise HTTPException(status_code=400, detail="Invalid problem format")

    if problem.get('size', 0) > 1000:
        raise HTTPException(status_code=400, detail="Problem too large")

    # Process request
    return solve_problem(problem)
```

### 2. Environment Variables

```bash
# .env (never commit this)
DATABASE_URL=postgresql://user:pass@host/db
JWT_SECRET=your-secret-key-min-32-chars
API_KEY=your-api-key
LOG_LEVEL=INFO
WORKERS=4

# Use python-dotenv
from dotenv import load_dotenv
import os

load_dotenv()
db_url = os.getenv("DATABASE_URL")
secret = os.getenv("JWT_SECRET")

# Verify required variables
required_vars = ["DATABASE_URL", "JWT_SECRET"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required env var: {var}")
```

### 3. Dependency Management

```bash
# Keep dependencies up to date
pip list --outdated

# Use pip-audit to check for vulnerabilities
pip install pip-audit
pip-audit --skip-editable

# Lock versions in production
pip freeze > requirements.lock.txt
pip install -r requirements.lock.txt
```

---

## Scaling & Performance

### 1. Horizontal Scaling

```bash
# Using Docker Swarm
docker swarm init
docker service create --name Librex.QAP-api \
  --replicas 3 \
  --publish 8000:8000 \
  Librex.QAP-new:latest

# Scale up if needed
docker service scale Librex.QAP-api=5
```

### 2. Caching Strategy

```python
from functools import lru_cache
from cachetools import TTLCache
import redis

# In-memory cache (single instance)
@lru_cache(maxsize=128)
def cached_solution(problem_hash, method):
    return solve_problem(problem_hash, method)

# Distributed cache (multiple instances)
redis_client = redis.Redis(host='localhost', port=6379)

def get_cached_result(key):
    value = redis_client.get(key)
    return json.loads(value) if value else None

def cache_result(key, value, ttl=3600):
    redis_client.setex(
        key,
        ttl,
        json.dumps(value)
    )
```

### 3. Load Balancing Configuration

```nginx
# nginx.conf
upstream Librex.QAP {
    least_conn;
    server Librex.QAP-1:8000;
    server Librex.QAP-2:8000;
    server Librex.QAP-3:8000;

    keepalive 32;
}

server {
    listen 80;
    server_name api.Librex.QAP.com;

    location / {
        proxy_pass http://Librex.QAP;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 30s;
    }

    location /health {
        access_log off;
        proxy_pass http://Librex.QAP;
    }
}
```

---

## Disaster Recovery

### 1. Backup Strategy

```bash
#!/bin/bash

# Daily backup script
BACKUP_DIR="/backups/Librex.QAP"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Backup database
pg_dump production_db | gzip > ${BACKUP_DIR}/db_${DATE}.sql.gz

# Backup configuration
tar -czf ${BACKUP_DIR}/config_${DATE}.tar.gz /app/config/

# Upload to S3
aws s3 cp ${BACKUP_DIR}/ s3://Librex.QAP-backups/ --recursive

# Clean old backups
find ${BACKUP_DIR} -name "*.gz" -mtime +${RETENTION_DAYS} -delete

echo "Backup completed: ${DATE}"
```

### 2. Recovery Procedure

```bash
#!/bin/bash

# Restore from backup
BACKUP_FILE="db_20240101_000000.sql.gz"
BACKUP_DIR="/backups/Librex.QAP"

# Stop application
systemctl stop Librex.QAP-api

# Restore database
gunzip < ${BACKUP_DIR}/${BACKUP_FILE} | psql production_db

# Verify restoration
psql production_db -c "SELECT COUNT(*) FROM results;"

# Restart application
systemctl start Librex.QAP-api

# Verify application
curl http://localhost:8000/health
```

---

## Maintenance Tasks

### Daily
- [ ] Monitor error logs
- [ ] Check system resource usage
- [ ] Verify backup completion

### Weekly
- [ ] Review performance metrics
- [ ] Check for security updates
- [ ] Verify backup restoration

### Monthly
- [ ] Update dependencies (non-critical)
- [ ] Review access logs for anomalies
- [ ] Test disaster recovery
- [ ] Generate performance reports

### Quarterly
- [ ] Major security audit
- [ ] Capacity planning
- [ ] Update documentation
- [ ] Team training review

---

## Troubleshooting Guide

### Issue: High Memory Usage

```python
# Identify memory leaks
import tracemalloc

tracemalloc.start()

# Run operations
for i in range(1000):
    solve_problem(problem)

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.2f}MB; Peak: {peak / 1024 / 1024:.2f}MB")
```

### Issue: Slow Optimization

```bash
# Profile the code
python -m cProfile -s cumulative app.py > profile.txt
head -50 profile.txt

# Use py-spy for production profiling
pip install py-spy
py-spy record -o profile.svg -- python app.py
```

### Issue: Database Connection Issues

```python
# Test connection
import sqlalchemy
engine = sqlalchemy.create_engine(DATABASE_URL)
with engine.connect() as conn:
    result = conn.execute("SELECT 1")
    print("Database connection OK")

# Check connection pool
print(f"Pool size: {engine.pool.size()}")
print(f"Checked in: {engine.pool.checkedin()}")
print(f"Checked out: {engine.pool.checkedout()}")
```

---

## Performance Optimization Tips

1. **Batch Operations**: Process multiple problems together
2. **Connection Pooling**: Reuse database connections
3. **Caching**: Cache frequently accessed data
4. **Async/Await**: Use async operations for I/O
5. **Index Database**: Optimize database queries
6. **Compression**: Compress large responses
7. **CDN**: Serve static files from CDN
8. **Rate Limiting**: Prevent resource exhaustion

---

**Last Updated:** November 2024
