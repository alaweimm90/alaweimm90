# Librex.QAP-new Enhanced Server Guide

## Overview

Production-grade FastAPI server with enterprise features for Quadratic Assignment Problem optimization.

**Version**: 2.0.0
**File**: `server_enhanced.py`

---

## Features

### 1. Advanced Error Handling

#### Custom Error Types
```python
- ValidationError: Input validation failures with recovery suggestions
- TimeoutError: Request timeout with optimization recommendations
- ServerError: Internal errors with contact information
- RateLimitError: Rate limit exceeded with retry-after timing
- CircuitBreakerError: Service unavailable due to high error rate
```

#### Error Response Format
```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Matrix size 5x5 doesn't match problem_size 10",
  "suggestion": "Ensure matrix has 10x10 dimensions",
  "timestamp": "2025-11-19T10:30:00",
  "request_id": "uuid-here",
  "details": {}
}
```

### 2. Security Enhancements

#### Rate Limiting
- **Algorithm**: Token bucket with automatic refill
- **Default**: 100 requests per minute per IP
- **Headers**:
  - `X-RateLimit-Limit`: Maximum requests
  - `X-RateLimit-Reset`: Time until bucket refill

#### Input Validation
- Matrix bounds checking (2-1000 dimensions)
- Size limit enforcement
- Type validation for all numeric values
- Square matrix verification

#### Security Headers
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000
Content-Security-Policy: default-src 'self'
```

#### CORS Configuration
- Configurable origins
- Credential support
- Exposed custom headers
- Method restrictions

### 3. Performance Features

#### Response Caching
```python
# Automatic caching with TTL
- Cache key: SHA256 hash of request parameters
- Default TTL: 300 seconds (5 minutes)
- LRU eviction: 1000 entry maximum
- Deduplication: Identical in-flight requests share results
```

#### Compression
- GZip middleware for responses > 1KB
- Automatic content negotiation

#### Async Operations
- Background task processing
- Non-blocking optimization execution
- Thread pool for CPU-intensive work

### 4. Advanced API Features

#### Batch Solving
```bash
POST /solve/batch
```

**Request**:
```json
{
  "problems": [
    {
      "problem_size": 5,
      "problem_matrix": [[...]],
      "method": "fft_laplace"
    },
    {
      "problem_size": 10,
      "problem_matrix": [[...]],
      "method": "genetic_algorithm"
    }
  ],
  "parallel": true
}
```

**Response**:
```json
{
  "batch_id": "uuid",
  "total_problems": 2,
  "successful": 2,
  "failed": 0,
  "total_runtime_seconds": 3.45,
  "results": [...]
}
```

#### Async Solve with Polling
```bash
# 1. Submit async request
POST /solve-async
{
  "problem_matrix": [[...]],
  "problem_size": 100,
  "webhook_url": "https://your-service.com/webhook"
}

# Response
{
  "request_id": "uuid",
  "status": "queued",
  "poll_url": "/solve/{request_id}/status"
}

# 2. Poll for status
GET /solve/{request_id}/status

# Response (in progress)
{
  "request_id": "uuid",
  "status": "running",
  "progress": 45,
  "method": "fft_laplace"
}

# Response (completed)
{
  "request_id": "uuid",
  "status": "completed",
  "result": {...}
}
```

#### Webhook Support
Automatically POST results to webhook URL when async operation completes:
```json
POST https://your-service.com/webhook
{
  "request_id": "uuid",
  "method": "fft_laplace",
  "objective_value": 1234.56,
  "best_solution": [0, 1, 2, 3, 4],
  ...
}
```

### 5. Data Management

#### Request History
```bash
GET /analytics/history?limit=100&method=fft_laplace
```

#### Export Endpoints

**CSV Export**:
```bash
GET /analytics/export/csv
# Downloads: history.csv
```

**JSON Export**:
```bash
GET /analytics/export/json
```

#### Data Cleanup
Automatic cleanup on server shutdown:
- Clears active requests
- Flushes cache
- Saves completed request history

### 6. Monitoring & Observability

#### Health Check
```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2025-11-19T10:30:00",
  "uptime_seconds": 3600.5,
  "components": {
    "api": "ok",
    "optimization_engine": "ok",
    "cache": "ok",
    "rate_limiter": "ok",
    "circuit_breaker": "closed"
  },
  "metrics": {
    "total_requests": 1234,
    "error_rate": 0.02,
    "cache_size": 456,
    "active_requests": 5,
    "avg_response_time_ms": 125.3
  }
}
```

#### Prometheus Metrics
```bash
GET /metrics/prometheus
```

**Output**:
```prometheus
# HELP qap_requests_total Total number of requests
# TYPE qap_requests_total counter
qap_requests_total 1234

# HELP qap_errors_total Total number of errors
# TYPE qap_errors_total counter
qap_errors_total 25

# HELP qap_request_duration_seconds Request duration
# TYPE qap_request_duration_seconds summary
qap_request_duration_seconds{quantile="0.5"} 0.125
qap_request_duration_seconds{quantile="0.95"} 0.450
qap_request_duration_seconds{quantile="0.99"} 0.890

# HELP qap_uptime_seconds Server uptime
# TYPE qap_uptime_seconds gauge
qap_uptime_seconds 3600.5
```

#### Performance Metrics
```bash
GET /metrics
```

**Response**:
```json
{
  "uptime_seconds": 3600.5,
  "total_requests": 1234,
  "total_errors": 25,
  "error_rate": 0.02,
  "requests_per_second": 0.34,
  "avg_response_time_ms": 125.3,
  "p50_ms": 125.0,
  "p95_ms": 450.0,
  "p99_ms": 890.0,
  "method_counts": {
    "fft_laplace": 456,
    "genetic_algorithm": 234
  },
  "status_counts": {
    "200": 1180,
    "400": 15,
    "429": 10,
    "500": 4
  }
}
```

#### Detailed Statistics
```bash
GET /stats
```

### 7. Production Features

#### Graceful Shutdown
- SIGINT/SIGTERM signal handling
- Active request completion
- Cache flush
- Connection cleanup
- Background task cancellation

#### Circuit Breaker
```python
# Automatic circuit breaking on high error rates
- Failure threshold: 5 consecutive failures
- Timeout: 60 seconds
- States: closed, open, half-open
- Auto-recovery with exponential backoff
```

**Check Status**:
```bash
GET /admin/status
```

**Reset Circuit Breaker**:
```bash
POST /admin/circuit-breaker/reset
```

#### Request Deduplication
- Identical in-flight requests share results
- SHA256 hash-based detection
- Automatic result distribution

---

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements_enhanced.txt
```

### 2. Start Server

**Development**:
```bash
python server_enhanced.py
```

**Production with Uvicorn**:
```bash
uvicorn server_enhanced:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

**Production with Gunicorn**:
```bash
gunicorn server_enhanced:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

---

## Configuration

### Environment Variables

```bash
# Server Configuration
export QAP_HOST="0.0.0.0"
export QAP_PORT="8000"
export QAP_WORKERS="4"

# Rate Limiting
export QAP_RATE_LIMIT_CAPACITY="100"
export QAP_RATE_LIMIT_REFILL_RATE="1.0"

# Cache Configuration
export QAP_CACHE_MAX_SIZE="1000"
export QAP_CACHE_DEFAULT_TTL="300"

# Circuit Breaker
export QAP_CIRCUIT_BREAKER_THRESHOLD="5"
export QAP_CIRCUIT_BREAKER_TIMEOUT="60"

# Security
export QAP_CORS_ORIGINS="https://yourdomain.com,https://dashboard.yourdomain.com"
```

### Code Configuration

Edit `server_enhanced.py`:

```python
# Rate Limiter Configuration
state = ServerState()
state.rate_limiter = RateLimiter(
    capacity=100,      # Max tokens in bucket
    refill_rate=1.0   # Tokens per second
)

# Cache Configuration
state.cache = Cache(
    max_size=1000,     # Max entries
    default_ttl=300    # Default TTL in seconds
)

# Circuit Breaker Configuration
state.circuit_breaker = CircuitBreaker(
    failure_threshold=5,  # Failures before opening
    timeout=60           # Seconds before retry
)
```

---

## API Examples

### Basic Solve
```python
import requests

response = requests.post(
    "http://localhost:8000/solve",
    json={
        "problem_size": 5,
        "problem_matrix": [
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0]
        ],
        "method": "fft_laplace",
        "iterations": 1000,
        "timeout_seconds": 300
    }
)

result = response.json()
print(f"Best solution: {result['best_solution']}")
print(f"Objective value: {result['objective_value']}")
print(f"Runtime: {result['runtime_seconds']}s")
```

### Batch Solve
```python
import requests

response = requests.post(
    "http://localhost:8000/solve/batch",
    json={
        "parallel": True,
        "problems": [
            {
                "problem_size": 5,
                "problem_matrix": [[...]],
                "method": "fft_laplace"
            },
            {
                "problem_size": 5,
                "problem_matrix": [[...]],
                "method": "genetic_algorithm"
            }
        ]
    }
)

batch_result = response.json()
print(f"Successful: {batch_result['successful']}")
print(f"Failed: {batch_result['failed']}")
```

### Async Solve with Webhook
```python
import requests

# Submit async request
response = requests.post(
    "http://localhost:8000/solve-async",
    json={
        "problem_size": 100,
        "problem_matrix": [[...]],
        "method": "fft_laplace",
        "webhook_url": "https://your-service.com/webhook"
    }
)

request_id = response.json()["request_id"]

# Poll for status
while True:
    status = requests.get(
        f"http://localhost:8000/solve/{request_id}/status"
    ).json()

    if status["status"] == "completed":
        result = status["result"]
        break

    time.sleep(1)
```

---

## Monitoring Integration

### Prometheus Integration

**prometheus.yml**:
```yaml
scrape_configs:
  - job_name: 'Librex.QAP'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/prometheus'
```

### Grafana Dashboard

**Query Examples**:
```promql
# Request rate
rate(qap_requests_total[5m])

# Error rate
rate(qap_errors_total[5m]) / rate(qap_requests_total[5m])

# Response time p95
qap_request_duration_seconds{quantile="0.95"}

# Uptime
qap_uptime_seconds
```

---

## Admin Operations

### Clear All State
```bash
curl -X POST http://localhost:8000/admin/clear
```

### Reset Circuit Breaker
```bash
curl -X POST http://localhost:8000/admin/circuit-breaker/reset
```

### Check Admin Status
```bash
curl http://localhost:8000/admin/status
```

**Response**:
```json
{
  "service": "healthy",
  "version": "2.0.0",
  "uptime_seconds": 3600.5,
  "requests": {
    "active": 5,
    "completed": 1234,
    "total": 1239,
    "errors": 25
  },
  "cache": {
    "size": 456,
    "max_size": 1000,
    "hit_rate": 0.78
  },
  "circuit_breaker": {
    "state": "closed",
    "failures": 0
  },
  "memory": {
    "rss_mb": 145.6,
    "vms_mb": 892.3
  }
}
```

---

## Error Handling

### Validation Error
```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Matrix size 5x5 doesn't match problem_size 10",
  "suggestion": "Ensure matrix has 10x10 dimensions",
  "timestamp": "2025-11-19T10:30:00",
  "request_id": "uuid"
}
```

### Timeout Error
```json
{
  "error_code": "TIMEOUT_ERROR",
  "message": "Optimization exceeded timeout of 300s",
  "suggestion": "Try reducing problem size, iterations, or increase timeout",
  "timestamp": "2025-11-19T10:30:00",
  "request_id": "uuid"
}
```

### Rate Limit Error
```json
{
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded for IP 192.168.1.1",
  "suggestion": "Retry after 60 seconds",
  "timestamp": "2025-11-19T10:30:00",
  "request_id": "uuid"
}
```
**Response Headers**:
```
Retry-After: 60
```

### Circuit Breaker Open
```json
{
  "error_code": "CIRCUIT_BREAKER_OPEN",
  "message": "Service temporarily unavailable due to high error rate",
  "suggestion": "Service temporarily unavailable, try again in a few minutes",
  "timestamp": "2025-11-19T10:30:00",
  "request_id": "uuid"
}
```

---

## Performance Tuning

### Worker Configuration

**CPU-Bound Workloads**:
```bash
# Workers = (2 x CPU cores) + 1
uvicorn server_enhanced:app --workers 9  # For 4-core system
```

**I/O-Bound Workloads**:
```bash
# More workers for I/O operations
uvicorn server_enhanced:app --workers 16
```

### Cache Tuning

```python
# Larger cache for high-traffic scenarios
state.cache = Cache(
    max_size=10000,    # Increase cache size
    default_ttl=600    # Increase TTL
)
```

### Rate Limit Tuning

```python
# Stricter rate limiting
state.rate_limiter = RateLimiter(
    capacity=50,       # Reduce capacity
    refill_rate=0.5    # Slower refill
)
```

---

## Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 10 -p request.json -T application/json \
   http://localhost:8000/solve

# Using wrk
wrk -t4 -c100 -d30s --latency \
    -s post.lua http://localhost:8000/solve
```

### Integration Tests
```python
import pytest
import httpx

@pytest.mark.asyncio
async def test_solve_endpoint():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/solve",
            json={
                "problem_size": 5,
                "problem_matrix": [[0, 1, 2, 3, 4]] * 5,
                "method": "fft_laplace"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert "best_solution" in result
        assert len(result["best_solution"]) == 5
```

---

## Migration from Original Server

### Backward Compatibility

The enhanced server is **100% backward compatible** with the original `server.py`:

- All original endpoints preserved
- Same request/response formats
- Same data models
- Dashboard integration works without changes

### New Features (Optional)

New features are opt-in:
- Batch solving: Use `/solve/batch` instead of `/solve`
- Webhooks: Add `webhook_url` to request
- Priority: Add `priority` field to request

### Migration Steps

1. **Install new requirements**:
   ```bash
   pip install -r requirements_enhanced.txt
   ```

2. **Test with existing clients**:
   ```bash
   # Start enhanced server
   python server_enhanced.py

   # Test with dashboard (should work unchanged)
   # Test with existing API clients
   ```

3. **Gradually adopt new features**:
   - Enable monitoring endpoints
   - Configure rate limiting
   - Add webhook support
   - Use batch endpoints

---

## Production Deployment

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_enhanced.txt .
RUN pip install --no-cache-dir -r requirements_enhanced.txt

COPY server_enhanced.py .

EXPOSE 8000

CMD ["uvicorn", "server_enhanced:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Build and Run**:
```bash
docker build -t Librex.QAP-enhanced .
docker run -p 8000:8000 Librex.QAP-enhanced
```

### Kubernetes Deployment

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: Librex.QAP-enhanced
spec:
  replicas: 3
  selector:
    matchLabels:
      app: Librex.QAP
  template:
    metadata:
      labels:
        app: Librex.QAP
    spec:
      containers:
      - name: Librex.QAP
        image: Librex.QAP-enhanced:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: Librex.QAP-service
spec:
  selector:
    app: Librex.QAP
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Troubleshooting

### High Memory Usage

**Check Cache Size**:
```bash
curl http://localhost:8000/admin/status | jq '.cache'
```

**Reduce Cache**:
```python
state.cache = Cache(max_size=500, default_ttl=180)
```

### Circuit Breaker Stuck Open

**Check Status**:
```bash
curl http://localhost:8000/admin/status | jq '.circuit_breaker'
```

**Reset**:
```bash
curl -X POST http://localhost:8000/admin/circuit-breaker/reset
```

### Rate Limit Issues

**Check Bucket Status**:
```bash
curl http://localhost:8000/admin/status | jq '.rate_limiter'
```

**Adjust Limits**:
```python
state.rate_limiter = RateLimiter(capacity=200, refill_rate=2.0)
```

---

## Support

For issues, questions, or contributions:
- Check logs: `journalctl -u Librex.QAP -f`
- Monitor metrics: `/metrics/prometheus`
- Admin status: `/admin/status`
- Health check: `/health`

---

**Version**: 2.0.0
**Last Updated**: 2025-11-19
