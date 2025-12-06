# Quick Start Guide - Enhanced Server

Get up and running with the production-grade Librex.QAP-new Enhanced Server in 5 minutes.

---

## Installation

### 1. Install Dependencies
```bash
cd /home/user/2-CLAUDE-CODE/Librex.QAP-new
pip install -r requirements_enhanced.txt
```

### 2. Start the Server
```bash
python server_enhanced.py
```

You should see:
```
======================================================================
Librex.QAP-new Enhanced API Server Starting
======================================================================
Version: 2.0.0 (Production-Grade)
Available methods: 8
Rate limiting: 100 req/min per IP
Cache size: 1000 entries
Documentation: http://localhost:8000/docs
Metrics: http://localhost:8000/metrics/prometheus
Health check: http://localhost:8000/health
======================================================================
```

### 3. Verify Server is Running
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2025-11-19T10:30:00",
  "uptime_seconds": 1.23,
  "components": {
    "api": "ok",
    "optimization_engine": "ok",
    "cache": "ok",
    "rate_limiter": "ok",
    "circuit_breaker": "closed"
  },
  "metrics": {
    "total_requests": 0,
    "error_rate": 0.0,
    "cache_size": 0,
    "active_requests": 0,
    "avg_response_time_ms": 0.0
  }
}
```

---

## Basic Usage

### Example 1: Solve a QAP Problem

**Python**:
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
        "iterations": 1000
    }
)

result = response.json()
print(f"Best solution: {result['best_solution']}")
print(f"Objective value: {result['objective_value']}")
print(f"Runtime: {result['runtime_seconds']}s")
```

**curl**:
```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "problem_size": 5,
    "problem_matrix": [
      [0, 1, 2, 3, 4],
      [1, 0, 1, 2, 3],
      [2, 1, 0, 1, 2],
      [3, 2, 1, 0, 1],
      [4, 3, 2, 1, 0]
    ],
    "method": "fft_laplace",
    "iterations": 1000
  }'
```

---

## Test All Features

### Run Comprehensive Test Suite
```bash
python test_enhanced_features.py
```

This will test:
- ✅ Health checks
- ✅ Error handling
- ✅ Basic solving
- ✅ Caching & deduplication
- ✅ Batch solving
- ✅ Async solving
- ✅ Metrics & monitoring
- ✅ Analytics & history
- ✅ Admin operations
- ✅ Methods information
- ✅ Rate limiting
- ✅ Benchmarking

### Run Specific Tests
```bash
# Test only health checks
python test_enhanced_features.py health

# Test caching
python test_enhanced_features.py cache

# Test batch solving
python test_enhanced_features.py batch
```

---

## Dashboard Integration

The enhanced server is **100% backward compatible** with the existing dashboard.

### Start Both Services

**Terminal 1** - Enhanced Server:
```bash
python server_enhanced.py
```

**Terminal 2** - Dashboard:
```bash
cd dashboard
python -m http.server 3000
```

Open browser: `http://localhost:3000`

The dashboard will automatically connect to the enhanced server at `http://localhost:8000`.

---

## Explore API Documentation

### Interactive API Docs (Swagger UI)
Open browser: `http://localhost:8000/docs`

Features:
- Try all endpoints interactively
- See request/response schemas
- Test error handling
- View parameter descriptions

### Alternative Docs (ReDoc)
Open browser: `http://localhost:8000/redoc`

Features:
- Cleaner, more readable format
- Better for reference
- Downloadable OpenAPI spec

---

## Key Features to Try

### 1. Automatic Caching
Run the same request twice and see speed improvement:

```python
import requests
import time

problem = {
    "problem_size": 5,
    "problem_matrix": [[0, 1, 2, 3, 4]] * 5,
    "method": "genetic_algorithm"
}

# First request (cache miss)
start = time.time()
requests.post("http://localhost:8000/solve", json=problem)
print(f"First request: {time.time() - start:.2f}s")

# Second request (cache hit)
start = time.time()
response = requests.post("http://localhost:8000/solve", json=problem)
print(f"Second request: {time.time() - start:.2f}s")
print(f"Cached: {response.json()['cached']}")
```

### 2. Batch Solving
Solve multiple problems in one request:

```python
import requests

response = requests.post(
    "http://localhost:8000/solve/batch",
    json={
        "parallel": True,
        "problems": [
            {
                "problem_size": 5,
                "problem_matrix": [[0, 1, 2, 3, 4]] * 5,
                "method": "fft_laplace"
            },
            {
                "problem_size": 5,
                "problem_matrix": [[0, 2, 4, 6, 8]] * 5,
                "method": "genetic_algorithm"
            }
        ]
    }
)

result = response.json()
print(f"Solved {result['successful']} problems in {result['total_runtime_seconds']:.2f}s")
```

### 3. Async Solving with Polling
Submit long-running tasks and poll for results:

```python
import requests
import time

# Submit async request
response = requests.post(
    "http://localhost:8000/solve-async",
    json={
        "problem_size": 5,
        "problem_matrix": [[0, 1, 2, 3, 4]] * 5,
        "method": "reverse_time"
    }
)

request_id = response.json()["request_id"]
print(f"Request ID: {request_id}")

# Poll for completion
while True:
    status = requests.get(f"http://localhost:8000/solve/{request_id}/status").json()
    print(f"Status: {status['status']}")

    if status["status"] == "completed":
        print(f"Result: {status['result']}")
        break

    time.sleep(1)
```

### 4. View Metrics
Monitor server performance:

```bash
# General metrics
curl http://localhost:8000/metrics | jq

# Detailed statistics
curl http://localhost:8000/stats | jq

# Prometheus format
curl http://localhost:8000/metrics/prometheus
```

### 5. Export Analytics
Get request history:

```bash
# View recent requests
curl http://localhost:8000/analytics/history?limit=10 | jq

# Export as CSV
curl http://localhost:8000/analytics/export/csv -o history.csv

# Export as JSON
curl http://localhost:8000/analytics/export/json | jq
```

---

## Monitoring

### Check Server Health
```bash
curl http://localhost:8000/health
```

### View Admin Status
```bash
curl http://localhost:8000/admin/status | jq
```

### Monitor in Real-Time

**Terminal 1** - Server:
```bash
python server_enhanced.py
```

**Terminal 2** - Watch Metrics:
```bash
watch -n 2 'curl -s http://localhost:8000/metrics | jq ".total_requests, .error_rate, .avg_response_time_ms"'
```

**Terminal 3** - Send Requests:
```bash
# Run test suite
python test_enhanced_features.py
```

---

## Common Operations

### Clear Cache and History
```bash
curl -X POST http://localhost:8000/admin/clear
```

### Reset Circuit Breaker
```bash
curl -X POST http://localhost:8000/admin/circuit-breaker/reset
```

### List Available Methods
```bash
curl http://localhost:8000/methods | jq
```

### Get Method Details
```bash
curl http://localhost:8000/methods/fft_laplace | jq
```

### Run Benchmark
```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "instance_name": "test_5x5",
    "methods": ["fft_laplace", "genetic_algorithm"],
    "num_runs": 5,
    "iterations_per_run": 1000
  }' | jq
```

---

## Error Handling Examples

### Invalid Matrix Size
```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "problem_size": 10,
    "problem_matrix": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    "method": "fft_laplace"
  }'
```

Response:
```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Matrix size 3x3 doesn't match problem_size 10",
  "suggestion": "Ensure matrix has 10x10 dimensions",
  "timestamp": "2025-11-19T10:30:00",
  "request_id": "..."
}
```

### Problem Size Too Large
```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "problem_size": 2000,
    "problem_matrix": [[0]],
    "method": "fft_laplace"
  }'
```

Response:
```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Problem size 2000 exceeds maximum of 1000",
  "suggestion": "Reduce problem size or contact support for larger instances",
  "timestamp": "2025-11-19T10:30:00",
  "request_id": "..."
}
```

---

## Performance Tips

### 1. Use Caching
Send identical requests to benefit from caching:
- First request: ~1-2 seconds
- Cached request: ~10-50ms (20-100x faster)

### 2. Use Batch Endpoint
For multiple problems:
- Parallel execution (default)
- Shared connection overhead
- Single request/response

### 3. Use Async for Long-Running Tasks
For large problems or many iterations:
- Non-blocking submission
- Poll when convenient
- Optional webhook notification

### 4. Monitor Performance
Track metrics to optimize:
```bash
curl http://localhost:8000/metrics | jq '.p95_ms, .p99_ms'
```

---

## Production Deployment

### Run with Multiple Workers
```bash
uvicorn server_enhanced:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

### Run in Background
```bash
nohup python server_enhanced.py > server.log 2>&1 &
```

### Stop Server
```bash
# Find process
ps aux | grep server_enhanced

# Kill gracefully (SIGTERM)
kill <PID>

# Force kill (SIGKILL) - use only if needed
kill -9 <PID>
```

---

## Troubleshooting

### Server Won't Start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process using port
kill $(lsof -t -i :8000)

# Start server again
python server_enhanced.py
```

### High Memory Usage
```bash
# Check memory usage
curl http://localhost:8000/admin/status | jq '.memory'

# Clear cache
curl -X POST http://localhost:8000/admin/clear
```

### Circuit Breaker Open
```bash
# Check status
curl http://localhost:8000/admin/status | jq '.circuit_breaker'

# Reset if needed
curl -X POST http://localhost:8000/admin/circuit-breaker/reset
```

### Rate Limited
```bash
# Check rate limit headers
curl -I http://localhost:8000/health | grep RateLimit

# Wait for reset or clear state
curl -X POST http://localhost:8000/admin/clear
```

---

## Next Steps

1. **Explore API Docs**: Visit `http://localhost:8000/docs`
2. **Run Tests**: Execute `python test_enhanced_features.py`
3. **Read Full Guide**: See `ENHANCED_SERVER_GUIDE.md`
4. **Integrate Dashboard**: Use with existing dashboard
5. **Monitor Metrics**: Set up Prometheus/Grafana

---

## Support & Resources

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **Admin Status**: http://localhost:8000/admin/status
- **Full Guide**: ENHANCED_SERVER_GUIDE.md

---

**Happy optimizing!** 🚀
