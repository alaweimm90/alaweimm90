# MEZAN Benchmarking Guide

## Overview

This guide provides comprehensive documentation for benchmarking the MEZAN/ORCHEX system, including performance metrics, testing methodologies, and optimization strategies.

## Table of Contents

1. [Benchmark Categories](#benchmark-categories)
2. [Performance Metrics](#performance-metrics)
3. [Benchmarking Tools](#benchmarking-tools)
4. [Running Benchmarks](#running-benchmarks)
5. [Interpreting Results](#interpreting-results)
6. [Performance Tuning](#performance-tuning)
7. [Baseline Metrics](#baseline-metrics)

## Benchmark Categories

### 1. Throughput Benchmarks

Measure the system's capacity to process requests/workflows:

- **Workflow Throughput**: Workflows completed per second
- **Agent Throughput**: Agent tasks processed per second
- **API Throughput**: HTTP requests handled per second
- **Event Throughput**: Events processed per second
- **Blackboard Throughput**: Read/write operations per second

### 2. Latency Benchmarks

Measure response times and delays:

- **End-to-End Latency**: Total workflow completion time
- **Agent Latency**: Individual agent processing time
- **API Latency**: HTTP request response time
- **Event Latency**: Event publish to consumption time
- **Network Latency**: Inter-node communication delay

### 3. Scalability Benchmarks

Test system scaling capabilities:

- **Horizontal Scaling**: Performance with additional nodes
- **Vertical Scaling**: Performance with increased resources
- **Load Scaling**: Performance under varying load levels
- **Data Scaling**: Performance with increasing data volumes

### 4. Resource Usage Benchmarks

Profile system resource consumption:

- **CPU Usage**: Processor utilization patterns
- **Memory Usage**: RAM consumption and patterns
- **Disk I/O**: Read/write performance
- **Network I/O**: Bandwidth utilization

### 5. Distributed Performance

Test multi-node system performance:

- **Data Parallel Processing**: Distributed data processing
- **Task Parallel Execution**: Distributed task execution
- **Consensus Performance**: Distributed consensus protocols
- **Communication Overhead**: Inter-node communication costs

## Performance Metrics

### Key Performance Indicators (KPIs)

| Metric | Target | Critical | Description |
|--------|--------|----------|-------------|
| Workflow Throughput | >100/sec | <50/sec | Workflows completed per second |
| P95 Latency | <500ms | >2000ms | 95th percentile response time |
| P99 Latency | <1000ms | >5000ms | 99th percentile response time |
| Error Rate | <1% | >5% | Percentage of failed requests |
| CPU Usage | <70% | >90% | Average CPU utilization |
| Memory Usage | <80% | >95% | RAM utilization percentage |
| Network Throughput | >100MB/s | <10MB/s | Data transfer rate |

### Detailed Metrics

#### Throughput Metrics
```python
{
    "workflows_per_second": 150,
    "agents_per_second": 1000,
    "events_per_second": 5000,
    "api_requests_per_second": 2000,
    "database_ops_per_second": 10000
}
```

#### Latency Metrics
```python
{
    "mean_latency_ms": 125,
    "median_latency_ms": 100,
    "p50_latency_ms": 100,
    "p75_latency_ms": 150,
    "p90_latency_ms": 250,
    "p95_latency_ms": 500,
    "p99_latency_ms": 1000,
    "p99_9_latency_ms": 2000,
    "max_latency_ms": 5000
}
```

#### Resource Metrics
```python
{
    "cpu_usage_percent": 65,
    "memory_usage_mb": 2048,
    "disk_read_mb_s": 100,
    "disk_write_mb_s": 50,
    "network_rx_mb_s": 25,
    "network_tx_mb_s": 20,
    "open_connections": 500,
    "thread_count": 100
}
```

## Benchmarking Tools

### 1. Built-in Benchmarks

```bash
# Run throughput benchmarks
python tests/performance/benchmark_throughput.py

# Run latency benchmarks
python tests/performance/benchmark_latency.py

# Run scalability benchmarks
python tests/performance/benchmark_scalability.py

# Run resource profiling
python tests/performance/benchmark_resource_usage.py
```

### 2. Locust (Load Testing)

```bash
# Web UI mode
locust -f tests/performance/load_test.py --host=http://localhost:8080

# Headless mode
locust -f tests/performance/load_test.py \
    --host=http://localhost:8080 \
    --headless \
    --users 100 \
    --spawn-rate 10 \
    --run-time 10m
```

### 3. K6 (JavaScript Load Testing)

```javascript
// k6_script.js
import http from 'k6/http';
import { check } from 'k6';

export let options = {
    stages: [
        { duration: '2m', target: 100 },
        { duration: '5m', target: 100 },
        { duration: '2m', target: 0 },
    ],
};

export default function() {
    let response = http.get('http://localhost:8080/api/v1/workflow/list');
    check(response, {
        'status is 200': (r) => r.status === 200,
        'response time < 500ms': (r) => r.timings.duration < 500,
    });
}
```

Run with:
```bash
k6 run k6_script.js
```

### 4. Apache Bench (ab)

```bash
# Simple throughput test
ab -n 10000 -c 100 http://localhost:8080/api/v1/health

# POST request test
ab -n 1000 -c 10 -p workflow.json -T application/json \
    http://localhost:8080/api/v1/workflow/create
```

### 5. wrk (Modern HTTP benchmarking)

```bash
# Basic test
wrk -t12 -c400 -d30s http://localhost:8080/api/v1/workflow/list

# With Lua script for complex scenarios
wrk -t12 -c400 -d30s -s script.lua http://localhost:8080
```

## Running Benchmarks

### Quick Benchmark

```bash
# Run essential benchmarks (5 minutes)
make benchmark-quick

# Runs:
# - Basic throughput test
# - Latency measurements
# - Resource snapshot
```

### Full Benchmark Suite

```bash
# Run complete benchmark suite (30 minutes)
make benchmark-full

# Runs:
# - All throughput tests
# - All latency tests
# - Scalability tests
# - Resource profiling
# - Distributed tests
```

### Continuous Benchmarking

```bash
# Run benchmarks continuously
make benchmark-continuous

# Runs benchmarks every hour and tracks trends
```

### Custom Benchmark Configuration

```python
# benchmark_config.py
config = {
    "duration": 300,  # 5 minutes
    "warmup": 60,     # 1 minute warmup
    "cooldown": 30,   # 30 second cooldown
    "workload": {
        "type": "mixed",
        "distribution": {
            "read": 0.7,
            "write": 0.2,
            "compute": 0.1
        }
    },
    "load_pattern": {
        "type": "stepped",
        "steps": [10, 50, 100, 200, 500]
    }
}
```

## Interpreting Results

### Performance Report Structure

```
================================================================================
MEZAN PERFORMANCE BENCHMARK REPORT
================================================================================
Generated: 2025-01-15T10:30:00Z
Environment: Production
Duration: 600 seconds

SUMMARY
-------
✓ Workflow Throughput: 156 workflows/sec (Target: >100)
✓ P95 Latency: 450ms (Target: <500ms)
✗ P99 Latency: 1200ms (Target: <1000ms)
✓ Error Rate: 0.5% (Target: <1%)

DETAILED RESULTS
----------------

1. THROUGHPUT METRICS
   Workflows:     156/sec (avg), 180/sec (peak)
   API Requests:  2,340/sec (avg), 3,100/sec (peak)
   Events:        5,200/sec (avg), 6,800/sec (peak)

2. LATENCY DISTRIBUTION
   Min:     10ms
   P50:     95ms
   P75:     150ms
   P90:     280ms
   P95:     450ms
   P99:     1,200ms
   Max:     5,420ms

3. RESOURCE UTILIZATION
   CPU:     68% (avg), 85% (peak)
   Memory:  2.1GB (avg), 2.8GB (peak)
   Network: 25MB/s (avg), 40MB/s (peak)

4. ERROR ANALYSIS
   Total Requests:    1,404,000
   Successful:        1,397,000 (99.5%)
   Failed:           7,000 (0.5%)
   Error Types:
   - Timeout:        4,200 (60%)
   - Connection:     1,400 (20%)
   - Server Error:   1,400 (20%)

RECOMMENDATIONS
---------------
1. Optimize P99 latency - investigate slow queries
2. Add caching for frequently accessed data
3. Scale horizontally during peak hours
4. Implement connection pooling

COMPARISON WITH BASELINE
------------------------
Metric              Current    Baseline   Change
Throughput          156/sec    140/sec    +11.4%
P95 Latency         450ms      400ms      +12.5%
CPU Usage           68%        65%        +4.6%
Memory Usage        2.1GB      1.8GB      +16.7%
================================================================================
```

### Identifying Bottlenecks

1. **CPU Bottleneck**
   - High CPU usage (>80%)
   - Low I/O wait
   - Thread contention

2. **Memory Bottleneck**
   - High memory usage (>90%)
   - Frequent garbage collection
   - Swap usage

3. **I/O Bottleneck**
   - High I/O wait
   - Slow disk operations
   - Queue buildup

4. **Network Bottleneck**
   - High network latency
   - Packet loss
   - Bandwidth saturation

### Performance Trends

```python
# Generate trend analysis
python analyze_trends.py --input benchmark_results/ --output trends.html

# Tracks:
# - Performance over time
# - Regression detection
# - Capacity planning
# - Anomaly detection
```

## Performance Tuning

### Configuration Optimization

```yaml
# performance.yaml
engine:
  workers: 16
  max_connections: 1000
  connection_pool_size: 100

cache:
  enabled: true
  size: 10000
  ttl: 300

database:
  pool_size: 50
  max_overflow: 100
  timeout: 30

redis:
  max_connections: 200
  socket_keepalive: true
  socket_timeout: 5
```

### Code Optimizations

1. **Async I/O**
   ```python
   # Before
   results = []
   for item in items:
       results.append(process(item))

   # After
   results = await asyncio.gather(*[process(item) for item in items])
   ```

2. **Caching**
   ```python
   @lru_cache(maxsize=1000)
   def expensive_operation(key):
       # Computation
       return result
   ```

3. **Connection Pooling**
   ```python
   pool = ConnectionPool(min_size=10, max_size=100)
   async with pool.acquire() as conn:
       await conn.execute(query)
   ```

### System Tuning

```bash
# Increase file descriptors
ulimit -n 65536

# Optimize TCP settings
sysctl -w net.core.somaxconn=65535
sysctl -w net.ipv4.tcp_max_syn_backlog=65535

# Increase memory limits
sysctl -w vm.max_map_count=262144

# CPU governor for performance
cpupower frequency-set -g performance
```

## Baseline Metrics

### Development Environment

| Metric | Value | Hardware |
|--------|-------|----------|
| Workflow Throughput | 50/sec | 4 cores, 8GB RAM |
| P95 Latency | 200ms | Local Redis |
| CPU Usage | 40% | Intel i5 |
| Memory Usage | 500MB | |

### Staging Environment

| Metric | Value | Hardware |
|--------|-------|----------|
| Workflow Throughput | 100/sec | 8 cores, 16GB RAM |
| P95 Latency | 150ms | Redis cluster |
| CPU Usage | 50% | Intel Xeon |
| Memory Usage | 2GB | |

### Production Environment

| Metric | Value | Hardware |
|--------|-------|----------|
| Workflow Throughput | 200/sec | 16 cores, 32GB RAM |
| P95 Latency | 100ms | Redis cluster |
| CPU Usage | 60% | Intel Xeon |
| Memory Usage | 4GB | |

### Distributed Setup (4 nodes)

| Metric | Value | Configuration |
|--------|-------|---------------|
| Workflow Throughput | 600/sec | 4x 8-core nodes |
| P95 Latency | 150ms | 10Gbps network |
| CPU Usage | 65% | Per node |
| Memory Usage | 3GB | Per node |

## Benchmark Automation

### Nightly Benchmarks

```yaml
# .github/workflows/nightly-benchmark.yml
name: Nightly Benchmark

on:
  schedule:
    - cron: '0 2 * * *'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmarks
        run: make benchmark-full
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: results/
      - name: Compare with baseline
        run: python compare_benchmarks.py
      - name: Alert on regression
        if: failure()
        run: |
          echo "Performance regression detected"
          # Send alert
```

### Continuous Performance Monitoring

```python
# monitor.py
import time
import asyncio
from prometheus_client import Gauge, Counter, Histogram

# Define metrics
throughput_gauge = Gauge('atlas_throughput', 'Workflows per second')
latency_histogram = Histogram('atlas_latency', 'Response latency')
error_counter = Counter('atlas_errors', 'Error count')

async def monitor_performance():
    """Continuously monitor performance metrics."""
    while True:
        metrics = await collect_metrics()

        throughput_gauge.set(metrics['throughput'])
        latency_histogram.observe(metrics['latency'])

        if metrics['error']:
            error_counter.inc()

        await asyncio.sleep(10)
```

## Best Practices

1. **Warm-up Period**
   - Always include warm-up time
   - Discard warm-up measurements
   - Allow caches to populate

2. **Realistic Workloads**
   - Use production-like data
   - Simulate real usage patterns
   - Include error scenarios

3. **Consistent Environment**
   - Control background processes
   - Use dedicated hardware
   - Document configuration

4. **Statistical Significance**
   - Run multiple iterations
   - Calculate confidence intervals
   - Report standard deviation

5. **Incremental Testing**
   - Start with small loads
   - Gradually increase
   - Find breaking points

## Troubleshooting

### Common Issues

1. **Inconsistent Results**
   - Check for background processes
   - Verify network stability
   - Ensure sufficient warm-up

2. **Resource Exhaustion**
   - Monitor file descriptors
   - Check memory limits
   - Verify connection pools

3. **Timeouts**
   - Increase timeout values
   - Check network latency
   - Verify service health

### Debug Commands

```bash
# Monitor in real-time
htop  # CPU and memory
iotop  # Disk I/O
iftop  # Network traffic
netstat -an | grep ESTABLISHED | wc -l  # Connection count

# Profile application
py-spy top --pid $(pgrep -f ORCHEX)  # Python profiling
strace -p $(pgrep -f ORCHEX)  # System calls
tcpdump -i any port 8080  # Network packets
```

## Resources

- [Performance Testing Best Practices](https://www.performancetesting.guide)
- [Locust Documentation](https://docs.locust.io)
- [Systems Performance by Brendan Gregg](http://www.brendangregg.com/systems-performance-2nd-edition-book.html)
- [High Performance Browser Networking](https://hpbn.co/)

---

For questions or support, contact the MEZAN team.