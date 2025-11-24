# Performance Benchmarking Scripts

**Engineering Excellence Framework - Performance & Scalability**

This directory contains automated performance benchmarking scripts for measuring and monitoring system performance across the monorepo.

## 🏃‍♂️ Benchmarking Strategy

### Purpose
- **Performance Regression Detection**: Automatically identify performance degradations
- **Capacity Planning**: Measure system limits and scaling requirements
- **Optimization Validation**: Verify performance improvements meet targets
- **SLA Compliance**: Ensure systems meet performance requirements

### Scope
Benchmarks cover:
- **Application Performance**: API response times, throughput, latency
- **Infrastructure Performance**: CPU, memory, disk, network utilization
- **Database Performance**: Query performance, connection pooling, caching
- **CI/CD Performance**: Build times, test execution speed, deployment duration

## 📊 Benchmark Categories

### Micro-Benchmarks
**Location**: `micro/` - Component-level performance testing
- **Algorithm Complexity**: Big O analysis for core algorithms
- **Memory Usage**: Allocation and garbage collection patterns
- **I/O Operations**: File system, network, and database I/O

### Macro-Benchmarks
**Location**: `macro/` - System-level performance testing
- **API Throughput**: Requests per second, concurrent users
- **Database Load**: Query performance under load
- **Integration Tests**: End-to-end workflow performance

### Chaos-Aware Benchmarks
**Location**: `chaos/` - Performance under failure conditions
- **Degraded Mode Performance**: System behavior during failures
- **Recovery Performance**: Time to restore normal operation
- **Chaos Impact Measurement**: Performance metrics during chaos experiments

## 🛠️ Benchmark Tools

### Python Benchmarks (`python/`)
```bash
# Run core algorithm benchmarks
python benchmark-scripts/python/core_algorithms.py --output results/core_perf.json

# Run memory usage analysis
python benchmark-scripts/python/memory_analysis.py --track-gc --output results/memory.json

# Run I/O performance tests
python benchmark-scripts/python/io_performance.py --workload mixed --output results/io.json
```

### API Benchmarks (`api/`)
```bash
# Load testing with Locust
locust -f benchmark-scripts/api/locustfile.py --host https://api.example.com --users 100 --spawn-rate 10

# Artillery.io artillery testing
npm run artillery run benchmark-scripts/api/artillery-config.yml
```

### Database Benchmarks (`database/`)
```bash
# PostgreSQL benchmarking with pgbench
pgbench -h localhost -U user -d database -c 10 -j 2 -T 60

# Redis benchmark
redis-benchmark -t set,get -n 100000 -q
```

## 📈 Metrics Collection

### Key Performance Indicators (KPIs)

#### Application KPIs
```json
{
  "api_response_time": {
    "p50": "150ms",
    "p95": "450ms",
    "p99": "800ms"
  },
  "throughput": {
    "rps": 1200,
    "success_rate": "99.9%"
  },
  "error_rate": {
    "4xx": "0.1%",
    "5xx": "0.01%"
  }
}
```

#### Infrastructure KPIs
```json
{
  "cpu": {
    "user": "65%",
    "system": "15%",
    "idle": "20%"
  },
  "memory": {
    "used": "4.2GB",
    "available": "7.8GB",
    "usage_percent": "35%"
  },
  "disk": {
    "read_iops": 1500,
    "write_iops": 800,
    "latency": "2ms"
  }
}
```

### Performance Baselines

#### Establishment Process
1. **Initial Measurement**: Establish baseline in known stable state
2. **Statistical Analysis**: Calculate mean, standard deviation, confidence intervals
3. **Regression Thresholds**: Define acceptable performance ranges
4. **Monitoring Setup**: Configure alerts for threshold violations

#### Regression Detection
```python
def detect_regression(current_metrics, baseline_metrics, threshold=0.05):
    """Detect performance regression using statistical comparison."""
    # Calculate percentage change
    change_pct = (current_metrics - baseline_metrics) / baseline_metrics

    # Apply statistical significance test
    if abs(change_pct) > threshold:
        return {
            'regression_detected': True,
            'change_percent': change_pct,
            'severity': 'high' if abs(change_pct) > 0.15 else 'medium'
        }

    return {'regression_detected': False}
```

## 🚀 Execution Framework

### Scheduled Benchmarks
```bash
# Daily performance regression check
0 2 * * * /path/to/benchmark-scripts/run-daily.sh

# Weekly capacity testing
0 3 * * 1 /path/to/benchmark-scripts/run-capacity-tests.sh

# Monthly chaos-aware benchmarking
0 4 1 * * /path/to/benchmark-scripts/run-chaos-benchmarks.sh
```

### CI/CD Integration
```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks
on: [pull_request, push]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Benchmarks
        run: ./benchmark-scripts/ci-benchmark.sh
      - name: Compare Results
        run: ./benchmark-scripts/compare-results.py --baseline master --current ${{ github.sha }}
      - name: Comment Results
        uses: dorny/test-reporter@v1
        with:
          name: Performance Benchmarks
          path: 'results/benchmark-*.json'
```

### Benchmark Result Analysis
```python
# benchmark-analysis.py
import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_results(results_file):
    """Analyze benchmark results and generate insights."""
    with open(results_file) as f:
        data = json.load(f)

    # Trend analysis
    trends = calculate_trends(data)

    # Anomaly detection
    anomalies = detect_anomalies(data)

    # Performance insights
    insights = generate_insights(data, trends, anomalies)

    return {
        'trends': trends,
        'anomalies': anomalies,
        'insights': insights,
        'recommendations': generate_recommendations(insights)
    }
```

## 📋 Benchmark Templates

### API Load Testing Template
```python
# api_load_test.py
import locust
from locust import HttpUser, task, between

class ApiUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def get_endpoint(self):
        self.client.get("/api/endpoint")

    @task(3)
    def post_endpoint(self):
        self.client.post("/api/endpoint", json={"data": "test"})

    def on_start(self):
        self.client.headers = {"Authorization": "Bearer <token>"}
```

### Memory Benchmark Template
```python
# memory_benchmark.py
import tracemalloc
import gc
import psutil

def benchmark_memory_usage(func, *args, **kwargs):
    """Benchmark memory usage of a function."""
    tracemalloc.start()
    process = psutil.Process()

    # Record initial memory
    initial_memory = process.memory_info().rss

    # Execute function
    result = func(*args, **kwargs)
    gc.collect()

    # Record final memory
    final_memory = process.memory_info().rss

    # Calculate statistics
    memory_used = final_memory - initial_memory
    peak_memory = tracemalloc.get_traced_memory()[1]

    tracemalloc.stop()

    return {
        'memory_used': memory_used,
        'peak_memory': peak_memory,
        'result': result
    }
```

## 📊 Reporting & Visualization

### Performance Dashboard
```python
# dashboard_generator.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_dashboard(metrics_data, output_file):
    """Generate interactive performance dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Response Time', 'Throughput', 'Memory Usage', 'Error Rate')
    )

    # Response time chart
    fig.add_trace(
        go.Scatter(x=metrics_data['timestamps'], y=metrics_data['response_times'],
                  mode='lines+markers', name='Response Time'),
        row=1, col=1
    )

    # Throughput chart
    fig.add_trace(
        go.Bar(x=metrics_data['timestamps'], y=metrics_data['throughput'],
               name='Throughput'),
        row=1, col=2
    )

    fig.write_html(output_file)
```

### Performance Comparison Reports
```python
# comparison_report.py
def generate_comparison_report(baseline_results, current_results):
    """Generate detailed performance comparison."""
    report = {
        'summary': {
            'improvements': 0,
            'regressions': 0,
            'unchanged': 0
        },
        'details': []
    }

    for benchmark in baseline_results:
        baseline = baseline_results[benchmark]
        current = current_results.get(benchmark)

        if not current:
            continue

        # Calculate change percentage
        change = ((current['median'] - baseline['median']) / baseline['median']) * 100

        category = 'improvement' if change > 5 else 'regression' if change < -5 else 'unchanged'
        report['summary'][category] += 1

        report['details'].append({
            'benchmark': benchmark,
            'baseline': baseline['median'],
            'current': current['median'],
            'change_percent': change,
            'category': category
        })

    return report
```

## 🔧 Integration Points

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: performance-baseline-check
      name: Performance Regression Check
      entry: benchmark-scripts/check-performance-regression.sh
```

### CI/CD Gates
```yaml
# Enforce performance standards
- name: Performance Gate
  run: |
    results=$(./benchmark-scripts/run-performance-tests.sh)
    if [ "$(echo "$results" | jq '.failures')" -gt 0 ]; then
      echo "Performance tests failed"
      exit 1
    fi
```

### Monitoring Integration
```python
# prometheus_metrics.py
from prometheus_client import Gauge, Histogram

# Define metrics
response_time = Histogram('api_response_time_seconds', 'API response time')
throughput = Gauge('api_requests_per_second', 'API throughput')
error_rate = Gauge('api_error_rate_percent', 'API error rate percentage')

def update_metrics(results):
    """Update Prometheus metrics from benchmark results."""
    response_time.observe(results['response_time'])
    throughput.set(results['throughput'])
    error_rate.set(results['error_rate'])
```

## 📈 Results Storage & Analysis

### Database Schema
```sql
-- Performance benchmark results storage
CREATE TABLE benchmark_results (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    benchmark_name VARCHAR(255) NOT NULL,
    commit_sha VARCHAR(40),
    branch VARCHAR(255),
    results JSONB,
    environment VARCHAR(100)
);

-- Performance analysis insights
CREATE TABLE performance_insights (
    id SERIAL PRIMARY KEY,
    benchmark_result_id INTEGER REFERENCES benchmark_results(id),
    insight_type VARCHAR(100),
    description TEXT,
    severity VARCHAR(20),
    recommendations JSONB
);
```

### Trend Analysis
```python
# trend_analyzer.py
def analyze_trends(results_history, window_days=30):
    """Analyze performance trends over time."""
    recent_results = filter_by_date(results_history, days=window_days)

    trends = {}
    for benchmark in recent_results[0]['results']:
        values = [r['results'][benchmark] for r in recent_results if benchmark in r['results']]
        if len(values) < 2:
            continue

        trend = calculate_linear_trend(values)
        trends[benchmark] = {
            'slope': trend['slope'],
            'direction': 'improving' if trend['slope'] < 0 else 'degrading',
            'confidence': trend['r_squared'],
            'change_rate': trend['slope_per_day']
        }

    return trends
```

## 🚨 Alerting & Notifications

### Performance Alert Rules
```yaml
# alerting_rules.yml
rules:
  - name: High Response Time
    condition: response_time_p95 > 500ms for 5m
    severity: warning
    message: "P95 response time exceeded 500ms"

  - name: Low Throughput
    condition: requests_per_second < 100 for 10m
    severity: error
    message: "Request throughput below acceptable threshold"

  - name: High Error Rate
    condition: error_rate > 1% for 5m
    severity: critical
    message: "Error rate above 1% threshold"
```

### Automated Issue Creation
```python
# issue_creator.py
def create_performance_issue(violations):
    """Create GitHub issue for performance violations."""
    title = f"🎯 Performance Violation: {violations[0]['violation']}"
    body = generate_issue_body(violations)

    # Use GitHub API to create issue
    # ...
```

## 🔄 Maintenance & Evolution

### Benchmark Updates
- **Monthly**: Review and update benchmarks for new features
- **Quarterly**: Comprehensive benchmark suite review
- **Breaking Changes**: Update baselines when intentional performance changes occur

### Environment Consistency
- **Standardized Hardware**: Run benchmarks on consistent infrastructure
- **Environment Documentation**: Detail test environment specifications
- **Noise Reduction**: Account for environmental variability in results

### Continuous Improvement
- **False Positive Reduction**: Refine regression detection algorithms
- **Insight Quality**: Improve automated performance insight generation
- **Developer Experience**: Make benchmark results more actionable

---

**Engineering Excellence Framework Compliance**: These benchmarking scripts ensure continuous performance monitoring and optimization throughout the development lifecycle.
