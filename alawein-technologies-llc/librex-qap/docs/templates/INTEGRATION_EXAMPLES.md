# Integration Examples for Librex.QAP-new

Real-world examples for integrating Librex.QAP with other systems and platforms.

---

## 1. Jupyter Notebook Integration

Save as `notebooks/Librex.QAP_demo.ipynb`:

```python
# Installation
!pip install Librex.QAP-new jupyter-lab pandas matplotlib plotly

# Import libraries
import numpy as np
import pandas as pd
from Librex.QAP import OptimizationPipeline, load_qap_instance
import matplotlib.pyplot as plt
import plotly.express as px

# ============================================================================
# EXAMPLE 1: Basic Problem Solving
# ============================================================================

# Create or load a problem
problem = load_qap_instance("data/qaplib/nug20.dat")

# Create optimization pipeline
pipeline = OptimizationPipeline(problem_size=20)

# Solve with different methods
methods = ["fft_laplace", "reverse_time", "genetic_algorithm"]
results = {}

for method in methods:
    result = pipeline.solve(
        problem,
        method=method,
        iterations=500
    )
    results[method] = result
    print(f"{method}: quality={result.objective_value:.2f}, "
          f"time={result.runtime:.2f}s")

# ============================================================================
# EXAMPLE 2: Visualization
# ============================================================================

# Compare results
comparison_df = pd.DataFrame({
    'Method': list(results.keys()),
    'Quality': [r.objective_value for r in results.values()],
    'Time': [r.runtime for r in results.values()]
})

fig = px.scatter(
    comparison_df,
    x='Time',
    y='Quality',
    size='Quality',
    hover_name='Method',
    title='Method Performance Comparison'
)
fig.show()

# ============================================================================
# EXAMPLE 3: Parameter Tuning
# ============================================================================

from scipy.optimize import differential_evolution

def objective(params):
    """Objective function for optimization."""
    lr, momentum = params
    result = pipeline.solve(
        problem,
        method="fft_laplace",
        learning_rate=lr,
        momentum=momentum,
        iterations=100
    )
    return result.objective_value

# Bayesian optimization
bounds = [(0.001, 1.0), (0.0, 1.0)]  # lr, momentum
result = differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=20
)

print(f"Optimal parameters: learning_rate={result.x[0]:.4f}, "
      f"momentum={result.x[1]:.4f}")

# ============================================================================
# EXAMPLE 4: Benchmark Analysis
# ============================================================================

from Librex.QAP.benchmarking import BenchmarkSuite

suite = BenchmarkSuite()
benchmark_data = []

instances = ["data/qaplib/nug12.dat", "data/qaplib/nug20.dat"]

for instance_path in instances:
    problem = load_qap_instance(instance_path)
    instance_name = instance_path.split('/')[-1].replace('.dat', '')

    for method in methods:
        result = suite.solve(problem, method=method)
        benchmark_data.append({
            'Instance': instance_name,
            'Method': method,
            'Quality': result.objective_value,
            'Time': result.runtime
        })

bench_df = pd.DataFrame(benchmark_data)

# Visualization
fig = px.box(
    bench_df,
    x='Method',
    y='Quality',
    color='Method',
    facet_col='Instance',
    title='Benchmark Results Across Instances'
)
fig.show()
```

---

## 2. Database Integration (PostgreSQL)

Save as `integrations/db_integration.py`:

```python
"""Integration with PostgreSQL for storing optimization results."""

import psycopg2
from psycopg2.extras import execute_values
import json
from datetime import datetime
from typing import Dict, List

class OptimizationDatabase:
    """PostgreSQL backend for storing optimization results."""

    def __init__(self, connection_string: str):
        """Initialize database connection."""
        self.conn = psycopg2.connect(connection_string)
        self.cur = self.conn.cursor()
        self._initialize_schema()

    def _initialize_schema(self):
        """Create tables if they don't exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS optimizations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            method VARCHAR(50) NOT NULL,
            problem_size INT NOT NULL,
            objective_value FLOAT NOT NULL,
            runtime_seconds FLOAT NOT NULL,
            iterations INT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS benchmarks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            instance_name VARCHAR(50) NOT NULL,
            method VARCHAR(50) NOT NULL,
            gap_percent FLOAT,
            runtime_seconds FLOAT,
            robustness FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_method ON optimizations(method);
        CREATE INDEX IF NOT EXISTS idx_created ON optimizations(created_at);
        """

        for statement in schema.split(';'):
            if statement.strip():
                self.cur.execute(statement)
        self.conn.commit()

    def store_optimization(self, result: Dict) -> str:
        """Store optimization result."""
        query = """
        INSERT INTO optimizations
        (method, problem_size, objective_value, runtime_seconds, iterations)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """

        self.cur.execute(query, (
            result['method'],
            result['problem_size'],
            result['objective_value'],
            result['runtime_seconds'],
            result['iterations']
        ))

        opt_id = self.cur.fetchone()[0]
        self.conn.commit()
        return str(opt_id)

    def store_benchmarks(self, benchmarks: List[Dict]):
        """Store benchmark results."""
        query = """
        INSERT INTO benchmarks
        (instance_name, method, gap_percent, runtime_seconds, robustness)
        VALUES %s
        """

        values = [
            (
                b['instance_name'],
                b['method'],
                b.get('gap_percent'),
                b['runtime_seconds'],
                b.get('robustness')
            )
            for b in benchmarks
        ]

        execute_values(self.cur, query, values)
        self.conn.commit()

    def get_method_statistics(self, method: str) -> Dict:
        """Get statistics for a method."""
        query = """
        SELECT
            COUNT(*) as total,
            AVG(objective_value) as avg_quality,
            AVG(runtime_seconds) as avg_runtime,
            MIN(objective_value) as best_quality,
            MAX(objective_value) as worst_quality
        FROM optimizations
        WHERE method = %s
        """

        self.cur.execute(query, (method,))
        result = self.cur.fetchone()

        return {
            'total': result[0],
            'avg_quality': result[1],
            'avg_runtime': result[2],
            'best_quality': result[3],
            'worst_quality': result[4]
        }

    def get_trending_methods(self, days: int = 7) -> List[Dict]:
        """Get trending methods over period."""
        query = """
        SELECT
            method,
            COUNT(*) as usage_count,
            AVG(objective_value) as avg_quality
        FROM optimizations
        WHERE created_at >= NOW() - INTERVAL '%s days'
        GROUP BY method
        ORDER BY usage_count DESC
        """

        self.cur.execute(query, (days,))
        columns = ['method', 'usage_count', 'avg_quality']
        return [dict(zip(columns, row)) for row in self.cur.fetchall()]

    def export_results(self, output_file: str, **filters):
        """Export results to CSV."""
        import csv

        query = "SELECT * FROM optimizations WHERE 1=1"
        params = []

        if method := filters.get('method'):
            query += " AND method = %s"
            params.append(method)

        self.cur.execute(query, params)

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([desc[0] for desc in self.cur.description])
            writer.writerows(self.cur.fetchall())

    def close(self):
        """Close database connection."""
        self.cur.close()
        self.conn.close()

# Usage
if __name__ == "__main__":
    db = OptimizationDatabase("postgresql://user:pass@localhost/Librex.QAP")

    # Store result
    result = {
        'method': 'fft_laplace',
        'problem_size': 20,
        'objective_value': 2570.0,
        'runtime_seconds': 0.85,
        'iterations': 500
    }
    opt_id = db.store_optimization(result)
    print(f"Stored optimization: {opt_id}")

    # Get statistics
    stats = db.get_method_statistics('fft_laplace')
    print(f"Method statistics: {stats}")

    # Get trending
    trending = db.get_trending_methods(days=7)
    print(f"Trending methods: {trending}")

    db.close()
```

---

## 3. REST API Integration

Save as `integrations/api_client.py`:

```python
"""Client for consuming Librex.QAP REST API."""

import requests
import json
from typing import Dict, List, Optional
import time

class QAPlbriaClient:
    """HTTP client for Librex.QAP API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API client."""
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if service is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def solve(self, problem_matrix: List[List[float]],
              method: str = "fft_laplace", iterations: int = 500) -> Dict:
        """Solve optimization problem."""
        payload = {
            "problem_size": len(problem_matrix),
            "problem_matrix": problem_matrix,
            "method": method,
            "iterations": iterations
        }

        response = self.session.post(
            f"{self.base_url}/solve",
            json=payload,
            timeout=300
        )

        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")

        return response.json()

    def solve_async(self, problem_matrix: List[List[float]],
                    method: str = "fft_laplace") -> str:
        """Start async optimization, returns request ID."""
        payload = {
            "problem_size": len(problem_matrix),
            "problem_matrix": problem_matrix,
            "method": method,
            "iterations": 500
        }

        response = self.session.post(
            f"{self.base_url}/solve-async",
            json=payload
        )

        return response.json()['request_id']

    def get_status(self, request_id: str) -> Dict:
        """Check async request status."""
        response = self.session.get(
            f"{self.base_url}/solve/{request_id}/status"
        )
        return response.json()

    def wait_for_result(self, request_id: str, timeout: int = 300) -> Dict:
        """Wait for async result with polling."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status(request_id)
            if status['status'] == 'completed':
                return status['result']
            time.sleep(1)
        raise TimeoutError(f"Request {request_id} timed out")

    def list_methods(self) -> List[Dict]:
        """Get available methods."""
        response = self.session.get(f"{self.base_url}/methods")
        return response.json()

    def get_method(self, method_name: str) -> Dict:
        """Get method details."""
        response = self.session.get(
            f"{self.base_url}/methods/{method_name}"
        )
        return response.json()

    def run_benchmark(self, instance: str, methods: List[str]) -> str:
        """Start benchmark run, returns benchmark ID."""
        payload = {
            "instance_name": instance,
            "methods": methods,
            "num_runs": 5
        }

        response = self.session.post(
            f"{self.base_url}/benchmark",
            json=payload
        )

        return response.json()['benchmark_id']

    def get_benchmark_results(self, benchmark_id: str) -> Dict:
        """Get benchmark results."""
        response = self.session.get(
            f"{self.base_url}/benchmark/{benchmark_id}/status"
        )
        return response.json()

    def get_metrics(self) -> Dict:
        """Get service metrics."""
        response = self.session.get(f"{self.base_url}/metrics")
        return response.json()

# Usage
if __name__ == "__main__":
    client = QAPlbriaClient()

    # Check health
    if not client.health_check():
        print("API is not available")
        exit(1)

    # Create problem
    n = 20
    problem_matrix = [[i+j for j in range(n)] for i in range(n)]

    # Solve synchronously
    print("Solving synchronously...")
    result = client.solve(problem_matrix, method="fft_laplace")
    print(f"Quality: {result['objective_value']:.2f}")
    print(f"Time: {result['runtime_seconds']:.2f}s")

    # Solve asynchronously
    print("\nSolving asynchronously...")
    request_id = client.solve_async(problem_matrix)
    print(f"Request ID: {request_id}")

    result = client.wait_for_result(request_id)
    print(f"Quality: {result['objective_value']:.2f}")

    # Get methods
    print("\nAvailable methods:")
    for method in client.list_methods():
        print(f"- {method['name']}: {method['description']}")

    # Get metrics
    print("\nService metrics:")
    metrics = client.get_metrics()
    print(json.dumps(metrics, indent=2))
```

---

## 4. Message Queue Integration (Celery)

Save as `integrations/celery_integration.py`:

```python
"""Celery integration for distributed optimization tasks."""

from celery import Celery
from Librex.QAP import OptimizationPipeline, load_qap_instance
import logging

# Initialize Celery app
app = Celery(
    'Librex.QAP_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

logger = logging.getLogger(__name__)

@app.task(bind=True, max_retries=3)
def solve_optimization(self, problem_size, method, iterations=500):
    """Asynchronous optimization task."""
    try:
        pipeline = OptimizationPipeline(problem_size=problem_size)

        # Simulate problem generation (use real data in production)
        import numpy as np
        problem = type('Problem', (), {
            'size': problem_size,
            'cost_matrix': np.random.rand(problem_size, problem_size)
        })()

        result = pipeline.solve(
            problem,
            method=method,
            iterations=iterations
        )

        return {
            'status': 'success',
            'objective_value': float(result.objective_value),
            'runtime': float(result.runtime),
            'method': method
        }

    except Exception as exc:
        logger.error(f"Task failed: {exc}")
        raise self.retry(exc=exc, countdown=5)

@app.task
def run_benchmark_task(instance_name, methods, num_runs=5):
    """Benchmark task."""
    results = []

    for method in methods:
        for run in range(num_runs):
            result = solve_optimization.apply_async(
                args=(20, method, 500),
                queue='high_priority'
            ).get(timeout=300)

            results.append(result)

    return {
        'instance': instance_name,
        'methods': methods,
        'results': results
    }

@app.task
def batch_optimize(problems_data):
    """Process batch of problems."""
    results = []

    for problem_data in problems_data:
        result = solve_optimization.apply_async(
            args=(
                problem_data['size'],
                problem_data['method'],
                problem_data.get('iterations', 500)
            )
        ).get()

        results.append(result)

    return results

# Usage
if __name__ == "__main__":
    # Submit task
    task = solve_optimization.apply_async(
        args=(20, 'fft_laplace', 500),
        queue='default'
    )

    print(f"Task ID: {task.id}")

    # Get result
    result = task.get(timeout=300)
    print(f"Result: {result}")

    # Submit benchmark
    benchmark_task = run_benchmark_task.apply_async(
        args=('nug20', ['fft_laplace', 'reverse_time']),
        queue='high_priority'
    )

    print(f"Benchmark task: {benchmark_task.id}")
```

---

## 5. Cloud Integration (AWS Lambda)

Save as `integrations/aws_lambda.py`:

```python
"""AWS Lambda integration for serverless optimization."""

import json
import boto3
from Librex.QAP import OptimizationPipeline

# For AWS Lambda, install as Lambda layer
# dependencies should be in /opt/python/lib/python3.9/site-packages/

def lambda_handler(event, context):
    """Lambda handler for optimization requests."""

    try:
        # Parse request
        body = json.loads(event.get('body', '{}'))
        problem_matrix = body.get('problem_matrix', [])
        method = body.get('method', 'fft_laplace')
        iterations = body.get('iterations', 500)

        if not problem_matrix:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'problem_matrix required'})
            }

        # Solve
        pipeline = OptimizationPipeline(problem_size=len(problem_matrix))

        # Create problem object
        import numpy as np
        problem = type('Problem', (), {
            'size': len(problem_matrix),
            'cost_matrix': np.array(problem_matrix)
        })()

        result = pipeline.solve(
            problem,
            method=method,
            iterations=iterations
        )

        # Return result
        return {
            'statusCode': 200,
            'body': json.dumps({
                'objective_value': float(result.objective_value),
                'runtime': float(result.runtime),
                'method': method,
                'iterations': iterations
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def benchmark_handler(event, context):
    """Lambda for running benchmarks."""

    instance = event.get('instance', 'nug20')
    methods = event.get('methods', ['fft_laplace', 'reverse_time'])

    results = {}

    for method in methods:
        # In production, load actual instance from S3
        problem_size = 20

        pipeline = OptimizationPipeline(problem_size=problem_size)

        # Solve
        result = pipeline.solve(
            None,  # Use actual problem
            method=method,
            iterations=500
        )

        results[method] = {
            'quality': float(result.objective_value),
            'runtime': float(result.runtime)
        }

    # Save results to S3
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket='Librex.QAP-results',
        Key=f'benchmarks/{instance}.json',
        Body=json.dumps(results)
    )

    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }
```

---

## 6. Monitoring Integration (Prometheus)

Save as `integrations/monitoring.py`:

```python
"""Prometheus metrics integration."""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Define metrics
optimization_requests = Counter(
    'Librex.QAP_optimizations_total',
    'Total optimization requests',
    ['method', 'status']
)

optimization_duration = Histogram(
    'Librex.QAP_optimization_duration_seconds',
    'Optimization duration',
    ['method'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

solution_quality = Gauge(
    'Librex.QAP_solution_quality',
    'Best solution quality found',
    ['method']
)

queue_length = Gauge(
    'Librex.QAP_queue_length',
    'Number of pending tasks'
)

def record_optimization(method, duration, quality, status='success'):
    """Record optimization metrics."""
    optimization_requests.labels(method=method, status=status).inc()
    optimization_duration.labels(method=method).observe(duration)
    solution_quality.labels(method=method).set(quality)

def get_metrics():
    """Get Prometheus metrics."""
    return generate_latest()
```

---

**Last Updated:** November 2024
