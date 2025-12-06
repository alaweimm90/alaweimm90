# How to Benchmark Optimization Methods

**Goal:** Run comprehensive benchmarks comparing optimization methods across multiple instances

**Time Required:** 30 minutes (setup), 1-2 hours (benchmark execution)

**Prerequisites:** Librex.QAP installed, understanding of optimization methods

---

## Overview

Benchmarking is critical for validating that new methods actually improve performance. This guide covers systematic benchmarking methodology.

## Step 1: Basic Benchmarking

```python
from Librex.QAP.benchmarking_suite import BenchmarkSuite
from Librex.QAP.utils import load_qap_instance
import time

suite = BenchmarkSuite()

# Run simple benchmark
problem = load_qap_instance("data/qaplib/nug20.dat")

methods = ["fft_laplace", "reverse_time", "simulated_annealing"]

for method in methods:
    start = time.time()
    result = suite.solve(problem, method=method, iterations=100)
    elapsed = time.time() - start

    print(f"{method:20s}: {result.objective_value:10.2f} ({elapsed:6.3f}s)")
```

## Step 2: Multi-Instance Benchmarking

```python
instances = [
    "data/qaplib/nug12.dat",
    "data/qaplib/nug20.dat",
    "data/qaplib/tai30a.dat",
    "data/qaplib/tai40a.dat"
]

methods = ["fft_laplace", "reverse_time", "genetic_algorithm"]

results = []

for instance_path in instances:
    problem = load_qap_instance(instance_path)
    instance_name = instance_path.split('/')[-1].replace('.dat', '')

    print(f"\nBenchmarking {instance_name} (n={problem.size})")

    for method in methods:
        times = []
        qualities = []

        # Run 5 times for stability
        for run in range(5):
            start = time.time()
            result = suite.solve(problem, method=method, iterations=500)
            elapsed = time.time() - start

            times.append(elapsed)
            qualities.append(result.objective_value)

        avg_time = np.mean(times)
        avg_quality = np.mean(qualities)
        std_quality = np.std(qualities)

        results.append({
            'instance': instance_name,
            'method': method,
            'avg_time': avg_time,
            'avg_quality': avg_quality,
            'std_quality': std_quality
        })

        print(f"  {method:20s}: {avg_quality:10.2f} ± {std_quality:6.2f} ({avg_time:6.3f}s)")

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('benchmark_results.csv', index=False)
```

## Step 3: Comparison Metrics

```python
def compute_gap(result, optimal):
    """Compute gap to optimal."""
    return (result.objective_value - optimal) / optimal * 100


def compute_speedup(method_time, baseline_time):
    """Compute speedup vs baseline."""
    return baseline_time / method_time


# Example with known optima
optima = {
    "nug12.dat": 578,
    "nug20.dat": 2570,
    "tai30a.dat": 1818146,
    "tai40a.dat": 3139370
}

# Compute gaps
for idx, row in df.iterrows():
    optimal = optima.get(row['instance'] + '.dat', None)
    if optimal:
        row['gap_percent'] = compute_gap(row, optimal)
        df.loc[idx] = row
```

## Step 4: Statistical Analysis

```python
import numpy as np
import scipy.stats as stats

# T-test to compare two methods
method1_results = df[df['method'] == 'fft_laplace']['avg_quality']
method2_results = df[df['method'] == 'simulated_annealing']['avg_quality']

t_stat, p_value = stats.ttest_ind(method1_results, method2_results)

if p_value < 0.05:
    print(f"Method 1 significantly better (p={p_value:.4f})")
else:
    print(f"No significant difference (p={p_value:.4f})")

# Ranking
ranking = df.groupby('method')['avg_quality'].mean().sort_values()
print("\nMethod Ranking (lower is better):")
for i, (method, score) in enumerate(ranking.items(), 1):
    print(f"  {i}. {method}: {score:.2f}")
```

## Step 5: Visualization

```python
import matplotlib.pyplot as plt

# Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Objective values by method
ax = axes[0]
for method in methods:
    data = df[df['method'] == method]
    ax.plot(data['instance'], data['avg_quality'], marker='o', label=method)
ax.set_xlabel('Instance')
ax.set_ylabel('Objective Value')
ax.set_title('Quality Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Runtime comparison
ax = axes[1]
for method in methods:
    data = df[df['method'] == method]
    ax.bar(np.arange(len(data)) + methods.index(method) * 0.25,
           data['avg_time'], width=0.25, label=method)
ax.set_xlabel('Instance')
ax.set_ylabel('Time (seconds)')
ax.set_title('Runtime Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Best Practices

✅ **DO:**
- Run each method multiple times (5-10) for statistical stability
- Use same random seed for fair comparison
- Test on variety of instance sizes
- Report both mean and standard deviation
- Include confidence intervals
- Document all parameters

❌ **DON'T:**
- Run methods only once (random variance is high)
- Use different hardware for different methods
- Change parameters between runs
- Cherry-pick favorable instances
- Report only best-case results

## Benchmark Report Template

```markdown
# Benchmark Report: [Method Name]

## Methodology
- Instances: [list]
- Runs per instance: 5
- Iterations per run: 500
- Random seed: 42

## Results

| Instance | Size | Method 1 | Method 2 | Method 3 |
|----------|------|----------|----------|----------|
| nug12    | 12   | 578      | 580      | 585      |
| nug20    | 20   | 2570     | 2575     | 2590     |
| tai30a   | 30   | 1.82E6   | 1.84E6   | 1.88E6   |

## Analysis
[Summary of findings]

## Conclusions
[Key insights]
```

---

**Happy benchmarking!** 🚀

Last Updated: November 2024
