# QAPLIB Benchmark Suite

Complete implementation of the QAPLIB benchmark collection for the Quadratic Assignment Problem (QAP).

## Overview

This module provides access to all 138 standard QAPLIB benchmark instances, enabling rigorous benchmarking of optimization algorithms on well-established QAP problems.

## Features

- **Complete Registry**: All 138 QAPLIB instances with metadata
- **Embedded Data**: 20+ small instances with data embedded for instant access
- **Lazy Loading**: Larger instances downloaded on demand with caching
- **Benchmark Runner**: Comprehensive benchmarking framework
- **Easy Integration**: Works seamlessly with Librex optimization methods

## Quick Start

```python
from Librex.benchmarks.qaplib import load_qaplib_instance, run_qaplib_benchmark

# Load a specific instance
instance = load_qaplib_instance("nug12")
flow_matrix = instance["flow_matrix"]
distance_matrix = instance["distance_matrix"]

# Benchmark your optimizer
def my_optimizer(instance_data):
    # Your optimization logic here
    return {"solution": solution, "objective": objective}

results = run_qaplib_benchmark(
    method=my_optimizer,
    instances="small",  # or ["nug12", "chr12a", ...]
    method_name="MyOptimizer"
)
```

## Available Functions

### Core Functions
- `load_qaplib_instance(name)` - Load instance data
- `list_qaplib_instances(filter_by_size, filter_by_class)` - List instances
- `get_qaplib_metadata(name)` - Get instance metadata

### Registry Functions
- `get_instance_by_size(min_size, max_size)` - Filter by size
- `get_instance_by_class(problem_class)` - Filter by problem class
- `get_small_instances(max_size=20)` - Get small instances

### Benchmarking
- `run_qaplib_benchmark(method, instances, ...)` - Run benchmarks
- `QAPLIBBenchmark` - Advanced benchmark runner class

## Instance Categories

| Problem Class | Count | Description |
|--------------|-------|-------------|
| real-world | 33 | Real applications (hospital layout, keyboard design) |
| random-structured | 24 | Taillard's structured random instances |
| structured | 20 | Eschermann-Schweitzer instances |
| grid | 19 | Grid-based problems |
| random | 17 | Pure random instances |
| generated | 16 | Lipa's generated instances |
| Hadamard | 9 | Based on Hadamard matrices |

## Size Distribution

- **Small (n≤20)**: 54 instances - Good for testing
- **Medium (20<n≤50)**: 53 instances - Standard benchmarking
- **Large (n>50)**: 31 instances - Challenging problems
- **Very Large (n>100)**: 4 instances - Extreme challenges

## Examples

See `/home/user/AlaweinOS/Librex/examples/qaplib_benchmark_example.py` for comprehensive examples.

## References

- QAPLIB: https://qaplib.mgi.polymtl.ca/
- Burkard, R.E., Karisch, S.E., Rendl, F.: QAPLIB – A Quadratic Assignment Problem Library