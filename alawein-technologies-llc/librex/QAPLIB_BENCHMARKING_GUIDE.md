# QAPLIB Benchmarking Guide

Comprehensive guide for benchmarking QAP optimization methods against QAPLIB instances.

## QAPLIB Dataset

QAPLIB is the standard benchmark library for Quadratic Assignment Problems containing 138+ instances with known optimal solutions.

## Available Instances

Librex includes QAPLIB instances in `data/qaplib/`:
- **Small**: n < 20 (e.g., chr12a, nug12)
- **Medium**: 20 ≤ n < 50 (e.g., esc32a, tai40a)
- **Large**: n ≥ 50 (e.g., tai60a, tai80a)

## Running QAPLIB Benchmarks

```python
from Librex.Librex.QAP import optimize_qap
from Librex.utils.qaplib_loader import load_qaplib_instance

# Load instance
flow, distance, optimal = load_qaplib_instance("chr12a")

# Optimize
result = optimize_qap(
    flow, distance,
    method="simulated_annealing",
    config={"iterations": 10000}
)

# Calculate gap from optimal
gap = (result["objective"] - optimal) / optimal * 100
print(f"Gap from optimal: {gap:.2f}%")
```

## Performance Standards

Target gaps from optimal:
- **QAP instances**: ≤ 15% gap threshold
- **TSP instances**: ≤ 5% gap threshold

## Benchmark Validation

The compliance check workflow validates:
1. Benchmark results existence
2. Gap thresholds compliance
3. Statistical significance of results

## See Also

- [BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md) - General benchmarking
- [README.md](README.md) - Main documentation
