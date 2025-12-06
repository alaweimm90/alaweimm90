# Benchmarking Guide

This guide explains how to benchmark Librex optimization methods.

## Overview

Librex includes comprehensive benchmarking infrastructure for evaluating optimization method performance against standard problem instances.

## QAPLIB Benchmarking

For QAP-specific benchmarking, see [QAPLIB_BENCHMARKING_GUIDE.md](QAPLIB_BENCHMARKING_GUIDE.md).

## Running Benchmarks

```bash
# Run all benchmarks
pytest tests/ -m benchmark

# Run specific method benchmarks
pytest tests/ -m benchmark -k "simulated_annealing"

# Generate benchmark report
make bench-html  # if Makefile is configured
```

## Performance Targets

- **Small instances** (n < 20): < 1 second, optimal solutions
- **Medium instances** (20 ≤ n < 50): < 30 seconds, < 1% optimality gap
- **Large instances** (n ≥ 50): < 5 minutes, < 2% optimality gap

## Metrics

Benchmarks measure:
- **Solution quality**: Gap from known optimal
- **Runtime**: Wall-clock time
- **Convergence**: Iterations to solution
- **Reliability**: Success rate across multiple runs

## See Also

- [QAPLIB_BENCHMARKING_GUIDE.md](QAPLIB_BENCHMARKING_GUIDE.md)
- [ADVANCED_METHODS.md](ADVANCED_METHODS.md)
