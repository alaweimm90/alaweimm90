# QAPLIB: Benchmark Instances

Standard Quadratic Assignment Problem benchmark instances for testing and validation.

## What Is This Directory?

The **data/qaplib/** directory contains 14 benchmark instances from the QAPLIB database. These are standard test problems used to evaluate optimization methods.

**Quick Facts:**
- 14 instances ready to use
- Sizes: 12 to 50
- Standard QAPLIB format
- Documented optimal solutions (known or best known)
- Suitable for benchmarking

## Available Instances

### Small Instances (Sizes 12-20)

Perfect for quick testing and development:

| Instance | Size | Optimal | Type |
|----------|------|---------|------|
| `chr12c.dat` | 12 | 11156 | Christofides |
| `chr20a.dat` | 20 | 2192 | Christofides |
| `chr20b.dat` | 20 | 2298 | Christofides |
| `had12.dat` | 12 | 1652 | Hadley |
| `had20.dat` | 20 | 6922 | Hadley |
| `nug12.dat` | 12 | 578 | Nugent |
| `nug20.dat` | 20 | 2570 | Nugent |
| `rou20.dat` | 20 | 725524 | Roucairol |
| `tai12a.dat` | 12 | 224416 | Taillard |
| `tai20a.dat` | 20 | 703482 | Taillard |

### Medium Instances (Sizes 30-40)

For comprehensive benchmarking:

| Instance | Size | Best Known | Type |
|----------|------|-----------|------|
| `ste36a.dat` | 36 | 25395 | Steinberg |
| `tai30a.dat` | 30 | 1818146 | Taillard |
| `tai40a.dat` | 40 | 3139370 | Taillard |

### Large Instances (Size 50+)

For stress testing and optimization:

| Instance | Size | Best Known | Type |
|----------|------|-----------|------|
| `tai50a.dat` | 50 | 4893643 | Taillard |

## Using Benchmark Instances

### Quick Load

```python
from Librex.QAP.utils import load_qap_instance

# Load instance
problem = load_qap_instance("data/qaplib/nug20.dat")

print(f"Size: {problem.size}")
print(f"Distance matrix shape: {problem.distance_matrix.shape}")
print(f"Flow matrix shape: {problem.flow_matrix.shape}")
```

### Solve and Compare

```python
from Librex.QAP.core import OptimizationPipeline

pipeline = OptimizationPipeline(problem_size=20)
result = pipeline.solve(problem, method="fft_laplace")

optimal = 2570  # Known optimal for nug20.dat
gap = (result.objective_value - optimal) / optimal * 100

print(f"Solution: {result.objective_value}")
print(f"Optimal:  {optimal}")
print(f"Gap:      {gap:.2f}%")
```

### Benchmark All Methods

```python
from Librex.QAP.benchmarking_suite import BenchmarkSuite

suite = BenchmarkSuite()
results = suite.benchmark_all_instances(
    instances=["nug20.dat", "tai20a.dat", "tai30a.dat"],
    methods=["fft_laplace", "reverse_time", "genetic_algorithm"]
)

suite.print_results(results)
suite.save_results(results, "benchmark_results.csv")
```

## File Format

QAPLIB format is a standard text format:

```
n
Flow matrix (n x n)
Distance matrix (n x n)
```

### Example (First few lines of nug20.dat)

```
20
     0    76    49     50     45     61     57     70     42     66
    76     0    46     41     35     41     59     65     75     57
    ...
   57    59     54     55     50     66     72     68     89     69
```

### Loading Custom Format

If needed to load custom format:

```python
import numpy as np

def load_custom_format(filename):
    """Load custom format if different from QAPLIB."""
    # Implement based on your format
    pass

# Use custom loader
problem = load_custom_format("my_instance.txt")
```

## Instance Categories

### By Size

- **Tiny** (n<15): Quick tests, learning
- **Small** (15≤n<30): Fast optimization
- **Medium** (30≤n<50): Reasonable benchmarks
- **Large** (n≥50): Stress tests

### By Type

- **Christofides**: Facility location style
- **Hadley**: Classical instances
- **Nugent**: Commonly used
- **Roucairol**: Special structure
- **Taillard**: Modern benchmarks
- **Steinberg**: Special class

## Best Known Solutions

See `MANIFEST.md` for:
- Optimal solutions (when known)
- Best known solutions
- Solution provenance
- Instance creation date

## Adding New Instances

To add a custom QAP instance:

1. **Create file** in QAPLIB format:
   ```
   n
   flow_matrix (n x n)
   distance_matrix (n x n)
   ```

2. **Validate format**:
   ```bash
   python data/qaplib/validate_format.py my_instance.dat
   ```

3. **Test loading**:
   ```python
   problem = load_qap_instance("data/qaplib/my_instance.dat")
   assert problem is not None
   ```

4. **Update MANIFEST.md** with instance details

5. **Commit** with clear message

## Validation

Validate instance files:

```bash
# Validate single file
python data/qaplib/validate_format.py chr12c.dat

# Validate all files
python data/qaplib/validate_format.py *.dat

# Output:
# ✓ chr12c.dat (size: 12, valid)
# ✓ chr20a.dat (size: 20, valid)
# ...
```

## Performance Tips

### For Quick Testing
Use small instances:
- `nug12.dat` (size 12)
- `had12.dat` (size 12)
- `chr12c.dat` (size 12)

### For Real Benchmarks
Use medium instances:
- `nug20.dat` (size 20)
- `tai20a.dat` (size 20)
- `tai30a.dat` (size 30)

### For Stress Testing
Use large instances:
- `tai40a.dat` (size 40)
- `tai50a.dat` (size 50)

## Benchmark Standards

When benchmarking:

1. **Warm up** - Run one iteration first
2. **Multiple runs** - Run each at least 3 times
3. **Time execution** - Not just quality
4. **Compare methods** - Against same instances
5. **Report** - Include size, time, quality

Example:

```python
import time

instance = "data/qaplib/nug20.dat"
problem = load_qap_instance(instance)

times = []
values = []

for i in range(5):  # 5 runs
    start = time.time()
    result = pipeline.solve(problem, method="fft_laplace")
    elapsed = time.time() - start

    times.append(elapsed)
    values.append(result.objective_value)

print(f"Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
print(f"Quality: {np.mean(values):.0f} ± {np.std(values):.0f}")
```

## Related Documentation

- **PROJECT.md** - Project overview
- **STRUCTURE.md** - Directory structure
- `MANIFEST.md` - Instance details
- `validate_format.py` - Validation script
- `DOWNLOAD_REPORT.txt` - Download history

## Quick Reference

### Common Tasks

```python
# Load instance
problem = load_qap_instance("data/qaplib/nug20.dat")

# Get problem info
print(problem.size)
print(problem.optimal_known)
print(problem.best_known_value)

# Benchmark all
from Librex.QAP.benchmarking_suite import BenchmarkSuite
suite = BenchmarkSuite()
results = suite.benchmark_all_instances()
```

### Instance Sizes

```
Tiny:    chr12c.dat, had12.dat, nug12.dat, tai12a.dat (n=12)
Small:   chr20a.dat, chr20b.dat, had20.dat, nug20.dat, rou20.dat, tai20a.dat (n=20)
Medium:  ste36a.dat (n=36), tai30a.dat (n=30), tai40a.dat (n=40)
Large:   tai50a.dat (n=50)
```

### Optimal Values (When Known)

```
chr12c:  11156
chr20a:  2192
nug12:   578
nug20:   2570
tai12a:  224416
tai20a:  703482
```

## Statistics

```
Total Instances:  14
Total Size Range: 12-50
Average Size:     ~28
Small (≤20):      10 instances
Medium (>20):     4 instances

Total Test Capacity: 196 problems (14 instances × 14 sizes)
Benchmark Time:      ~2-3 minutes (all methods × all instances)
```

## References

- QAPLIB: http://www.seas.upenn.edu/~qaplib/
- Problem descriptions in MANIFEST.md
- Best known solutions documented

## Notes

- These instances are **read-only** (don't modify)
- All sizes are from QAPLIB standard
- Solutions are documented best-known values
- Use for scientific benchmarking
- Cite QAPLIB when publishing results

---

**Happy benchmarking!** 🚀

Questions? See `PROJECT.md` or `STRUCTURE.md`.

Last Updated: November 2024
