# Librex Quick Start Guide

Get started with Librex in 5 minutes!

## Installation

```bash
# Clone and install
cd /path/to/AlaweinOS/Librex
pip install -e .
```

## Your First Optimization (30 seconds)

```python
import numpy as np
from Librex.Librex.QAP import optimize_qap

# Define a simple QAP instance
flow = np.array([[0, 5, 2], [5, 0, 3], [2, 3, 0]])
distance = np.array([[0, 8, 15], [8, 0, 13], [15, 13, 0]])

# Optimize!
result = optimize_qap(flow, distance, method="simulated_annealing")

print(f"Solution: {result['solution']}")
print(f"Objective: {result['objective']}")
```

**Output:**
```
Solution: [0 2 1]  # or similar permutation
Objective: 146.0   # or similar value
```

## Run Examples

```bash
# QAP example
python examples/qap/simple_qap_example.py

# TSP example
python examples/tsp/simple_tsp_example.py
```

## Next Steps

1. **Read the README:** Full documentation in [README.md](../README.md)
2. **Try different methods:** `random_search`, `simulated_annealing`, `local_search`, `genetic_algorithm`, `tabu_search`
3. **Explore adapters:** [QAP](../Librex/adapters/qap/), [TSP](../Librex/adapters/tsp/)

## Common Patterns

### Pattern 1: Compare Multiple Methods

```python
from Librex import optimize
from Librex.adapters.qap import QAPAdapter

problem = {...}  # Your QAP instance
adapter = QAPAdapter()

for method in ['simulated_annealing', 'genetic_algorithm', 'tabu_search']:
    result = optimize(problem, adapter, method=method)
    print(f"{method}: {result['objective']}")
```

### Pattern 2: Tune Hyperparameters

```python
result = optimize(
    problem,
    adapter,
    method='simulated_annealing',
    config={
        'iterations': 50000,
        'initial_temp': 200.0,
        'cooling_rate': 0.98,
        'seed': 42  # For reproducibility
    }
)
```

### Pattern 3: Custom TSP from Coordinates

```python
from Librex import optimize
from Librex.adapters.tsp import TSPAdapter

cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
problem = {'coordinates': cities}

adapter = TSPAdapter()
result = optimize(problem, adapter, method='genetic_algorithm')
```

## Need Help?

- **Documentation:** See [docs/](../docs/)
- **Examples:** See [examples/](.)
- **Issues:** https://github.com/AlaweinOS/AlaweinOS/issues

## Tips

✅ **Start with `simulated_annealing`** - Best general-purpose method
✅ **Use `seed=42`** for reproducible results during testing
✅ **Check `result['is_valid']`** to ensure solution validity
✅ **Compare multiple methods** to find the best for your problem

---

**Happy Optimizing!** 🚀
