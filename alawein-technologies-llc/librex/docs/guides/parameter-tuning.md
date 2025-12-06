# How to Tune Optimization Method Parameters

**Goal:** Find optimal hyperparameters for maximum performance

**Time Required:** 1-2 hours

**Prerequisites:** Basic understanding of your method, benchmarking knowledge

---

## Overview

Hyperparameter tuning is crucial for method performance. This guide covers systematic approaches.

## Method 1: Grid Search

```python
from itertools import product
import numpy as np

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'momentum': [0.7, 0.8, 0.9],
    'iterations': [500, 1000]
}

best_score = float('inf')
best_params = {}

problem = load_qap_instance("data/qaplib/nug20.dat")

# Generate all combinations
for lr in param_grid['learning_rate']:
    for momentum in param_grid['momentum']:
        for iterations in param_grid['iterations']:
            result = your_method(
                problem,
                learning_rate=lr,
                momentum=momentum,
                iterations=iterations
            )

            if result.objective_value < best_score:
                best_score = result.objective_value
                best_params = {
                    'learning_rate': lr,
                    'momentum': momentum,
                    'iterations': iterations
                }

print(f"Best params: {best_params}")
print(f"Best score: {best_score}")
```

## Method 2: Random Search

```python
from scipy.stats import uniform

n_trials = 100
best_score = float('inf')

for trial in range(n_trials):
    # Sample random parameters
    lr = uniform.rvs(0.001, 1.0)
    momentum = uniform.rvs(0.0, 1.0)
    iterations = int(uniform.rvs(100, 5000))

    result = your_method(
        problem,
        learning_rate=lr,
        momentum=momentum,
        iterations=iterations
    )

    if result.objective_value < best_score:
        best_score = result.objective_value
        print(f"Trial {trial}: Found better params: {best_score:.2f}")
```

## Method 3: Bayesian Optimization

```python
from skopt import gp_minimize

def objective(params):
    """Function to minimize."""
    lr, momentum, iterations = params
    result = your_method(
        problem,
        learning_rate=lr,
        momentum=momentum,
        iterations=int(iterations)
    )
    return result.objective_value

# Define parameter ranges
space = [
    (0.001, 1.0),       # learning_rate
    (0.0, 1.0),         # momentum
    (100, 5000)         # iterations
]

# Optimize
result = gp_minimize(objective, space, n_calls=50, random_state=42)

print(f"Best params: {result.x}")
print(f"Best score: {result.fun}")
```

## Method 4: Sensitivity Analysis

```python
import matplotlib.pyplot as plt

# Vary one parameter at a time
learning_rates = np.logspace(-3, 0, 20)
results = []

for lr in learning_rates:
    result = your_method(problem, learning_rate=lr)
    results.append(result.objective_value)

plt.figure()
plt.semilogx(learning_rates, results, 'o-')
plt.xlabel('Learning Rate')
plt.ylabel('Objective Value')
plt.title('Sensitivity to Learning Rate')
plt.grid(True, alpha=0.3)
plt.show()
```

## Best Practices

✅ **DO:**
- Test on multiple instances
- Use validation set for tuning
- Test on different problem sizes
- Report final performance on held-out test set
- Document all parameters used

❌ **DON'T:**
- Tune on test set (causes overfitting)
- Use too large parameter ranges
- Ignore computational cost
- Report best trial results

---

**Happy tuning!** 🚀

Last Updated: November 2024
