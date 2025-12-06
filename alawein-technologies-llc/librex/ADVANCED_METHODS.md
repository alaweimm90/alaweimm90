# Advanced Methods

Documentation for advanced and novel optimization methods in Librex.

## Baseline Methods

Librex includes 5 production-ready baseline methods:

### 1. Random Search
- **Type**: Baseline
- **Best for**: Quick baseline, small problems
- **Status**: ✅ Ready

### 2. Simulated Annealing
- **Type**: Metaheuristic
- **Best for**: General-purpose, medium problems
- **Status**: ✅ Ready

### 3. Local Search
- **Type**: Hill Climbing
- **Best for**: Fast local optimization
- **Status**: ✅ Ready

### 4. Genetic Algorithm
- **Type**: Evolutionary
- **Best for**: Large problems, population-based
- **Status**: ✅ Ready

### 5. Tabu Search
- **Type**: Metaheuristic
- **Best for**: Avoiding local optima, memory-based
- **Status**: ✅ Ready

## Novel Methods (Research Grade)

### FFT-Laplace Preconditioning
- **Status**: ⚠️ **DEPRECATED** - Under mathematical review
- **Reason**: Fundamental issues with spectral Laplacian formulation for discrete optimization
- **Alternative**: Use baseline methods listed above

### Reverse-Time Saddle Escape
- **Status**: 🚧 Experimental
- **Documentation**: See method-specific documentation

## Method Selection

Use the AI method selector for automatic algorithm selection:

```python
from Librex.ai.method_selector import select_method

method = select_method(problem_features)
```

## See Also

- [NOVEL_METHODS.md](NOVEL_METHODS.md) - Detailed novel method documentation
- [README.md](README.md) - Main documentation
