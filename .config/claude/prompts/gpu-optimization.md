# GPU Optimization Superprompt

> **Optimize Python code for GPU acceleration with JAX/CUDA**

## Metadata

```yaml
name: gpu-optimization
version: 1.0.0
tags: [gpu, cuda, jax, performance, optimization]
tools: [amazon-q, claude, windsurf, cline]
domain: computational-physics
author: Meshal Alawein
last_updated: 2025-01-XX
```

## Purpose

Transform CPU-bound NumPy code to GPU-accelerated JAX code with 5-10x speedup.

## Prompt

```markdown
I need to optimize this code for GPU execution.

## Context
- **Current Implementation**: NumPy on CPU
- **Target**: JAX on GPU
- **Expected Speedup**: 5-10x
- **Hardware**: [GPU model, e.g., NVIDIA RTX 4090]

## Code to Optimize
[Paste your code here]

## Requirements

### 1. JAX Conversion
- Replace NumPy with JAX NumPy (jnp)
- Use jax.jit for compilation
- Vectorize operations with jax.vmap
- Handle random number generation properly

### 2. Performance
- Profile before/after with timeit
- Minimize host-device transfers
- Batch operations when possible
- Use appropriate dtypes (float32 for GPU)

### 3. Correctness
- Verify numerical equivalence
- Test edge cases
- Check memory usage
- Validate gradients if using autodiff

## Workflow

1. **Profile**: Identify bottlenecks
2. **Convert**: NumPy → JAX NumPy
3. **JIT**: Add @jax.jit decorators
4. **Vectorize**: Use vmap for batch operations
5. **Benchmark**: Compare CPU vs GPU
6. **Validate**: Ensure correctness

## Example

### Before (NumPy)
```python
import numpy as np

def compute_energy(positions, charges):
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            energy += charges[i] * charges[j] / r
    return energy
```

### After (JAX)
```python
import jax.numpy as jnp
from jax import jit, vmap

@jit
def compute_energy(positions, charges):
    # Vectorized pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]
    r = jnp.linalg.norm(diff, axis=-1)
    
    # Avoid self-interaction
    r = jnp.where(r == 0, jnp.inf, r)
    
    # Vectorized energy computation
    charge_products = charges[:, None] * charges[None, :]
    energy = jnp.sum(jnp.triu(charge_products / r, k=1))
    
    return energy
```

## Success Criteria
- [ ] 5-10x speedup on GPU
- [ ] Numerical accuracy maintained
- [ ] Memory usage acceptable
- [ ] Code remains readable
```

## Usage

```
@prompt gpu-optimization

Code:
[paste your NumPy code]

Hardware: NVIDIA RTX 4090
```

## Related Resources

- [Performance Review](../code-review/performance-review.md)
- [Benchmark Workflow](../../workflows/development/benchmark-driven-optimization.py)
