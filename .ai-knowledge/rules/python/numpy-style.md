# NumPy/JAX Style Guide

## Principles

1. **Vectorize everything**: Avoid explicit loops
2. **Broadcasting**: Use implicit dimension expansion
3. **Pure functions**: No side effects for JIT compatibility
4. **Type hints**: Specify array shapes in docstrings

## Examples

### Bad
```python
def compute_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total
```

### Good
```python
def compute_sum(arr: jnp.ndarray) -> float:
    """Sum array elements.
    
    Args:
        arr: Array of shape (n,)
    
    Returns:
        Sum of elements
    """
    return jnp.sum(arr)
```

## Array Shape Notation

Use Einstein notation in docstrings:
- `(n,)`: 1D array of length n
- `(n, m)`: 2D array
- `(*, 3)`: Any leading dims, last dim is 3
- `(b, n, n)`: Batch of square matrices
