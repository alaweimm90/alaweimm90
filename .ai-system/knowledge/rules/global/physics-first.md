# Physics-First Development Rule

Always prioritize physics correctness over optimization.

## Principle

Before optimizing code, verify:
1. Conservation laws hold
2. Units are consistent
3. Boundary conditions are correct
4. Physical constraints are respected

## Example

```python
# Bad: Fast but wrong
def optimize(x):
    return x * 2  # Violates energy conservation

# Good: Correct physics
def optimize(x):
    energy_before = compute_energy(x)
    x_new = gradient_step(x)
    x_new = project_to_energy_surface(x_new, energy_before)
    return x_new
```
