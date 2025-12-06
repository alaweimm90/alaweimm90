# Conservation Laws Rule

## Principle

All physics simulations must respect conservation laws.

## Required Checks

### Energy Conservation
```python
def validate_energy_conservation(system, tolerance=1e-6):
    """Verify total energy is conserved."""
    E_initial = system.total_energy()
    system.step()
    E_final = system.total_energy()
    assert abs(E_final - E_initial) < tolerance
```

### Momentum Conservation
```python
def validate_momentum_conservation(system, tolerance=1e-6):
    """Verify total momentum is conserved."""
    p_initial = system.total_momentum()
    system.step()
    p_final = system.total_momentum()
    assert jnp.allclose(p_final, p_initial, atol=tolerance)
```

## When to Apply

- Molecular dynamics
- Quantum mechanics
- Optimization on manifolds
- Hamiltonian systems

## Testing

Add conservation tests to every physics module:
```python
def test_energy_conservation():
    system = PhysicsSystem()
    validate_energy_conservation(system)
```
