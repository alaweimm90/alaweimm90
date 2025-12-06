# Librex (Optilibria) v2.0 Roadmap

## Current State
- Comprehensive optimization library with 31+ algorithms
- JAX-based GPU acceleration
- QAP, TSP, portfolio optimization, NAS support
- Used in Berkeley materials discovery

## v2.0 Goals
- Add L-BFGS and SR1 quasi-Newton methods
- Benchmark vs SciPy
- Production documentation
- PyPI release as "Librex"

## Week 1: L-BFGS

**File**: `Librex/methods/advanced/lbfgs.py`

```python
import jax.numpy as jnp
from jax import grad, jit

@jit
def lbfgs_step(x, grad_x, s_hist, y_hist):
    q = grad_x
    alpha = []
    
    for i in range(len(s_hist)-1, -1, -1):
        rho = 1.0 / jnp.dot(y_hist[i], s_hist[i])
        a = rho * jnp.dot(s_hist[i], q)
        alpha.append(a)
        q = q - a * y_hist[i]
    
    if len(s_hist) > 0:
        gamma = jnp.dot(s_hist[-1], y_hist[-1]) / jnp.dot(y_hist[-1], y_hist[-1])
        r = gamma * q
    else:
        r = q
    
    for i in range(len(s_hist)):
        rho = 1.0 / jnp.dot(y_hist[i], s_hist[i])
        beta = rho * jnp.dot(y_hist[i], r)
        r = r + s_hist[i] * (alpha[len(s_hist)-1-i] - beta)
    
    return -r

def optimize_lbfgs(f, x0, max_iter=100, tol=1e-6, m=10):
    x = x0
    s_hist, y_hist = [], []
    grad_f = grad(f)
    
    for i in range(max_iter):
        g = grad_f(x)
        if jnp.linalg.norm(g) < tol:
            break
        
        p = lbfgs_step(x, g, s_hist, y_hist)
        
        alpha = 1.0
        while f(x + alpha * p) > f(x) + 1e-4 * alpha * jnp.dot(g, p):
            alpha *= 0.5
            if alpha < 1e-10:
                break
        
        x_new = x + alpha * p
        s = x_new - x
        y = grad_f(x_new) - g
        
        s_hist.append(s)
        y_hist.append(y)
        if len(s_hist) > m:
            s_hist.pop(0)
            y_hist.pop(0)
        
        x = x_new
    
    return x
```

**Test**: `tests/methods/test_lbfgs.py`

## Week 2: SR1 + Benchmarks

**File**: `Librex/methods/advanced/sr1.py`

```python
def optimize_sr1(f, x0, max_iter=100, tol=1e-6):
    x = x0
    n = len(x0)
    B = jnp.eye(n)
    grad_f = grad(f)
    
    for i in range(max_iter):
        g = grad_f(x)
        if jnp.linalg.norm(g) < tol:
            break
        
        p = -jnp.linalg.solve(B, g)
        alpha = 1.0
        while f(x + alpha * p) > f(x) + 1e-4 * alpha * jnp.dot(g, p):
            alpha *= 0.5
        
        x_new = x + alpha * p
        s = x_new - x
        y = grad_f(x_new) - g
        
        # SR1 update
        Bs = B @ s
        denom = jnp.dot(y - Bs, s)
        if abs(denom) > 1e-8:
            B = B + jnp.outer(y - Bs, y - Bs) / denom
        
        x = x_new
    
    return x
```

**Benchmark**: `benchmarks/scipy_comparison.py`

```python
from scipy.optimize import minimize
import time

problems = {
    'rosenbrock': lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2,
    'sphere': lambda x: jnp.sum(x**2)
}

for name, f in problems.items():
    x0 = jnp.ones(10) * 5.0
    
    start = time.time()
    eq_result = optimize_lbfgs(f, x0)
    eq_time = time.time() - start
    
    start = time.time()
    sp_result = minimize(f, x0, method='L-BFGS-B')
    sp_time = time.time() - start
    
    print(f"{name}: Librex {eq_time:.4f}s, SciPy {sp_time:.4f}s")
```

## Week 3: Documentation + Release

**README.md**
```markdown
# Librex

GPU-accelerated optimization library.

## Install
pip install Librex

## Quick Start
from Librex import LBFGS

def f(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

result = LBFGS().optimize(f, x0=[0, 0], gpu=True)
```

**PyPI Release**
- Update `pyproject.toml`
- Build: `python -m build`
- Upload: `twine upload dist/*`

## Timeline
- Week 1: L-BFGS implementation + tests
- Week 2: SR1 + benchmarking
- Week 3: Docs + PyPI release
