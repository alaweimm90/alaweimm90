# Librex.QAP Notation Standards

This document establishes consistent mathematical notation across all Librex.QAP documentation and implementations.

## Problem Formulation

### Standard QAP Formulation
```
minimize    f(X) = trace(A X B X^T)
subject to  X ∈ Π_n
```

Where:
- **A, B** ∈ ℝ^{n×n}: Flow and distance matrices
- **X** ∈ ℝ^{n×n}: Assignment matrix (permutation or doubly stochastic)
- **Π_n**: Set of n×n permutation matrices
- **trace()**: Matrix trace operator

### Birkhoff Polytope Relaxation
```
minimize    f(X) = trace(A X B X^T)
subject to  X ∈ B_n
```

Where:
- **B_n**: Birkhoff polytope (set of doubly stochastic matrices)
- **X** ≥ 0: Element-wise non-negativity
- **X1 = 1**: Row sum constraints
- **X^T1 = 1**: Column sum constraints

## Matrix Notation

### Basic Matrices
| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| A, B | n×n | Problem data matrices (flow, distance) |
| X | n×n | Continuous solution matrix |
| P | n×n | Permutation matrix (discrete solution) |
| I | n×n | Identity matrix |
| 1 | n×1 | Vector of ones |
| 0 | n×1 | Vector of zeros |

### Derived Matrices
| Symbol | Meaning |
|--------|---------|
| ∇f(X) | Gradient of objective function |
| H_f(X) | Hessian of objective function |
| J | Jacobian of constraints |
| L | Lagrangian function |

## Vector Notation

### Solution Vectors
| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| x | n²×1 | Vectorized form of X |
| p | n²×1 | Vectorized form of P |
| λ | m×1 | Lagrange multipliers |
| μ | m×1 | Dual variables |

### Gradient Vectors
| Symbol | Meaning |
|--------|---------|
| g | Gradient vector (∇f(x)) |
| g_k | Gradient at iteration k |
| g^T | Gradient transpose |

## Algorithmic Notation

### Iteration Counters
| Symbol | Meaning |
|--------|---------|
| k | Iteration counter |
| t | Time step / continuous time |
| i, j | Matrix element indices |
| α, β | Algorithm parameters |

### Step Sizes and Parameters
| Symbol | Meaning |
|--------|---------|
| η | Learning rate / step size |
| τ | Temperature parameter |
| ε | Convergence tolerance |
| δ | Perturbation parameter |
| λ | Regularization parameter |

## Complexity Notation

### Time Complexity
| Symbol | Meaning |
|--------|---------|
| O(n³) | Cubic complexity |
| O(n² log n) | Near-quadratic complexity |
| O(n!) | Factorial complexity |
| O(2^n) | Exponential complexity |

### Space Complexity
| Symbol | Meaning |
|--------|---------|
| O(n²) | Quadratic memory |
| O(n) | Linear memory |
| O(1) | Constant memory |

## Optimization Notation

### Function Values
| Symbol | Meaning |
|--------|---------|
| f(x) | Objective function value |
| f_k | Objective value at iteration k |
| f* | Optimal objective value |
| Δf | Function value improvement |

### Constraint Functions
| Symbol | Meaning |
|--------|---------|
| c(x) | Constraint function |
| c_i(x) | i-th constraint function |
| c_eq(x) | Equality constraints |
| c_ineq(x) | Inequality constraints |

## Probability and Statistics

### Random Variables
| Symbol | Meaning |
|--------|---------|
| X | Random matrix |
| x | Random vector |
| P(·) | Probability measure |
| E[·] | Expectation operator |
| Var[·] | Variance operator |

### Distributions
| Symbol | Meaning |
|--------|---------|
| N(μ, Σ) | Normal distribution |
| U(a, b) | Uniform distribution |
| Exp(λ) | Exponential distribution |
| Ber(p) | Bernoulli distribution |

## Quantum Notation (Specialized)

### Quantum States
| Symbol | Meaning |
|--------|---------|
| |ψ⟩ | Quantum state vector |
| ⟨ψ| | Dual state vector |
| ⟨ψ|φ⟩ | Inner product |
| |ψ⟩⟨φ| | Outer product |
| ρ | Density matrix |

### Quantum Operators
| Symbol | Meaning |
|--------|---------|
| H | Hamiltonian operator |
| U | Unitary operator |
| σ | Pauli matrices |
| ℏ | Reduced Planck constant |

## Implementation Standards

### Code Documentation
- Use consistent variable names matching mathematical notation
- Document complexity bounds for all algorithms
- Include convergence criteria and stopping conditions
- Provide parameter tuning guidelines

### File Naming
- Use lowercase with underscores: `qap_solver.py`
- Include version numbers: `method_v2.py`
- Add descriptors: `fft_preconditioner.py`

### Function Signatures
```python
def solve_qap(A: np.ndarray, B: np.ndarray, 
              method: str = "fft_laplace",
              max_iter: int = 1000,
              tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    Solve QAP using specified method.
    
    Parameters:
    -----------
    A : np.ndarray, shape (n, n)
        Flow matrix
    B : np.ndarray, shape (n, n) 
        Distance matrix
    method : str
        Solution method to use
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
        
    Returns:
    --------
    X : np.ndarray, shape (n, n)
        Optimal assignment matrix
    f_val : float
        Objective function value
    """
```

## Cross-Reference Standards

### Document Citations
- Use consistent citation format: `[Author Year]`
- Include DOI when available: `[DOI:10.1000/abc123]`
- Reference equations: `Equation (3.2)`
- Reference sections: `Section 2.1`

### Internal References
- Method references: `Method 4.1`
- Algorithm references: `Algorithm 3`
- Figure references: `Figure 2.1`
- Table references: `Table 1`

## Validation Checklist

### Mathematical Consistency
- [ ] All matrices use consistent capitalization
- [ ] Vector notation follows standards
- [ ] Greek letters used appropriately
- [ ] Subscripts and superscripts correct
- [ ] Function notation consistent

### Algorithm Consistency  
- [ ] Iteration counters consistent
- [ ] Step size notation uniform
- [ ] Convergence criteria clear
- [ ] Complexity bounds stated
- [ ] Parameter definitions complete

### Implementation Consistency
- [ ] Code matches mathematical notation
- [ ] Variable names follow standards
- [ ] Documentation complete
- [ ] Type hints included
- [ ] Error handling specified

---

*This notation standard ensures consistency across all Librex.QAP documentation, implementations, and communications.*