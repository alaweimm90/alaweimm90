# Librex.QAP Methods Catalog Overview

This document provides a comprehensive overview of all 34 optimization methods implemented in Librex.QAP, organized by category and functionality.

## Method Categories

### Novel Methods (14) ★ - Original Contributions
1. **FFT-Laplace Preconditioning** - Spectral acceleration using FFT
2. **Reverse-Time Saddle Escape** - Theoretical saddle point navigation  
3. **Entropy Continuation** - Adaptive entropy regularization
4. **IMEX Integration** - Implicit-explicit time stepping
5. **Adaptive Step Integrator** - Dynamic step size control
6. **Eigenvalue Monitoring** - Spectral stability tracking
7. **Multi-Scale Gradient** - Hierarchical gradient computation
8. **Adaptive Lambda Tuning** - Dynamic constraint parameter adjustment
9. **Probabilistic Rounding** - Stochastic discrete recovery
10. **Iterative Refinement** - Multi-pass solution polishing
11. **Combinatorial Clustering** - Problem structure exploitation
12. **Manifold Tracking** - Constraint manifold navigation
13. **Hybrid Sinkhorn-Forces** - Combined projection approaches
14. **Parallel Gradient Aggregation** - Distributed computation

### Baseline Methods (19) • - Literature Standards
1. **Basic Gradient Descent** - Standard first-order method
2. **Momentum Gradient** - Accelerated first-order approach
3. **Nesterov Acceleration** - Optimal convergence rate
4. **AdaGrad** - Adaptive gradient methods
5. **Sinkhorn Projection** - Entropic optimal transport
6. **Bregman Projection** - Generalized projection framework
7. **Constraint Forces** - Penalty-based constraints
8. **Hungarian Rounding** - Optimal assignment rounding
9. **2-opt Local Search** - Pairwise improvement
10. **3-opt Local Search** - Triple improvement
11. **k-opt Generalization** - General k-exchange
12. **Threshold Rounding** - Deterministic rounding
13. **Runge-Kutta 4** - High-order integration
14. **Explicit Euler** - Basic time integration
15. **Shannon Entropy** - Information-theoretic regularization
16. **Tsallis Entropy** - Non-extensive entropy
17. **Gradient Stagnation Detection** - Convergence monitoring
18. **Basin Clustering** - Landscape analysis
19. **Parallel Gradient** - Concurrent computation

### Hybrid Methods (1) ◆ - Adaptive Combinations
1. **Adaptive Hybrid Solver** - Dynamic method selection

## Functional Classification

### Gradient Computation (7 methods)
- Basic Gradient Descent •
- Momentum Gradient •
- Nesterov Acceleration •
- AdaGrad •
- Multi-Scale Gradient ★
- Parallel Gradient •
- Parallel Gradient Aggregation ★

### Constraint Handling (6 methods)
- Sinkhorn Projection •
- Bregman Projection •
- Constraint Forces •
- Adaptive Lambda Tuning ★
- Manifold Tracking ★
- Hybrid Sinkhorn-Forces ★

### Time Integration (5 methods)
- IMEX Integration ★
- Adaptive Step Integrator ★
- Runge-Kutta 4 •
- Explicit Euler •
- Reverse-Time Saddle Escape ★

### Discrete Recovery (4 methods)
- Hungarian Rounding •
- 2-opt Local Search •
- 3-opt Local Search •
- k-opt Generalization •
- Probabilistic Rounding ★
- Iterative Refinement ★
- Threshold Rounding •

### Preconditioning & Acceleration (3 methods)
- FFT-Laplace Preconditioning ★
- Entropy Continuation ★
- Eigenvalue Monitoring ★

### Landscape Navigation (3 methods)
- Reverse-Time Saddle Escape ★
- Gradient Stagnation Detection •
- Basin Clustering •

### Problem Structure (2 methods)
- Combinatorial Clustering ★
- Adaptive Hybrid Solver ◆

### Regularization (2 methods)
- Shannon Entropy •
- Tsallis Entropy •

## Performance Characteristics

### Computational Complexity
- **O(n² log n)**: FFT-Laplace Preconditioning
- **O(n³)**: Most gradient-based methods
- **O(n⁴)**: Some sophisticated preconditioners
- **O(k·n²)**: Local search methods (k = neighborhood size)

### Memory Requirements
- **O(n²)**: Standard matrix operations
- **O(n³)**: Some preconditioning schemes
- **O(n)**: Lightweight methods

### Convergence Properties
- **Linear**: Basic gradient methods
- **Quadratic**: Newton-type methods (when applicable)
- **Adaptive**: Modern accelerated methods
- **Problem-dependent**: Local search methods

## Usage Guidelines

### For Small Problems (n ≤ 20)
1. Start with **Basic Gradient Descent** for baseline
2. Apply **Hungarian Rounding** for discrete recovery
3. Polish with **2-opt Local Search**
4. Validate with **Eigenvalue Monitoring**

### For Medium Problems (20 < n ≤ 100)
1. Use **FFT-Laplace Preconditioning** for acceleration
2. Apply **IMEX Integration** for stability
3. Employ **Entropy Continuation** for robustness
4. Use **Adaptive Step Integrator** for efficiency

### For Large Problems (n > 100)
1. Enable **Parallel Gradient Aggregation**
2. Apply **Multi-Scale Gradient** computation
3. Use **Reverse-Time Saddle Escape** for landscape navigation
4. Implement **Probabilistic Rounding** for scalability

### For Adversarial Instances
1. Apply **Reverse-Time Saddle Escape**
2. Use **Entropy Continuation** with careful tuning
3. Enable **Eigenvalue Monitoring** for stability
4. Consider **Adaptive Hybrid Solver** for method selection

## Implementation Notes

All methods are implemented with:
- **Modular architecture**: Easy to combine and extend
- **Consistent interfaces**: Unified API across methods
- **Parameter validation**: Robust input checking
- **Performance monitoring**: Built-in profiling capabilities
- **Memory optimization**: Efficient memory usage patterns
- **Parallel support**: Multi-threading where applicable

## References

For detailed mathematical formulations, complexity analysis, and implementation details, see:
- [METHOD_REFERENCE_COMPLETE.md](summaries/METHOD_REFERENCE_COMPLETE.md)
- [COMPREHENSIVE_METHOD_INVENTORY.md](summaries/COMPREHENSIVE_METHOD_INVENTORY.md)
- [METHODS_REFERENCE_CONSOLIDATED.md](summaries/METHODS_REFERENCE_CONSOLIDATED.md)

For practical implementation examples, see:
- [complete_qap_methods_documentation.md](summaries/complete_qap_methods_documentation.md)
- [enhanced_qap_methods_documentation.md](summaries/enhanced_qap_methods_documentation.md)