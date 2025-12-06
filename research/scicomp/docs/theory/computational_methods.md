# Computational Methods Theory Reference
**SciComp - Computational Foundation**
*Author: Meshal Alawein (meshal@berkeley.edu)*
*Institution: University of California, Berkeley*
*Created: 2025*
---
## Table of Contents
1. [Numerical Analysis Fundamentals](#numerical-analysis-fundamentals)
2. [Finite Difference Methods](#finite-difference-methods)
3. [Finite Element Methods](#finite-element-methods)
4. [Spectral Methods](#spectral-methods)
5. [Time Integration Schemes](#time-integration-schemes)
6. [Monte Carlo Methods](#monte-carlo-methods)
7. [Linear Algebra Algorithms](#linear-algebra-algorithms)
8. [Optimization Algorithms](#optimization-algorithms)
---
## Numerical Analysis Fundamentals
### Error Analysis
**Absolute Error**: E_abs = |x_true - x_computed|
**Relative Error**: E_rel = |x_true - x_computed| / |x_true|
**Types of Errors**:
- **Truncation Error**: From approximating infinite processes
- **Round-off Error**: From finite precision arithmetic
- **Discretization Error**: From spatial/temporal discretization
**Error Propagation**: In function f(x₁, x₂, ..., xₙ):
```
δf ≈ Σᵢ |∂f/∂xᵢ| δxᵢ
```
### Condition Numbers
**Definition**: κ(A) = ||A|| ||A⁻¹||
**Physical Interpretation**: How much the output changes relative to input changes
**Well-Conditioned**: κ(A) ≈ 1
**Ill-Conditioned**: κ(A) >> 1
### Convergence Analysis
**Order of Convergence**: If error Eₙ₊₁ ≈ C·Eₙᵖ, then order p
**Convergence Rates**:
- Linear: p = 1
- Quadratic: p = 2
- Exponential: Error ~ e^(-αn)
### Stability Analysis
**Von Neumann Stability**: For linear schemes, analyze growth of Fourier modes
**CFL Condition**: For explicit time-stepping schemes:
```
Δt ≤ C·Δx/|u_max|
```
---
## Finite Difference Methods
### Taylor Series Foundation
**Forward Difference**:
```
f'(x) ≈ (f(x+h) - f(x))/h + O(h)
```
**Central Difference**:
```
f'(x) ≈ (f(x+h) - f(x-h))/(2h) + O(h²)
```
**Second Derivative**:
```
f''(x) ≈ (f(x+h) - 2f(x) + f(x-h))/h² + O(h²)
```
### Higher-Order Schemes
**Fourth-Order Central Difference**:
```
f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h))/(12h) + O(h⁴)
```
**Compact Schemes**: Higher accuracy with smaller stencils
```
f'ᵢ₋₁ + 4f'ᵢ + f'ᵢ₊₁ = 3(fᵢ₊₁ - fᵢ₋₁)/h
```
### Boundary Conditions
**Dirichlet**: u(boundary) = g(boundary)
**Neumann**: ∂u/∂n(boundary) = g(boundary)
**Robin**: αu + β∂u/∂n = g
**Implementation Strategies**:
- Ghost points
- One-sided differences
- Boundary modification of stencils
### Stability and Accuracy
**Lax Equivalence Theorem**: For linear problems:
Consistency + Stability ⟺ Convergence
**Modified Equation Analysis**: What PDE is actually being solved numerically
### Applications
**Heat Equation**: ∂u/∂t = α∇²u
**Wave Equation**: ∂²u/∂t² = c²∇²u
**Advection Equation**: ∂u/∂t + c∂u/∂x = 0
---
## Finite Element Methods
### Variational Formulation
**Strong Form**: Lu = f in Ω, Bu = g on ∂Ω
**Weak Form**: Find u ∈ V such that a(u,v) = L(v) ∀v ∈ V₀
**Bilinear Form**: a(u,v) = ∫_Ω ∇u · ∇v dx
**Linear Functional**: L(v) = ∫_Ω fv dx
### Galerkin Method
**Approximation**: u ≈ uₕ = Σᵢ uᵢφᵢ(x)
**Discrete System**: Σᵢ uᵢa(φᵢ,φⱼ) = L(φⱼ) ∀j
**Matrix Form**: Ku = f
- K_ij = a(φᵢ,φⱼ): Stiffness matrix
- f_j = L(φⱼ): Load vector
### Element Types
**1D Elements**:
- Linear: φ₁ = (1-ξ)/2, φ₂ = (1+ξ)/2
- Quadratic: φ₁ = ξ(ξ-1)/2, φ₂ = 1-ξ², φ₃ = ξ(ξ+1)/2
**2D Elements**:
- Triangular: Area coordinates
- Quadrilateral: Tensor products of 1D basis
**3D Elements**:
- Tetrahedral, hexahedral, prismatic
### Mesh Generation
**Delaunay Triangulation**: Maximizes minimum angle
**Advancing Front**: Grows mesh from boundaries
**Quadtree/Octree**: Hierarchical refinement
**Quality Measures**:
- Aspect ratio
- Skewness
- Jacobian determinant
### Error Estimation and Adaptivity
**A Posteriori Error Estimates**: Based on computed solution
**h-Refinement**: Decrease element size
**p-Refinement**: Increase polynomial order
**r-Refinement**: Relocate nodes
**Adaptive Strategy**:
1. Solve on current mesh
2. Estimate error
3. Mark elements for refinement
4. Refine mesh
5. Repeat
### Assembly Process
**Element Matrix**: K^e_ij = ∫_Ωᵉ ∇φⁱ · ∇φʲ dx
**Global Assembly**: K = Σₑ A^T K^e A
**Numerical Integration**: Gaussian quadrature
```
∫₋₁¹ f(ξ)dξ ≈ Σᵢ wᵢf(ξᵢ)
```
---
## Spectral Methods
### Theoretical Foundation
**Spectral Accuracy**: Exponential convergence for smooth functions
**Gibbs Phenomenon**: Oscillations near discontinuities
### Fourier Spectral Methods
**Basis Functions**: e^(ikx), k = -N/2, ..., N/2-1
**Discrete Fourier Transform**:
```
û_k = (1/N) Σⁿ⁼⁰^(N-1) u_n e^(-2πikn/N)
```
**Fast Fourier Transform**: O(N log N) algorithm
**Differentiation**: (∂u/∂x)̂_k = ik û_k
### Chebyshev Spectral Methods
**Chebyshev Points**: x_j = cos(πj/N), j = 0, ..., N
**Basis Functions**: T_n(x) = cos(n arccos(x))
**Differentiation Matrix**: D_ij = dT_j/dx|_{x_i}
**Advantages**:
- Non-periodic boundaries
- Exponential convergence
- Clustering near boundaries
### Spectral Element Methods
**Combination**: Finite element flexibility + spectral accuracy
**High-Order Elements**: Legendre-Gauss-Lobatto points
**C⁰ Continuity**: Only function values continuous
### Applications
**Fluid Dynamics**: Navier-Stokes equations
**Wave Propagation**: Acoustic and electromagnetic waves
**Quantum Mechanics**: Schrödinger equation
---
## Time Integration Schemes
### Explicit Methods
**Forward Euler**: u^(n+1) = u^n + Δt f(u^n, t^n)
- First-order accurate
- Conditionally stable
**Runge-Kutta Methods**:
**RK4**: Fourth-order accurate
```
k₁ = f(uⁿ, tⁿ)
k₂ = f(uⁿ + Δt k₁/2, tⁿ + Δt/2)
k₃ = f(uⁿ + Δt k₂/2, tⁿ + Δt/2)
k₄ = f(uⁿ + Δt k₃, tⁿ + Δt)
u^(n+1) = uⁿ + Δt(k₁ + 2k₂ + 2k₃ + k₄)/6
```
### Implicit Methods
**Backward Euler**: u^(n+1) = u^n + Δt f(u^(n+1), t^(n+1))
- First-order accurate
- Unconditionally stable
- Requires nonlinear solve
**Trapezoidal Rule**: u^(n+1) = u^n + Δt(f(u^n, t^n) + f(u^(n+1), t^(n+1)))/2
- Second-order accurate
- A-stable
### Multistep Methods
**Adams-Bashforth**: Explicit, based on polynomial extrapolation
**Adams-Moulton**: Implicit, higher stability
**BDF Methods**: Backward differentiation formulas
- Good for stiff problems
- Up to sixth order
### Stiff Equations
**Definition**: Problems where explicit methods require very small time steps
**Characteristics**:
- Multiple time scales
- Some modes decay rapidly
**L-Stability**: Damping of high-frequency modes
### Adaptive Time Stepping
**Error Control**: Estimate local truncation error
**Step Size Selection**:
```
Δt_new = Δt_old (tolerance/error)^(1/(p+1))
```
**Embedded Methods**: Runge-Kutta-Fehlberg, Dormand-Prince
---
## Monte Carlo Methods
### Random Number Generation
**Linear Congruential Generator**:
```
xₙ₊₁ = (axₙ + c) mod m
```
**Mersenne Twister**: High-quality pseudorandom numbers
**Quasi-Random Sequences**: Low-discrepancy sequences (Sobol, Halton)
### Basic Monte Carlo Integration
**Estimate**: ∫_Ω f(x)dx ≈ (|Ω|/N) Σᵢ f(xᵢ)
**Error**: σ/√N where σ² = Var[f]
**Convergence Rate**: O(N^(-1/2)) independent of dimension
### Variance Reduction
**Importance Sampling**: Sample from distribution close to |f|
**Control Variates**: Use correlated function with known integral
**Antithetic Variables**: Use negatively correlated samples
**Stratified Sampling**: Divide domain into strata
### Markov Chain Monte Carlo
**Metropolis-Hastings Algorithm**:
1. Propose new state x'
2. Compute acceptance ratio α = min(1, π(x')/π(x))
3. Accept with probability α
**Gibbs Sampling**: Sample each component conditionally
**Hamiltonian Monte Carlo**: Use physical dynamics for better proposals
### Quantum Monte Carlo
**Variational Monte Carlo**: Optimize trial wavefunction
**Diffusion Monte Carlo**: Project to ground state
**Path Integral Monte Carlo**: Sample quantum paths
### Applications
**Physics**: Statistical mechanics, quantum systems
**Finance**: Option pricing, risk assessment
**Engineering**: Reliability analysis, optimization
---
## Linear Algebra Algorithms
### Direct Methods
**LU Decomposition**: A = LU
- Forward substitution: Ly = b
- Back substitution: Ux = y
- Complexity: O(n³)
**Cholesky Decomposition**: A = LLᵀ (for symmetric positive definite)
- More efficient: O(n³/3)
- Better numerical stability
**QR Decomposition**: A = QR
- Q orthogonal, R upper triangular
- Applications: least squares, eigenvalue methods
### Iterative Methods
**Jacobi Method**:
```
x_i^(k+1) = (b_i - Σⱼ≠ᵢ a_ij x_j^(k))/a_ii
```
**Gauss-Seidel Method**:
```
x_i^(k+1) = (b_i - Σⱼ<ᵢ a_ij x_j^(k+1) - Σⱼ>ᵢ a_ij x_j^(k))/a_ii
```
**Successive Over-Relaxation (SOR)**:
```
x_i^(k+1) = (1-ω)x_i^(k) + ω(b_i - Σⱼ<ᵢ a_ij x_j^(k+1) - Σⱼ>ᵢ a_ij x_j^(k))/a_ii
```
### Krylov Subspace Methods
**Conjugate Gradient**: For symmetric positive definite systems
- Optimal in Krylov subspace
- Convergence depends on condition number
**GMRES**: For general nonsymmetric systems
- Minimizes residual in Krylov subspace
- Requires restart for large problems
**BiCGSTAB**: Stabilized biconjugate gradient
- Good for nonsymmetric systems
- Irregular convergence behavior
### Preconditioning
**Purpose**: Improve condition number κ(M⁻¹A)
**Jacobi Preconditioner**: M = diag(A)
**ILU Preconditioner**: Incomplete LU factorization
**Multigrid Preconditioner**: Multiple grid levels
### Eigenvalue Problems
**Power Method**: Finds largest eigenvalue
```
x^(k+1) = Ax^(k)/||Ax^(k)||
```
**QR Algorithm**: All eigenvalues
- Iterative application of QR decomposition
- Convergence to Schur form
**Arnoldi Method**: Large sparse matrices
- Builds orthonormal basis for Krylov subspace
- Finds several eigenvalues efficiently
---
## Optimization Algorithms
### Unconstrained Optimization
**Gradient Descent**:
```
x^(k+1) = x^k - α∇f(x^k)
```
**Newton's Method**:
```
x^(k+1) = x^k - (∇²f(x^k))⁻¹∇f(x^k)
```
**Quasi-Newton Methods**:
- BFGS: Approximate Hessian update
- L-BFGS: Limited memory version
**Trust Region Methods**:
- Solve subproblem in trusted region
- Adjust region based on success
### Constrained Optimization
**Lagrange Multipliers**: First-order necessary conditions
```
∇f(x*) + Σᵢ λᵢ∇gᵢ(x*) = 0
gᵢ(x*) = 0
```
**KKT Conditions**: For inequality constraints
```
∇f(x*) + Σᵢ λᵢ∇gᵢ(x*) + Σⱼ μⱼ∇hⱼ(x*) = 0
gᵢ(x*) = 0, hⱼ(x*) ≤ 0
μⱼ ≥ 0, μⱼhⱼ(x*) = 0
```
**Interior Point Methods**: Barrier functions
**Sequential Quadratic Programming**: Solve QP subproblems
**Penalty Methods**: Convert constraints to objective terms
### Global Optimization
**Simulated Annealing**: Probabilistic acceptance
**Genetic Algorithms**: Evolutionary approach
**Particle Swarm Optimization**: Swarm intelligence
**Differential Evolution**: Mutation and crossover
### Stochastic Optimization
**Stochastic Gradient Descent**: Use random subsets
**Momentum Methods**: Accelerate convergence
**Adam**: Adaptive moment estimation
**Natural Gradients**: Use information geometry
---
## Berkeley SciComp Implementation
### Software Architecture
**Multi-Language Support**:
- Python: SciPy, NumPy ecosystem
- MATLAB: Built-in linear algebra
- Mathematica: Symbolic computation
**Performance Optimization**:
- Vectorization
- Just-in-time compilation
- GPU acceleration
- Parallel computing
### Numerical Libraries
**BLAS/LAPACK**: Basic linear algebra
**FFTW**: Fast Fourier transforms
**PETSc**: Scalable linear algebra
**SUNDIALS**: ODE/DAE solvers
### Quality Assurance
**Verification**: Solve problems with known solutions
**Validation**: Compare with experimental data
**Unit Testing**: Test individual components
**Integration Testing**: Test combined systems
### Documentation
**Theory Manuals**: Mathematical foundations
**User Guides**: Practical instructions
**API References**: Detailed function descriptions
**Examples**: Worked problems and demonstrations
---
## References
1. Quarteroni, A., Sacco, R., Saleri, F. "Numerical Mathematics" (Springer, 2007)
2. Trefethen, L.N. "Spectral Methods in MATLAB" (SIAM, 2000)
3. Golub, G.H., Van Loan, C.F. "Matrix Computations" (Johns Hopkins, 2013)
4. Nocedal, J., Wright, S.J. "Numerical Optimization" (Springer, 2006)
5. LeVeque, R.J. "Finite Difference Methods for ODEs and PDEs" (SIAM, 2007)
---
*Copyright © 2025 Meshal Alawein — All rights reserved.*
*University of California, Berkeley*