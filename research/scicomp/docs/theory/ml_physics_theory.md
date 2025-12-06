# Machine Learning Physics Theory Reference
**SciComp - ML Physics Foundation**
*Author: Meshal Alawein (meshal@berkeley.edu)*
*Institution: University of California, Berkeley*
*Created: 2025*
---
## Table of Contents
1. [Physics-Informed Neural Networks](#physics-informed-neural-networks)
2. [Neural Operators](#neural-operators)
3. [Deep Learning Theory](#deep-learning-theory)
4. [Optimization in Physics](#optimization-in-physics)
5. [Uncertainty Quantification](#uncertainty-quantification)
6. [Multi-Fidelity Modeling](#multi-fidelity-modeling)
7. [Scientific Machine Learning](#scientific-machine-learning)
8. [Computational Implementation](#computational-implementation)
---
## Physics-Informed Neural Networks
### Theoretical Foundation
Physics-Informed Neural Networks (PINNs) integrate physical laws directly into the learning process by incorporating differential equations as soft constraints in the loss function.
**Core Principle**: Minimize a composite loss function:
```
L = L_data + λ_physics * L_physics + λ_boundary * L_boundary
```
where:
- L_data: Data fitting loss
- L_physics: Physics constraint violation
- L_boundary: Boundary condition satisfaction
### Universal Approximation Theorem
**Theorem**: A feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of ℝⁿ to arbitrary accuracy.
**Mathematical Formulation**:
For any continuous function f: [0,1]ⁿ → ℝ and ε > 0, there exists a neural network F_N such that:
```
sup_{x∈[0,1]ⁿ} |f(x) - F_N(x)| < ε
```
### PINN Architecture
**Network Approximation**:
```
u_θ(x,t) = NN_θ(x,t)
```
**Physics Residual**:
```
r(x,t) = ∂u_θ/∂t + N[u_θ] - f(x,t)
```
where N[·] is the differential operator and f(x,t) is the source term.
**Automatic Differentiation**: Enables exact computation of derivatives through computational graphs.
### Loss Function Components
**Physics Loss**:
```
L_physics = 1/N_p Σᵢ |r(xᵢ,tᵢ)|²
```
**Boundary Loss**:
```
L_boundary = 1/N_b Σⱼ |B[u_θ](xⱼ,tⱼ) - g(xⱼ,tⱼ)|²
```
**Initial Condition Loss**:
```
L_initial = 1/N_0 Σₖ |u_θ(xₖ,0) - u₀(xₖ)|²
```
### Schrödinger PINN
**Time-Dependent Schrödinger Equation**:
```
iℏ ∂ψ/∂t = -ℏ²/2m ∇²ψ + V(x)ψ
```
**Complex-Valued Networks**: ψ_θ(x,t) = u_θ(x,t) + iv_θ(x,t)
**Physics Residual**:
```
r = iℏ ∂ψ_θ/∂t + ℏ²/2m ∇²ψ_θ - V(x)ψ_θ
```
**Conservation Laws**:
- Probability: ∫|ψ|²dx = 1
- Energy: ⟨ψ|Ĥ|ψ⟩ = constant
### Navier-Stokes PINN
**Incompressible Flow Equations**:
```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
∇·u = 0
```
**Velocity-Pressure Formulation**:
```
u_θ(x,y,t) = [u_x(x,y,t), u_y(x,y,t)]
p_θ(x,y,t) = p(x,y,t)
```
**Reynolds Number**: Re = UL/ν (characteristic velocity × length / kinematic viscosity)
**Physics Residuals**:
```
r_momentum = ∂u_θ/∂t + (u_θ·∇)u_θ + ∇p_θ/ρ - ν∇²u_θ - f
r_continuity = ∇·u_θ
```
### Elasticity PINN
**Linear Elasticity Equations**:
```
∇·σ + b = 0 (equilibrium)
σ = C:ε (constitutive)
ε = ½(∇u + (∇u)ᵀ) (kinematic)
```
**Displacement Field**: u_θ(x,y) = [u_x(x,y), u_y(x,y)]
**Stress-Strain Relations** (plane stress):
```
[σₓₓ]   E/(1-ν²) [1  ν  0    ] [εₓₓ]
[σᵧᵧ] = --------- [ν  1  0    ] [εᵧᵧ]
[σₓᵧ]             [0  0  (1-ν)/2] [γₓᵧ]
```
### Training Strategies
**Adaptive Weights**: Dynamically balance loss components
```
λᵢ(epoch) = λᵢ⁽⁰⁾ * (Lᵢ⁽⁰⁾/Lᵢ(epoch))^α
```
**Curriculum Learning**: Gradually increase problem complexity
**Transfer Learning**: Pre-train on simpler problems, fine-tune on complex ones
**Multi-Task Learning**: Simultaneously learn multiple related PDEs
---
## Neural Operators
### Operator Learning Theory
**Operator**: Mapping between function spaces G: U → V
- U: Input function space
- V: Output function space
**Goal**: Learn the operator G from input-output pairs {uᵢ, G(uᵢ)}
### Fourier Neural Operator (FNO)
**Architecture**: Combines local and global transformations
**Fourier Transform**: F[u](k) = ∫ u(x)e^(-2πikx)dx
**Fourier Layer**:
```
(K(u))(x) = F⁻¹(R_θ · F[u])
```
where R_θ is a learnable weight matrix in Fourier space.
**Complete FNO Layer**:
```
v_{l+1} = σ(W_{l+1}v_l + K_l(v_l))
```
**Key Properties**:
- Resolution invariant
- Efficient in Fourier domain
- Captures global dependencies
**Approximation Theory**: FNO can approximate any nonlinear operator with sufficient width and depth.
### DeepONet Architecture
**Branch-Trunk Decomposition**:
```
G(u)(y) ≈ Σᵢ bᵢ(u) · tᵢ(y)
```
**Branch Network**: B(u) = [b₁(u), b₂(u), ..., bₚ(u)]
- Input: Function u evaluated at sensor points
- Output: p-dimensional basis coefficients
**Trunk Network**: T(y) = [t₁(y), t₂(y), ..., tₚ(y)]
- Input: Query coordinates y
- Output: p-dimensional basis functions
**Final Output**:
```
G_θ(u)(y) = B_θ(u) · T_θ(y) + b₀
```
**Universal Approximation**: DeepONet can approximate any nonlinear operator to arbitrary accuracy.
### Operator Approximation Theory
**Rademacher Complexity**: Bounds generalization error for operator learning:
```
R_n(F) = E_σ[sup_{G∈F} 1/n Σᵢ σᵢG(uᵢ)(yᵢ)]
```
**Approximation Error**: How well the architecture can represent the true operator
**Generalization Error**: How well the trained model performs on unseen data
---
## Deep Learning Theory
### Optimization Landscape
**Loss Surface**: Non-convex optimization landscape with multiple local minima
**Gradient Descent Dynamics**:
```
θ_{k+1} = θ_k - η∇L(θ_k)
```
**Adam Optimizer**: Adaptive moment estimation
```
m_k = β₁m_{k-1} + (1-β₁)∇L(θ_k)
m̂_k = m_k/(1-β₁^k)
v_k = β₂v_{k-1} + (1-β₂)(∇L(θ_k))²
v̂_k = v_k/(1-β₂^k)
θ_{k+1} = θ_k - η m̂_k/(√v̂_k + ε)
```
### Generalization Theory
**PAC-Bayes Bounds**: Relate generalization to network complexity
**Lottery Ticket Hypothesis**: Sparse subnetworks exist that can be trained in isolation
**Double Descent**: Test error decreases, increases, then decreases again with model complexity
### Regularization Techniques
**Weight Decay**: L₂ penalty on parameters
```
L_total = L_data + λ||θ||²
```
**Dropout**: Randomly set neurons to zero during training
**Batch Normalization**: Normalize layer inputs
```
BN(x) = γ((x-μ)/σ) + β
```
**Early Stopping**: Stop training when validation error starts increasing
---
## Optimization in Physics
### Physics-Constrained Optimization
**Constrained Problem**:
```
minimize   f(x)
subject to g(x) = 0 (physics constraints)
           h(x) ≤ 0 (inequality constraints)
```
**Lagrangian Formulation**:
```
L(x,λ,μ) = f(x) + λᵀg(x) + μᵀh(x)
```
### Variational Methods
**Principle of Least Action**: Physical systems follow paths that minimize action
```
S = ∫L(q,q̇,t)dt
```
**Euler-Lagrange Equations**:
```
d/dt(∂L/∂q̇) - ∂L/∂q = 0
```
### Inverse Problems
**Parameter Estimation**: Given data D, find parameters θ such that:
```
F(θ) ≈ D
```
**Regularized Least Squares**:
```
θ* = argmin_θ ||F(θ) - D||² + λR(θ)
```
**Tikhonov Regularization**: R(θ) = ||θ||²
**Total Variation Regularization**: R(θ) = ||∇θ||₁
---
## Uncertainty Quantification
### Types of Uncertainty
**Epistemic Uncertainty**: Model uncertainty (reducible with more data)
**Aleatoric Uncertainty**: Data uncertainty (irreducible noise)
### Bayesian Neural Networks
**Posterior Distribution**: p(θ|D) ∝ p(D|θ)p(θ)
**Variational Inference**: Approximate p(θ|D) with variational distribution q_φ(θ)
**ELBO Objective**:
```
L_ELBO = E_q[log p(D|θ)] - KL(q_φ(θ)||p(θ))
```
### Monte Carlo Dropout
**Training**: Apply dropout with probability p
**Inference**: Apply dropout T times, compute mean and variance:
```
μ(x) = 1/T Σₜ f_θₜ(x)
σ²(x) = 1/T Σₜ (f_θₜ(x) - μ(x))²
```
### Ensemble Methods
**Deep Ensembles**: Train multiple models with different initializations
```
μ_ensemble = 1/M Σₘ μₘ
σ²_ensemble = 1/M Σₘ (σ²ₘ + μₘ²) - μ²_ensemble
```
---
## Multi-Fidelity Modeling
### Hierarchical Modeling
**High-Fidelity**: Expensive, accurate simulations
**Low-Fidelity**: Cheap, approximate simulations
**Multi-Fidelity Approach**: Combine information from multiple fidelity levels
### Co-Kriging
**Multi-Fidelity Gaussian Process**:
```
[y_l]   ~ N([μ_l], [K_ll  K_lh])
[y_h]      ([μ_h]  [K_hl  K_hh])
```
**Prediction**: Use low-fidelity data to improve high-fidelity predictions
### Bi-Fidelity Networks
**Architecture**:
- Low-fidelity network: f_l(x)
- High-fidelity network: f_h(x) = f_l(x) + Δf(x)
**Training Strategy**:
1. Train f_l on abundant low-fidelity data
2. Train Δf on sparse high-fidelity data
---
## Scientific Machine Learning
### SciML Paradigms
**Physics-Informed**: Embed physics in ML models
**Physics-Constrained**: Enforce physics as hard constraints
**Physics-Guided**: Use physics to guide learning process
### Neural ODEs
**Continuous Dynamics**: Model as ODE
```
dh/dt = f_θ(h(t), t)
```
**Adjoint Method**: Efficient backpropagation through ODE solver
### Graph Neural Networks for Physics
**Graph Representation**: Particles/atoms as nodes, interactions as edges
**Message Passing**:
```
m_ij = f_e(h_i, h_j, e_ij)
h_i' = f_v(h_i, Σⱼ m_ij)
```
### Symmetry-Aware Networks
**Equivariance**: f(gx) = gf(x) for group action g
**Invariance**: f(gx) = f(x)
**E(3)-Equivariant Networks**: Preserve 3D rotation and translation symmetries
---
## Computational Implementation
### Automatic Differentiation
**Forward Mode**: Compute derivatives along with function values
**Reverse Mode**: Backpropagate derivatives from output to input
**Computational Graph**: Represent function as directed acyclic graph
### Parallel Computing
**Data Parallelism**: Distribute batches across devices
**Model Parallelism**: Distribute model components across devices
**Pipeline Parallelism**: Overlap forward and backward passes
### Memory Optimization
**Gradient Checkpointing**: Trade computation for memory
**Mixed Precision**: Use lower precision for some operations
### Software Frameworks
**TensorFlow**: Eager execution and graph mode
**PyTorch**: Dynamic computational graphs
**JAX**: NumPy-compatible with automatic differentiation
---
## Berkeley SciComp ML Physics
### Framework Design
**Modular Architecture**:
- `pinns/`: Physics-informed neural networks
- `neural_operators/`: Operator learning methods
- `uncertainty/`: Uncertainty quantification tools
- `optimization/`: Physics-constrained optimization
### Key Features
**Multi-Physics Support**: Quantum mechanics, fluid dynamics, elasticity
**Uncertainty Quantification**: Bayesian methods and ensembles
**High-Performance Computing**: GPU acceleration and distributed training
**Theoretical Rigor**: Mathematically sound implementations
### Research Applications
**Quantum Simulations**: Schrödinger equation solving
**Fluid Dynamics**: Turbulence modeling and flow prediction
**Materials Science**: Property prediction and design
**Climate Modeling**: Weather and climate prediction
---
## References
1. Karniadakis, G.E., et al. "Physics-informed machine learning" Nature Reviews Physics (2021)
2. Chen, T., Chen, H. "Universal approximation to nonlinear operators" arXiv preprint (2019)
3. Lu, L., Jin, P., Karniadakis, G.E. "DeepONet: Learning nonlinear operators" Nature Machine Intelligence (2021)
4. Li, Z., et al. "Fourier Neural Operator for Parametric Partial Differential Equations" ICLR (2021)
5. Raissi, M., Perdikaris, P., Karniadakis, G.E. "Physics-informed Neural Networks" Journal of Computational Physics (2019)
---
*Copyright © 2025 Meshal Alawein — All rights reserved.*
*University of California, Berkeley*