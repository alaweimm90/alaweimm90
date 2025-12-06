# Quantum Mechanics Theory Reference
**SciComp - Theoretical Foundation**
*Author: Meshal Alawein (meshal@berkeley.edu)*
*Institution: University of California, Berkeley*
*Created: 2025*
---
## Table of Contents
1. [Fundamental Principles](#fundamental-principles)
2. [Mathematical Formalism](#mathematical-formalism)
3. [Quantum Harmonic Oscillator](#quantum-harmonic-oscillator)
4. [Operator Algebra](#operator-algebra)
5. [Time Evolution](#time-evolution)
6. [Measurement Theory](#measurement-theory)
7. [Approximation Methods](#approximation-methods)
8. [Computational Implementation](#computational-implementation)
---
## Fundamental Principles
### Postulates of Quantum Mechanics
Quantum mechanics is built upon fundamental postulates that govern the behavior of quantum systems:
**Postulate I**: The state of a quantum system is completely described by a wavefunction Ψ(x,t) in the position representation, which contains all possible information about the system.
**Postulate II**: Physical observables correspond to Hermitian operators. For example:
- Position: x̂ = x
- Momentum: p̂ = -iℏ∇
- Energy: Ĥ (Hamiltonian operator)
**Postulate III**: The only possible results of measuring an observable  are the eigenvalues of the corresponding operator:
```
Â|ψₙ⟩ = aₙ|ψₙ⟩
```
**Postulate IV**: The probability of measuring eigenvalue aₙ is |⟨ψₙ|ψ⟩|², where |ψ⟩ is the system state before measurement.
**Postulate V**: After measurement yielding result aₙ, the system collapses to the corresponding eigenstate |ψₙ⟩.
**Postulate VI**: Time evolution is governed by the Schrödinger equation:
```
iℏ ∂|ψ⟩/∂t = Ĥ|ψ⟩
```
### Wave-Particle Duality
The fundamental duality of quantum objects manifests through:
**De Broglie Relations**:
- λ = h/p (wavelength-momentum relation)
- E = ℏω (energy-frequency relation)
**Uncertainty Principle**:
```
ΔxΔp ≥ ℏ/2
```
This fundamental limit arises from the non-commutativity of position and momentum operators:
```
[x̂, p̂] = iℏ
```
---
## Mathematical Formalism
### Hilbert Space Structure
Quantum states exist in a complex Hilbert space ℋ with:
**Inner Product**: ⟨ψ|φ⟩ = ∫ ψ*(x)φ(x)dx
**Norm**: ||ψ|| = √⟨ψ|ψ⟩
**Normalization Condition**: ⟨ψ|ψ⟩ = 1
### Dirac Notation
**Ket**: |ψ⟩ represents a quantum state
**Bra**: ⟨ψ| represents the complex conjugate
**Bracket**: ⟨φ|ψ⟩ represents the inner product
**Outer Product**: |ψ⟩⟨φ| represents an operator
### Operators and Commutators
**Hermitian Operators**: Â† = Â (observables)
**Unitary Operators**: Û†Û = ÛÛ† = Î (time evolution)
**Commutator**: [Â, B̂] = ÂB̂ - B̂Â
**Canonical Commutation Relations**:
```
[x̂ᵢ, p̂ⱼ] = iℏδᵢⱼ
[x̂ᵢ, x̂ⱼ] = 0
[p̂ᵢ, p̂ⱼ] = 0
```
---
## Quantum Harmonic Oscillator
### Classical Harmonic Oscillator
The classical system with potential V(x) = ½mω²x² has:
- Total energy: E = ½mẋ² + ½mω²x²
- Angular frequency: ω = √(k/m)
### Quantum Mechanical Treatment
**Hamiltonian**:
```
Ĥ = p̂²/(2m) + ½mω²x̂²
```
**Time-Independent Schrödinger Equation**:
```
Ĥψₙ(x) = Eₙψₙ(x)
```
### Ladder Operators
**Creation Operator**:
```
â† = √(mω/2ℏ)(x̂ - ip̂/(mω))
```
**Annihilation Operator**:
```
â = √(mω/2ℏ)(x̂ + ip̂/(mω))
```
**Commutation Relation**:
```
[â, â†] = 1
```
**Number Operator**:
```
n̂ = â†â
```
### Energy Eigenvalues and Eigenstates
**Energy Levels**:
```
Eₙ = ℏω(n + ½), n = 0, 1, 2, ...
```
**Ground State Energy**: E₀ = ℏω/2 (zero-point energy)
**Ladder Operator Actions**:
```
â†|n⟩ = √(n+1)|n+1⟩
â|n⟩ = √n|n-1⟩
n̂|n⟩ = n|n⟩
```
### Wavefunctions
**Ground State**:
```
ψ₀(x) = (mω/πℏ)^(1/4) exp(-mωx²/2ℏ)
```
**General State**:
```
ψₙ(x) = (mω/πℏ)^(1/4) * 1/√(2ⁿn!) * Hₙ(√(mω/ℏ)x) * exp(-mωx²/2ℏ)
```
where Hₙ(ξ) are Hermite polynomials.
### Properties
**Position Uncertainty**:
```
⟨Δx²⟩ₙ = (ℏ/2mω)(2n + 1)
```
**Momentum Uncertainty**:
```
⟨Δp²⟩ₙ = (mℏω/2)(2n + 1)
```
**Uncertainty Product**:
```
ΔxₙΔpₙ = ℏ(n + ½) ≥ ℏ/2
```
---
## Operator Algebra
### Pauli Matrices
**Pauli-X**: σₓ = [0 1; 1 0]
**Pauli-Y**: σᵧ = [0 -i; i 0]
**Pauli-Z**: σᵤ = [1 0; 0 -1]
**Anticommutation Relations**:
```
{σᵢ, σⱼ} = 2δᵢⱼI
```
**Commutation Relations**:
```
[σᵢ, σⱼ] = 2iεᵢⱼₖσₖ
```
### Angular Momentum
**Total Angular Momentum**:
```
Ĵ² = Ĵₓ² + Ĵᵧ² + Ĵᵤ²
```
**Commutation Relations**:
```
[Ĵᵢ, Ĵⱼ] = iℏεᵢⱼₖĴₖ
[Ĵ², Ĵᵢ] = 0
```
**Eigenvalue Equations**:
```
Ĵ²|j,m⟩ = ℏ²j(j+1)|j,m⟩
Ĵᵤ|j,m⟩ = ℏm|j,m⟩
```
where j = 0, ½, 1, 3/2, ... and m = -j, -j+1, ..., j-1, j.
### Ladder Operators for Angular Momentum
**Raising Operator**: Ĵ₊ = Ĵₓ + iĴᵧ
**Lowering Operator**: Ĵ₋ = Ĵₓ - iĴᵧ
**Actions**:
```
Ĵ₊|j,m⟩ = ℏ√((j-m)(j+m+1))|j,m+1⟩
Ĵ₋|j,m⟩ = ℏ√((j+m)(j-m+1))|j,m-1⟩
```
---
## Time Evolution
### Schrödinger Picture
**Time-Dependent Schrödinger Equation**:
```
iℏ ∂|ψ(t)⟩/∂t = Ĥ|ψ(t)⟩
```
**Formal Solution**:
```
|ψ(t)⟩ = Û(t)|ψ(0)⟩
```
**Time Evolution Operator**:
```
Û(t) = exp(-iĤt/ℏ)
```
### Heisenberg Picture
**Heisenberg Operators**:
```
ÂH(t) = Û†(t)ÂSÛ(t)
```
**Heisenberg Equation of Motion**:
```
iℏ dÂH/dt = [ÂH, Ĥ]
```
### Interaction Picture
Used for time-dependent Hamiltonians Ĥ = Ĥ₀ + V̂(t):
**States**: |ψI(t)⟩ = exp(iĤ₀t/ℏ)|ψS(t)⟩
**Operators**: ÂI(t) = exp(iĤ₀t/ℏ)ÂSexp(-iĤ₀t/ℏ)
---
## Measurement Theory
### Projection Operators
For observable  with eigenvalues {aₙ} and eigenstates {|ψₙ⟩}:
**Projection Operator**: P̂ₙ = |ψₙ⟩⟨ψₙ|
**Completeness**: Σₙ P̂ₙ = Î
**Measurement Probability**: P(aₙ) = ⟨ψ|P̂ₙ|ψ⟩ = |⟨ψₙ|ψ⟩|²
### Expectation Values
**Expectation Value**: ⟨Â⟩ = ⟨ψ|Â|ψ⟩
**Variance**: (ΔA)² = ⟨Â²⟩ - ⟨Â⟩²
### Quantum Non-Demolition Measurements
Measurements that don't disturb future measurements of the same observable:
**Condition**: [Â, Ĥ] = 0
---
## Approximation Methods
### Perturbation Theory
For Ĥ = Ĥ₀ + λV̂:
**First-Order Energy Correction**:
```
E₍ₙ⁾⁽¹⁾ = ⟨ψₙ⁽⁰⁾|V̂|ψₙ⁽⁰⁾⟩
```
**Second-Order Energy Correction**:
```
E₍ₙ⁾⁽²⁾ = Σₘ≠ₙ |⟨ψₘ⁽⁰⁾|V̂|ψₙ⁽⁰⁾⟩|² / (Eₙ⁽⁰⁾ - Eₘ⁽⁰⁾)
```
**First-Order Wavefunction Correction**:
```
|ψₙ⁽¹⁾⟩ = Σₘ≠ₙ ⟨ψₘ⁽⁰⁾|V̂|ψₙ⁽⁰⁾⟩/(Eₙ⁽⁰⁾ - Eₘ⁽⁰⁾)|ψₘ⁽⁰⁾⟩
```
### Variational Method
**Variational Principle**: E₀ ≤ ⟨ψₜᵣᵢₐₗ|Ĥ|ψₜᵣᵢₐₗ⟩
Used to find approximate ground state energies by minimizing:
```
E[ψ] = ⟨ψ|Ĥ|ψ⟩/⟨ψ|ψ⟩
```
### WKB Approximation
For slowly varying potentials:
**WKB Wavefunction**:
```
ψ(x) ≈ A/√p(x) exp(i/ℏ ∫ p(x')dx')
```
where p(x) = √(2m[E - V(x)]).
**Bohr-Sommerfeld Quantization**:
```
∮ p dx = (n + ½)h
```
---
## Computational Implementation
### Numerical Methods
**Finite Difference**: Discretize derivatives on spatial/temporal grids
**Spectral Methods**: Expand wavefunctions in basis sets (plane waves, Hermite functions)
**Split-Operator Method**: Separate kinetic and potential energy evolution
**Runge-Kutta Methods**: High-order time integration schemes
### Matrix Representations
**Position Basis**: ⟨x|ψ⟩ = ψ(x)
**Momentum Basis**: ⟨p|ψ⟩ = ψ̃(p)
**Matrix Elements**:
```
⟨i|Â|j⟩ = ∫ ψᵢ*(x)Âψⱼ(x)dx
```
### Computational Challenges
**Dimensionality**: Hilbert space grows exponentially with particle number
**Decoherence**: Environment interaction destroys quantum coherence
**Measurement**: Quantum measurement problem in simulations
### Quantum Monte Carlo
**Path Integral Monte Carlo**: Sample quantum paths
**Variational Monte Carlo**: Optimize trial wavefunctions
**Diffusion Monte Carlo**: Project to ground state
---
## Berkeley SciComp Implementation
### Framework Architecture
The Berkeley SciComp framework implements these theoretical concepts through:
**Python Modules**:
- `quantum_physics/`: Core quantum mechanics implementations
- `quantum_computing/`: Quantum algorithms and circuits
- `ml_physics/`: Physics-informed machine learning
**MATLAB Implementations**:
- Engineering-focused numerical methods
- Visualization and analysis tools
- Performance-optimized calculations
**Mathematica Packages**:
- Symbolic computation capabilities
- Exact analytical solutions
- Advanced mathematical analysis
### Key Features
**Multi-Platform Support**: Seamless integration across Python, MATLAB, and Mathematica
**Theoretical Rigor**: Implementations validated against analytical solutions
**Educational Focus**: Clear documentation linking theory to implementation
**Research Applications**: Production-ready tools for quantum research
---
## References
1. Griffiths, D.J. & Schroeter, D.F. "Introduction to Quantum Mechanics" (3rd Edition)
2. Shankar, R. "Principles of Quantum Mechanics" (2nd Edition)
3. Nielsen, M.A. & Chuang, I.L. "Quantum Computation and Quantum Information"
4. Tannor, D.J. "Introduction to Quantum Mechanics: A Time-Dependent Perspective"
5. Messiah, A. "Quantum Mechanics" (Dover Publications)
---
*Copyright © 2025 Meshal Alawein — All rights reserved.*
*University of California, Berkeley*