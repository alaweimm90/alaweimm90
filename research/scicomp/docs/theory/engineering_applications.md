# Engineering Applications Theory Reference
**SciComp - Engineering Foundation**
*Author: Meshal Alawein (meshal@berkeley.edu)*
*Institution: University of California, Berkeley*
*Created: 2025*
---
## Table of Contents
1. [Heat Transfer Theory](#heat-transfer-theory)
2. [Fluid Mechanics](#fluid-mechanics)
3. [Structural Mechanics](#structural-mechanics)
4. [Electromagnetic Theory](#electromagnetic-theory)
5. [Control Systems](#control-systems)
6. [Signal Processing](#signal-processing)
7. [Materials Science](#materials-science)
8. [System Engineering](#system-engineering)
---
## Heat Transfer Theory
### Fundamental Modes
**Conduction**: Heat transfer through molecular motion
**Convection**: Heat transfer by fluid motion
**Radiation**: Heat transfer by electromagnetic waves
### Fourier's Law of Heat Conduction
**One-Dimensional Form**:
```
q = -kA(dT/dx)
```
**General Form**:
```
q⃗ = -k∇T
```
where:
- q: Heat flux (W/m²)
- k: Thermal conductivity (W/m·K)
- T: Temperature (K)
- ∇T: Temperature gradient
### Heat Equation Derivation
**Energy Balance**: Rate of energy in - Rate of energy out = Rate of energy accumulation
**Differential Form**:
```
∂T/∂t = α∇²T + q'''/(ρcp)
```
where:
- α = k/(ρcp): Thermal diffusivity (m²/s)
- ρ: Density (kg/m³)
- cp: Specific heat (J/kg·K)
- q''': Heat generation per unit volume (W/m³)
### Boundary Conditions
**Dirichlet (Prescribed Temperature)**:
```
T(boundary) = Ts
```
**Neumann (Prescribed Heat Flux)**:
```
-k(∂T/∂n)|boundary = q''s
```
**Robin (Convective)**:
```
-k(∂T/∂n)|boundary = h(T|boundary - T∞)
```
**Radiation**:
```
-k(∂T/∂n)|boundary = εσ(T⁴|boundary - T⁴∞)
```
### Dimensional Analysis
**Fourier Number**: Fo = αt/L² (dimensionless time)
**Biot Number**: Bi = hL/k (ratio of internal to external thermal resistance)
**Peclet Number**: Pe = uL/α (ratio of advection to diffusion)
### Steady-State Solutions
**1D Plane Wall with Heat Generation**:
```
T(x) = T₀ + (q'''L²/2k)(1 - (x/L)²)
```
**Cylindrical Coordinates (Radial)**:
```
T(r) = C₁ln(r) + C₂ + q'''r²/(4k)
```
**Spherical Coordinates (Radial)**:
```
T(r) = C₁/r + C₂ + q'''r²/(6k)
```
### Transient Heat Transfer
**Lumped Capacitance Model** (Bi < 0.1):
```
T(t) - T∞ = (T₀ - T∞)exp(-hAs t/(ρVcp))
```
**Semi-Infinite Solid**:
```
T(x,t) - T∞ = (T₀ - T∞)erf(x/(2√(αt)))
```
### Convective Heat Transfer
**Newton's Law of Cooling**:
```
q'' = h(Ts - T∞)
```
**Nusselt Number**: Nu = hL/k (dimensionless heat transfer coefficient)
**Correlations**:
- **Forced Convection over Flat Plate**: Nu = 0.332Re^(1/2)Pr^(1/3)
- **Natural Convection Vertical Plate**: Nu = 0.59Ra^(1/4) (10⁴ < Ra < 10⁹)
- **Flow in Pipes**: Nu = 0.023Re^(0.8)Pr^n
where:
- Re: Reynolds number
- Pr: Prandtl number
- Ra: Rayleigh number
### Radiation Heat Transfer
**Stefan-Boltzmann Law**:
```
q = εσAT⁴
```
**Between Two Surfaces**:
```
q₁₂ = σA₁F₁₂(T₁⁴ - T₂⁴)/(1/ε₁ + (A₁/A₂)(1/ε₂ - 1))
```
**Radiation Exchange Networks**: Use electrical analogy with thermal resistances
---
## Fluid Mechanics
### Fundamental Equations
**Conservation of Mass (Continuity)**:
```
∂ρ/∂t + ∇·(ρu⃗) = 0
```
**Conservation of Momentum (Navier-Stokes)**:
```
ρ(∂u⃗/∂t + (u⃗·∇)u⃗) = -∇p + μ∇²u⃗ + ρg⃗
```
**Conservation of Energy**:
```
ρcp(∂T/∂t + u⃗·∇T) = k∇²T + Φ + q'''
```
### Dimensionless Parameters
**Reynolds Number**: Re = ρuL/μ
- Ratio of inertial to viscous forces
- Determines flow regime (laminar vs. turbulent)
**Froude Number**: Fr = u/√(gL)
- Ratio of inertial to gravitational forces
- Important for free surface flows
**Mach Number**: Ma = u/a
- Ratio of flow speed to sound speed
- Determines compressibility effects
**Prandtl Number**: Pr = μcp/k = ν/α
- Ratio of momentum to thermal diffusivity
### Potential Flow Theory
**Assumptions**:
- Inviscid flow
- Irrotational flow: ∇ × u⃗ = 0
**Velocity Potential**: u⃗ = ∇φ
**Laplace Equation**: ∇²φ = 0
**Complex Potential**: F(z) = φ + iψ (φ: potential, ψ: stream function)
**Elementary Flows**:
- **Uniform Flow**: F(z) = Uz
- **Source/Sink**: F(z) = (m/2π)ln(z)
- **Vortex**: F(z) = -(iΓ/2π)ln(z)
- **Doublet**: F(z) = μ/z
### Boundary Layer Theory
**Boundary Layer Thickness**: δ ≈ √(νx/U)
**Displacement Thickness**: δ* = ∫₀^∞ (1 - u/U)dy
**Momentum Thickness**: θ = ∫₀^∞ (u/U)(1 - u/U)dy
**Blasius Solution** (laminar flat plate):
```
u/U = f'(η), η = y√(U/νx)
```
**Skin Friction Coefficient**: cf = 0.664/√(Rex)
### Turbulent Flow
**Reynolds Decomposition**: u = ū + u' (mean + fluctuation)
**Reynolds-Averaged Navier-Stokes (RANS)**:
```
ρ(∂ūᵢ/∂t + ūⱼ∂ūᵢ/∂xⱼ) = -∂p̄/∂xᵢ + ∂/∂xⱼ[μ(∂ūᵢ/∂xⱼ) - ρu'ᵢu'ⱼ]
```
**Turbulence Models**:
- **k-ε Model**: Two-equation model for turbulent kinetic energy and dissipation
- **k-ω Model**: Alternative two-equation model
- **Large Eddy Simulation (LES)**: Resolve large scales, model small scales
- **Direct Numerical Simulation (DNS)**: Resolve all scales
### Compressible Flow
**Speed of Sound**: a = √(γRT)
**Isentropic Relations**:
```
T₀/T = 1 + (γ-1)/2 Ma²
p₀/p = (1 + (γ-1)/2 Ma²)^(γ/(γ-1))
```
**Normal Shock Relations**:
```
Ma₂² = (Ma₁² + 2/(γ-1))/(2γMa₁²/(γ-1) - 1)
```
**Oblique Shock**: θ-β-Ma relationship for shock angle β and deflection angle θ
---
## Structural Mechanics
### Fundamental Principles
**Equilibrium**: ΣF = 0, ΣM = 0
**Compatibility**: Deformations must be consistent with geometry
**Constitutive Relations**: Stress-strain relationships
### Stress and Strain
**Stress Tensor**:
```
σ = [σₓₓ  τₓᵧ  τₓᵤ]
    [τᵧₓ  σᵧᵧ  τᵧᵤ]
    [τᵤₓ  τᵤᵧ  σᵤᵤ]
```
**Strain Tensor**:
```
ε = [εₓₓ  γₓᵧ/2  γₓᵤ/2]
    [γᵧₓ/2  εᵧᵧ  γᵧᵤ/2]
    [γᵤₓ/2  γᵤᵧ/2  εᵤᵤ]
```
**Engineering Strain**: γ = 2ε (shear strain)
### Linear Elasticity
**Hooke's Law (Isotropic)**:
```
σᵢⱼ = λδᵢⱼεₖₖ + 2μεᵢⱼ
```
**Material Constants**:
- λ, μ: Lamé parameters
- E: Young's modulus
- ν: Poisson's ratio
- G: Shear modulus
**Relationships**:
```
λ = νE/((1+ν)(1-2ν))
μ = G = E/(2(1+ν))
```
### Plane Stress and Plane Strain
**Plane Stress** (thin plates): σᵤᵤ = τₓᵤ = τᵧᵤ = 0
**Plane Strain** (long cylinders): εᵤᵤ = γₓᵤ = γᵧᵤ = 0
**Stress-Strain Relations (Plane Stress)**:
```
[σₓₓ]   E/(1-ν²) [1  ν  0    ] [εₓₓ]
[σᵧᵧ] = -------- [ν  1  0    ] [εᵧᵧ]
[τₓᵧ]            [0  0  (1-ν)/2] [γₓᵧ]
```
### Beam Theory
**Euler-Bernoulli Beam Theory**:
- Plane sections remain plane
- Neglect shear deformation
**Beam Equation**:
```
EI(d⁴w/dx⁴) = q(x)
```
**Moment-Curvature Relation**:
```
M = EI(d²w/dx²)
```
**Common Solutions**:
- **Simply supported beam with uniform load**: w = (qx/24EI)(L³ - 2Lx² + x³)
- **Cantilever beam with end load**: w = (Px²/6EI)(3L - x)
**Timoshenko Beam Theory**: Includes shear deformation effects
### Plates and Shells
**Kirchhoff Plate Theory**:
```
D∇⁴w = q
```
where D = Et³/(12(1-ν²)) is the flexural rigidity.
**Von Karman Equations**: Large deflection plate theory
**Shell Theory**: Combines membrane and bending effects
- Membrane stresses
- Bending moments
- Geometric nonlinearity
### Failure Theories
**Maximum Stress Theory**: σmax < σallow
**Maximum Shear Stress Theory (Tresca)**:
```
τmax = (σ₁ - σ₃)/2 < σy/2
```
**Distortion Energy Theory (von Mises)**:
```
σe = √((σ₁ - σ₂)² + (σ₂ - σ₃)² + (σ₃ - σ₁)²)/√2 < σy
```
**Mohr-Coulomb Theory**: For brittle materials
### Dynamics
**Equation of Motion**:
```
ρ∂²u/∂t² = ∇·σ + ρb
```
**Modal Analysis**: Find natural frequencies and mode shapes
```
(K - ω²M)φ = 0
```
**Dynamic Response**:
- **Undamped**: ü + ω²u = F(t)/m
- **Damped**: ü + 2ζωu̇ + ω²u = F(t)/m
---
## Electromagnetic Theory
### Maxwell's Equations
**Differential Form**:
```
∇·D = ρ          (Gauss's law)
∇·B = 0          (No magnetic monopoles)
∇×E = -∂B/∂t     (Faraday's law)
∇×H = J + ∂D/∂t  (Ampère-Maxwell law)
```
**Integral Form**:
```
∮D·da = Q_enc
∮B·da = 0
∮E·dl = -∫(∂B/∂t)·da
∮H·dl = I_enc + ∫(∂D/∂t)·da
```
### Constitutive Relations
**Linear, Isotropic Media**:
```
D = εE
B = μH
J = σE
```
**Material Parameters**:
- ε: Permittivity
- μ: Permeability
- σ: Conductivity
### Wave Equations
**In Free Space**:
```
∇²E - μ₀ε₀∂²E/∂t² = 0
∇²B - μ₀ε₀∂²B/∂t² = 0
```
**Speed of Light**: c = 1/√(μ₀ε₀)
**Plane Wave Solution**:
```
E = E₀exp(i(k·r - ωt))
```
### Electromagnetic Waves
**Dispersion Relation**: ω² = c²k²
**Polarization**:
- Linear polarization
- Circular polarization
- Elliptical polarization
**Poynting Vector**: S = E × H (energy flow density)
**Wave Impedance**: Z = √(μ/ε)
### Transmission Lines
**Telegrapher's Equations**:
```
∂V/∂z = -L∂I/∂t - RI
∂I/∂z = -C∂V/∂t - GV
```
**Characteristic Impedance**: Z₀ = √(L/C)
**Propagation Constant**: γ = √((R+iωL)(G+iωC))
**Reflection Coefficient**: Γ = (ZL - Z₀)/(ZL + Z₀)
### Antenna Theory
**Radiation Pattern**: Directional distribution of radiated power
**Directivity**: D = 4π(maximum radiation intensity)/(average radiation intensity)
**Antenna Gain**: G = ηD (includes efficiency)
**Friis Transmission Equation**:
```
Pr/Pt = GtGr(λ/4πR)²
```
---
## Control Systems
### System Representation
**Transfer Function**: G(s) = Y(s)/U(s)
**State-Space Model**:
```
ẋ = Ax + Bu
y = Cx + Du
```
**Block Diagrams**: Graphical representation of system interconnections
### System Properties
**Stability**:
- **BIBO Stability**: Bounded input produces bounded output
- **Asymptotic Stability**: All poles in left half-plane
**Controllability**: Ability to drive system to any state
**Observability**: Ability to determine state from output
### Time Response
**First-Order System**: G(s) = K/(τs + 1)
- **Time Constant**: τ
- **Step Response**: y(t) = K(1 - e^(-t/τ))
**Second-Order System**: G(s) = ωn²/(s² + 2ζωns + ωn²)
- **Natural Frequency**: ωn
- **Damping Ratio**: ζ
- **Overshoot**: Mp = e^(-ζπ/√(1-ζ²))
### Frequency Response
**Bode Plots**: Magnitude and phase vs. frequency
**Nyquist Plot**: Polar plot of frequency response
**Stability Margins**:
- **Gain Margin**: GM = 1/|G(iωc)|
- **Phase Margin**: PM = 180° + ∠G(iωg)
### Controller Design
**PID Controller**: Gc(s) = Kp + Ki/s + Kds
**Root Locus**: Plot of closed-loop poles as parameter varies
**Frequency Domain Design**:
- Lead compensator: improves transient response
- Lag compensator: improves steady-state accuracy
**Modern Control**:
- **LQR**: Linear Quadratic Regulator
- **LQG**: Linear Quadratic Gaussian
- **H∞ Control**: Robust control design
---
## Signal Processing
### Fourier Analysis
**Fourier Series** (periodic signals):
```
x(t) = a₀/2 + Σn[an cos(nω₀t) + bn sin(nω₀t)]
```
**Fourier Transform** (aperiodic signals):
```
X(ω) = ∫_{-∞}^∞ x(t)e^{-iωt}dt
```
**Properties**:
- Linearity
- Time shifting
- Frequency shifting
- Convolution theorem: x(t)*y(t) ↔ X(ω)Y(ω)
### Discrete-Time Signals
**Discrete Fourier Transform (DFT)**:
```
X[k] = Σ_{n=0}^{N-1} x[n]e^{-i2πkn/N}
```
**Fast Fourier Transform (FFT)**: Efficient DFT computation
**Z-Transform**:
```
X(z) = Σ_{n=-∞}^∞ x[n]z^{-n}
```
### Filtering
**Ideal Filters**:
- Low-pass: |H(ω)| = 1 for |ω| < ωc, 0 otherwise
- High-pass: |H(ω)| = 0 for |ω| < ωc, 1 otherwise
- Band-pass: Pass specific frequency range
**Filter Design**:
- **Butterworth**: Maximally flat passband
- **Chebyshev**: Equiripple in passband or stopband
- **Elliptic**: Equiripple in both bands
**Digital Filters**:
- **FIR**: Finite Impulse Response
- **IIR**: Infinite Impulse Response
### Sampling Theory
**Nyquist-Shannon Theorem**: Sampling rate must be > 2 × highest frequency
**Aliasing**: High frequencies fold back into low frequencies
**Anti-Aliasing Filter**: Prevent aliasing with analog low-pass filter
### Spectral Analysis
**Power Spectral Density**: Sxx(ω) = |X(ω)|²
**Windowing**: Reduce spectral leakage
- Rectangular, Hamming, Hanning, Kaiser windows
**Correlation**: Measure of similarity between signals
- **Autocorrelation**: Rxx(τ) = E[x(t)x(t+τ)]
- **Cross-correlation**: Rxy(τ) = E[x(t)y(t+τ)]
---
## Materials Science
### Crystal Structure
**Lattice Types**:
- Simple cubic
- Body-centered cubic (BCC)
- Face-centered cubic (FCC)
- Hexagonal close-packed (HCP)
**Miller Indices**: Crystallographic direction and plane notation
**Defects**:
- **Point defects**: Vacancies, interstitials, substitutions
- **Line defects**: Dislocations
- **Planar defects**: Grain boundaries, stacking faults
### Mechanical Properties
**Stress-Strain Behavior**:
- **Elastic region**: σ = Eε
- **Yield strength**: σy
- **Ultimate tensile strength**: σu
- **Fracture strength**: σf
**Hardness**: Resistance to permanent deformation
- Brinell, Rockwell, Vickers scales
**Fracture Mechanics**:
- **Stress intensity factor**: K = σ√(πa)
- **Fracture toughness**: KIC
- **Paris law**: da/dN = C(ΔK)^m
### Phase Diagrams
**Binary Systems**:
- **Eutectic**: Lowest melting point composition
- **Eutectoid**: Solid-state decomposition
- **Peritectic**: Liquid + solid → new solid
**Lever Rule**: Determine phase fractions
**TTT Diagrams**: Time-Temperature-Transformation
### Diffusion
**Fick's Laws**:
- **First law**: J = -D(∂C/∂x)
- **Second law**: ∂C/∂t = D(∂²C/∂x²)
**Arrhenius Equation**: D = D₀exp(-Q/RT)
**Solutions**:
- **Semi-infinite solid**: C(x,t) = C₀erfc(x/(2√(Dt)))
- **Thin film**: C(x,t) = (M/√(πDt))exp(-x²/(4Dt))
### Electronic Properties
**Band Theory**: Energy bands separated by band gaps
**Conductivity**: σ = nqμ (carrier concentration × charge × mobility)
**Semiconductors**:
- **Intrinsic**: Pure semiconductor
- **Extrinsic**: Doped semiconductor
- **p-n junction**: Foundation of diodes and transistors
---
## System Engineering
### Systems Thinking
**System**: Interconnected components working toward common purpose
**Properties**:
- **Emergence**: System properties not evident in components
- **Hierarchy**: Systems within systems
- **Boundaries**: Interface with environment
### Reliability Engineering
**Failure Rate**: λ(t) = f(t)/R(t)
**Reliability Function**: R(t) = P(T > t)
**MTBF**: Mean Time Between Failures = 1/λ
**Common Distributions**:
- **Exponential**: Constant failure rate
- **Weibull**: Variable failure rate
- **Normal**: Wear-out failures
**System Reliability**:
- **Series**: Rs = ∏Ri
- **Parallel**: Rs = 1 - ∏(1-Ri)
### Optimization
**Design Variables**: Parameters to be optimized
**Objective Function**: Performance measure to optimize
**Constraints**: Limits on design variables or system behavior
**Multi-Objective Optimization**: Pareto optimality
**Algorithms**:
- **Gradient-based**: Efficient for smooth problems
- **Evolutionary**: Handle discrete variables and multiple objectives
- **Surrogate models**: Reduce computational cost
### Risk Analysis
**Risk**: Probability × Consequence
**Risk Assessment Process**:
1. Hazard identification
2. Risk analysis
3. Risk evaluation
4. Risk treatment
**Monte Carlo Simulation**: Probabilistic risk analysis
**Sensitivity Analysis**: Identify critical parameters
---
## Berkeley SciComp Engineering Applications
### Framework Integration
**Multi-Physics Coupling**:
- Thermal-structural analysis
- Fluid-structure interaction
- Electromagnetic-thermal coupling
**Scale Bridging**:
- Molecular to continuum
- Component to system level
- Multi-fidelity modeling
### Computational Tools
**Finite Element Analysis**: MATLAB implementations
**Computational Fluid Dynamics**: Python/MATLAB solvers
**Control System Design**: MATLAB Control Toolbox integration
**Signal Processing**: Advanced filtering and analysis
### Validation and Verification
**Code Verification**: Mathematical correctness
**Solution Verification**: Numerical accuracy
**Model Validation**: Physical accuracy
**Benchmark Problems**: Standardized test cases
**Experimental Validation**: Comparison with measurements
---
## References
1. Incropera, F.P., et al. "Fundamentals of Heat and Mass Transfer" (Wiley, 2017)
2. White, F.M. "Fluid Mechanics" (McGraw-Hill, 2016)
3. Hibbeler, R.C. "Mechanics of Materials" (Pearson, 2017)
4. Griffiths, D.J. "Introduction to Electrodynamics" (Cambridge, 2017)
5. Franklin, G.F., et al. "Feedback Control of Dynamic Systems" (Pearson, 2019)
---
*Copyright © 2025 Meshal Alawein — All rights reserved.*
*University of California, Berkeley*