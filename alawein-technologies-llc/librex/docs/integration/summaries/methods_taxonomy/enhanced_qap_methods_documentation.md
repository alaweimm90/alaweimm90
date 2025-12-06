
---
**Notation Standards**: See [NOTATION_STANDARDS.md](NOTATION_STANDARDS.md) for consistent mathematical notation across all Librex.QAP documentation.
---

Source: C:\Users\mesha\Pictures\random\notebooks\enhanced_qap_methods_documentation.md
Imported: 2025-11-17T14:08:47.200654

# Notebook Documentation

*Converted from: enhanced_qap_methods.ipynb*
*Date: 2025-11-17*

# Attractor Programming for QAP: Comprehensive Ablation Study

**Complete implementation with systematic ablation studies and benchmarking.**

This notebook implements our Attractor Programming framework for the Quadratic Assignment Problem (QAP) with:
- All method components
- Systematic ablation studies
- Comprehensive metrics tracking
- Complete figure generation
- Statistical analysis

**Date:** October 2025  
**Status:** Production-ready


**Plot settings**
`plt.rcParams['figure.figsize'] = (12, 8)`


## Configuration

Set QAPLIB path and instances to test.


**QAPLIB configuration**
`QAPLIB_PATH = './qaplib'  # Adjust as needed`
**Try alternative paths if default doesn't exist**
**Known optimal/best-known values from QAPLIB**
`QAPLIB_OPTIMAL = {`
**Test instances (small to medium)**
`TEST_INSTANCES = [`
**Scalability test instances**
`SCALABILITY_INSTANCES = [`


## QAPLIB Data Loader


**# Distance matrix D**
`D_lines = lines[1:n+1]`
**# Weight matrix W**
`W_lines = lines[n+1:2*n+1]`


## Core QAP Functions




## Constraint Handling: Sinkhorn Projection


**# Normalize rows**
`X = X / X.sum(axis=1, keepdims=True)`
**# Normalize columns**
`X = X / X.sum(axis=0, keepdims=True)`
**# Check convergence**
`row_sums = X.sum(axis=1)`


## Rounding: Hungarian Algorithm


**# Maximize similarity to X (minimize negative)**
`row_ind, col_ind = linear_sum_assignment(-X)`


## 1. Spectral Initialization

Multiple strategies for initializing from eigenvectors.


**# Randomized SVD (faster for large matrices)**
**# Fallback to full SVD**
`eigvals_D, eigvecs_D = linalg.eigh(D)`
**# Full eigendecomposition**
`eigvals_D, eigvecs_D = linalg.eigh(D)`
**# Outer product**
`X0 = np.outer(np.abs(v_D), np.abs(v_W))`
**# Randomized SVD**
**# Sort by eigenvalue (descending)**
`idx_D = np.argsort(s_D)[::-1]`
**# Fallback**
`eigvals_D, eigvecs_D = linalg.eigh(D)`
**# Full eigendecomposition**
`eigvals_D, eigvecs_D = linalg.eigh(D)`
**# Take top k**
`idx_D = np.argsort(eigvals_D)[::-1][:k_actual]`
**# Compute weights based on strategy**
`def compute_weights(eigvals, strategy):`
**# Lorentzian kernel based on gaps**
`weights = []`
**# Lorentzian: 1 / (1 + gap^2)**
`weight = 1.0 / (1.0 + gap**2)`
**# Golden ratio decay**
`phi = (1 + np.sqrt(5)) / 2`
**# Weight by eigenvalue magnitude**
`weights = np.abs(eigvals)`
**# Ensemble initialization**
`X0 = np.zeros((n, n))`


## 2. Preconditioning

FFT-Laplace and other preconditioners.


**# Compute eigenvalues**
**# Spectral scaling (simplified FFT-Laplace)**
**# Full version would use FFT to solve circulant system**
**# Fallback to Jacobi**


## 3. Regularization

Vertex-promotion regularizer.


**# Regularization value**
`reg_val = -epsilon * np.sum(X ** 2)`
**# Gradient of regularizer**
`grad_reg = -2 * epsilon * X`


## 4. Momentum

Polyak and Nesterov momentum.


**# v = beta * v + grad**
`self.velocity = self.beta * self.velocity + grad`
**# v_new = beta * v + grad**
**# return beta * v_new + grad (lookahead)**
`v_new = self.beta * self.velocity + grad`


## 5. Time Integration

Explicit Euler vs IMEX.


**# X_new = X - dt * grad**
**# Explicit step**
`X_explicit = X - dt * grad`
**# Implicit step (projection handles constraint)**
**# Will be applied after this function returns**


## 6. Rounding Strategy

Final, periodic, or adaptive rounding.


**# Check if stalled**
`if len(history['obj']) >= self.patience:`


## 7. Local Search: 2-opt

Post-rounding refinement.


**# Try all pairs**
**# Swap**
`perm_new = perm.copy()`
**# Evaluate**
`P_new = perm_to_matrix(perm_new)`
**# Accept if better**


## Unified Solver Framework

Modular solver that accepts all component configurations.


**# Initialize**
`X = init_fn(D, W)`
**# Reset momentum**
**# History tracking**
`history = {`
**# Compute objective gradient**
`grad = compute_gradient(D, W, X)`
**# Add regularization gradient**
`reg_val, grad_reg = regularizer_fn(X)`
**# Apply preconditioning**
`grad = precond_fn(grad, D, W)`
**# Apply momentum**
`grad = momentum.update(grad, X)`
**# Integration step**
`X_new = integrator_fn(X, grad, lr)`
**# Project to doubly-stochastic**
`X_new = sinkhorn_projection(X_new)`
**# Check if should round now**
**# Update X to rounded version (optional: helps escape local minima)**
**# X_new = P  # Uncomment to snap to permutation**

**# Record progress**
`perm, P = hungarian_rounding(X_new)`
**# Check convergence**
**# Final rounding**
`perm, P = hungarian_rounding(X)`
**# Optional 2-opt refinement**


## Ablation Configurations

Define all configurations to test.


**Configuration builder**
`def build_config(name, **components):`
**Define ablation configurations**
`ABLATION_CONFIGS = [`
**# Baseline**
**# Initialization ablation**
**# Preconditioning ablation**
**# Integration ablation**
**# Momentum ablation**
**# Regularization ablation**
**# 2-opt ablation**
**# Our complete method**


## Run Ablation Study

Test all configurations on all instances.


**Run experiments**
`all_results = []`
**# Load instance**
`n, D, W = load_qaplib_instance(filepath)`
**# Test each configuration**
**# Compute metrics**
**# Time-to-quality (TTQ) at 10%**
`ttq_10 = None`


## Results Analysis


**Create results dataframe**
`df_results = pd.DataFrame(all_results)`
**Summary table**
**Best configuration per instance**
`print("\n" + "="*80)`
**Average performance by configuration**
`print("\n" + "="*80)`
**Win rate**
`print("\n" + "="*80)`


## Figure Generation

### 1. Convergence Plots


**Plot convergence for each instance**
**# Objective convergence**
`ax = axes[0]`
**# DS error**
`ax = axes[1]`
**# Rounding loss**
`ax = axes[2]`


### 2. Ablation Study Visualization


**Ablation study: show incremental improvement**
**Focus on configurations that progressively add components**

**Filter results**
**# Average gap across instances**
`avg_gaps = df_ablation.groupby('Config')['Gap %'].mean()`
**# Annotate with gap values**


### 3. Gap CDF (Cumulative Distribution)


**Gap CDF for key configurations**


### 4. Runtime vs Gap Trade-off


**Runtime vs gap scatter plot**
`fig, ax = plt.subplots(figsize=(10, 6))`


### 5. Method Hierarchy Tree Diagram


**Create tree diagram showing QAP method hierarchy**
`fig, ax = plt.subplots(figsize=(14, 10))`


## Statistical Tables

Generate comprehensive statistical summaries.


**Per-instance detailed table**
**# Sort by gap**
`inst_data_sorted = inst_data.sort_values('Gap %')`
**# Display key metrics**
**Aggregated statistics**
`print("\n" + "="*100)`
**Save to CSV**


## Conclusions

Summary of findings from comprehensive ablation study.









