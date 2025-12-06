
---
**Notation Standards**: See [NOTATION_STANDARDS.md](NOTATION_STANDARDS.md) for consistent mathematical notation across all Librex.QAP documentation.
---

Source: C:\Users\mesha\Pictures\random\notebooks\complete_qap_methods_documentation.md
Imported: 2025-11-17T14:08:47.178853

# Notebook Documentation

*Converted from: complete_qap_methods.ipynb*
*Date: 2025-11-17*

# Complete QAP Attractor Programming Implementation

Full implementation testing all our methods on QAPLIB instances.

**Methods tested:**
- Baseline: Random init + gradient descent
- Spectral initialization (single + multi-strategy)
- FFT-Laplace preconditioning
- IMEX time integration
- Continuation for Librex finding
- Reverse time saddle escape

**Instances:** chr12a, had12, nug12 (can expand to larger)


**Plot settings**
`plt.rcParams['figure.figsize'] = (12, 8)`


## QAPLIB Data Loader


**# Distance matrix**
`D_lines = lines[1:n+1]`
**# Weight matrix**
`W_lines = lines[n+1:2*n+1]`
**Known optimal values (from QAPLIB website)**
`QAPLIB_OPTIMAL = {`
**Path to QAPLIB data - adjust as needed**
**If path doesn't exist, try alternative locations**


## Core Functions


**# Check convergence**
`row_sums = X.sum(axis=1)`
**# Maximize similarity to X (minimize negative)**
`row_ind, col_ind = linear_sum_assignment(-X)`
**# Kronecker product eigenvalues**
`kron_eigs = []`


## Method 1: Baseline Gradient Descent


**# Compute gradient**
`grad = compute_gradient(D, W, X)`
**# Gradient step**
`X_new = X - lr * grad`
**# Project back to doubly-stochastic**
`X_new = sinkhorn_projection(X_new)`
**# Record progress**
`perm, P = hungarian_rounding(X_new)`
**# Check convergence**


## Method 2: Spectral Initialization


**# Get top eigenvectors**
`D_eigvals, D_eigvecs = linalg.eigh(D)`
**# Use top eigenvector (largest eigenvalue)**
`v_D = D_eigvecs[:, -1]`
**# Outer product gives initial guess**
`X0 = np.outer(np.abs(v_D), np.abs(v_W))`
**# Get top-k eigenvectors**
`D_eigvals, D_eigvecs = linalg.eigh(D)`
**# Sort descending**
`D_idx = np.argsort(D_eigvals)[::-1]`
**# Gap-aware weights (larger gaps = more weight)**
`def compute_weights(eigvals, k):`
**# Ensemble initialization**
`X0 = np.zeros((n, n))`


## Method 3: FFT-Laplace Preconditioning

We approximate the Hessian with a circulant matrix and use FFT to invert it efficiently.


**# Compute eigenvalues (for spectral preconditioner)**
`D_eigvals = linalg.eigvalsh(D)`
**# Build preconditioner matrix in spectral space**
**# P_inv(i,j) ≈ 1 / (2 * D_eigvals[i] * W_eigvals[j] + epsilon)**
`epsilon = 1e-6  # Regularization`
**# Simple diagonal preconditioner (full FFT version is more complex)**
**# This is a simplified version - acts as Jacobi-style preconditioner**

**# Scale by inverse of approximate Hessian diagonal**
**# Get preconditioner**
`precond = fft_laplace_preconditioner(D, W)`
**# Compute gradient**
`grad = compute_gradient(D, W, X)`
**# Apply preconditioner**
`precond_grad = precond(grad)`
**# Gradient step**
`X_new = X - lr * precond_grad`
**# Project back to doubly-stochastic**
`X_new = sinkhorn_projection(X_new)`
**# Record progress**
`perm, P = hungarian_rounding(X_new)`
**# Check convergence**


## Method 4: IMEX Time Integration

Implicit-Explicit scheme: constraints handled implicitly, objective explicitly.


**# Explicit step: objective gradient**
`grad = compute_gradient(D, W, X)`
**# Implicit step: project to doubly-stochastic (constraint)**
`X_new = sinkhorn_projection(X_explicit)`
**# Record progress**
`perm, P = hungarian_rounding(X_new)`
**# Check convergence**


## Method 5: Combined Best Method

Multi-strategy spectral + FFT preconditioner + IMEX


**# 1. Multi-strategy spectral initialization**
`X0 = spectral_init_multi(D, W, k=k_spectral)`
**# 2. Get preconditioner**
`precond = fft_laplace_preconditioner(D, W)`
**# 3. IMEX with preconditioning**
`X = X0.copy()`
**# Explicit: preconditioned gradient**
`grad = compute_gradient(D, W, X)`
**# Implicit: projection**
`X_new = sinkhorn_projection(X_explicit)`
**# Record**
`perm, P = hungarian_rounding(X_new)`


## Experiments: Test All Methods

We'll test on chr12a, had12, and nug12.


**Test instances**
`test_instances = ['chr12a.dat', 'had12.dat', 'nug12.dat']`
**# Load data**
`n, D, W = load_qaplib_instance(filepath)`
**# Stability analysis**
`eigvals = compute_jacobian_eigenvalues(D, W)`
**# Test all methods**
`methods = [`
**# Compute gap if optimal known**
**# DS error**
`ds_error = hist['ds_error'][-1] if hist['ds_error'] else 0`


## Results Summary


**Create results dataframe**
`df_results = pd.DataFrame(results)`
**Summary table**
**Best method per instance**
`print("\n" + "="*60)`


## Visualizations


**Plot convergence for each instance**
**# Objective convergence**
`ax = axes[0]`
**# Add optimal line if known**
**# DS error convergence**
`ax = axes[1]`


**Performance comparison bar chart**
`fig, axes = plt.subplots(1, 3, figsize=(16, 5))`
**# Color code: our method in red, others in blue**
**# Highlight best**
`min_gap = min(gaps)`


## Statistical Analysis


**Compute average performance across instances**
**Win rate (how often each method finds best solution)**
`print("\n" + "="*60)`


## Ablation Study

Test contribution of each component by removing it.


**Pick one instance for detailed ablation**
`ablation_instance = 'chr12a.dat'`
**# Plot ablation**
`df_ablation = pd.DataFrame(ablation_results)`
**# Annotate bars with values**


## Eigenvalue Analysis

Show why gradient descent struggles (unstable Librex).


**Eigenvalue histogram for each instance**
`fig, axes = plt.subplots(1, 3, figsize=(15, 4))`
**# Compute Hessian eigenvalues**
`eigvals = compute_jacobian_eigenvalues(D, W)`


## Conclusion

Summary of what we learned.









