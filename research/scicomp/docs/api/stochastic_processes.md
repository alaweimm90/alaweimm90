# stochastic_processes
**Module:** `Python/Stochastic/stochastic_processes.py`
## Overview
Stochastic Processes Module
===========================
Implementation of various stochastic processes and methods for scientific computing,
including Brownian motion, random walks, SDEs, and Monte Carlo simulations.
Author: Berkeley SciComp Team
Date: 2024
## Constants
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
## Functions
### `demo_stochastic_processes()`
Demonstrate stochastic processes.
**Source:** [Line 568](Python/Stochastic/stochastic_processes.py#L568)
## Classes
### `StochasticProcess`
Base class for stochastic processes.
Provides common interface and utilities for various stochastic processes
used in scientific computing and financial modeling.
#### Methods
##### `__init__(self, seed)`
Initialize stochastic process.
Args:
seed: Random seed for reproducibility
**Source:** [Line 31](Python/Stochastic/stochastic_processes.py#L31)
##### `set_seed(self, seed)`
Set random seed.
**Source:** [Line 44](Python/Stochastic/stochastic_processes.py#L44)
**Class Source:** [Line 23](Python/Stochastic/stochastic_processes.py#L23)
### `BrownianMotion`
Brownian motion (Wiener process) implementation.
Generates sample paths of standard and geometric Brownian motion
with various extensions including fractional Brownian motion.
#### Methods
##### `__init__(self, drift, volatility, seed)`
Initialize Brownian motion.
Args:
drift: Drift coefficient (μ)
volatility: Diffusion coefficient (σ)
seed: Random seed
**Source:** [Line 57](Python/Stochastic/stochastic_processes.py#L57)
##### `generate_path(self, T, n_steps, x0)`
Generate a sample path of Brownian motion.
Args:
T: Total time
n_steps: Number of time steps
x0: Initial value
Returns:
Time array and path array
**Source:** [Line 71](Python/Stochastic/stochastic_processes.py#L71)
##### `generate_geometric_path(self, T, n_steps, S0)`
Generate geometric Brownian motion path.
Used for modeling stock prices and other positive processes.
Args:
T: Total time
n_steps: Number of steps
S0: Initial value
Returns:
Time array and geometric Brownian motion path
**Source:** [Line 98](Python/Stochastic/stochastic_processes.py#L98)
##### `generate_bridge(self, T, n_steps, x0, xT)`
Generate Brownian bridge.
Brownian motion conditioned to have specific endpoint.
Args:
T: Total time
n_steps: Number of steps
x0: Initial value
xT: Terminal value
Returns:
Time array and Brownian bridge path
**Source:** [Line 129](Python/Stochastic/stochastic_processes.py#L129)
##### `fractional_brownian_motion(self, T, n_steps, H)`
Generate fractional Brownian motion using Cholesky method.
Args:
T: Total time
n_steps: Number of steps
H: Hurst parameter (0 < H < 1)
Returns:
Time array and fBm path
**Source:** [Line 162](Python/Stochastic/stochastic_processes.py#L162)
**Class Source:** [Line 49](Python/Stochastic/stochastic_processes.py#L49)
### `RandomWalk`
Random walk processes in discrete and continuous time.
Implements various random walk models including simple, self-avoiding,
and Lévy walks.
#### Methods
##### `simple_walk_1d(self, n_steps, p)`
Generate 1D random walk.
Args:
n_steps: Number of steps
p: Probability of moving right
Returns:
Position array
**Source:** [Line 207](Python/Stochastic/stochastic_processes.py#L207)
##### `simple_walk_2d(self, n_steps)`
Generate 2D random walk.
Args:
n_steps: Number of steps
Returns:
x and y position arrays
**Source:** [Line 222](Python/Stochastic/stochastic_processes.py#L222)
##### `levy_walk(self, n_steps, alpha)`
Generate Lévy walk with power-law step sizes.
Args:
n_steps: Number of steps
alpha: Lévy exponent (0 < alpha <= 2)
Returns:
Position array
**Source:** [Line 242](Python/Stochastic/stochastic_processes.py#L242)
##### `self_avoiding_walk_2d(self, n_steps, max_attempts)`
Generate self-avoiding random walk in 2D.
Args:
n_steps: Desired number of steps
max_attempts: Maximum attempts to find valid step
Returns:
x and y position arrays
**Source:** [Line 271](Python/Stochastic/stochastic_processes.py#L271)
**Class Source:** [Line 199](Python/Stochastic/stochastic_processes.py#L199)
### `StochasticDifferentialEquation`
Numerical solutions for stochastic differential equations.
Implements various numerical schemes for solving SDEs including
Euler-Maruyama, Milstein, and higher-order methods.
#### Methods
##### `euler_maruyama(self, drift, diffusion, x0, T, n_steps)`
Solve SDE using Euler-Maruyama method.
dX_t = drift(X_t, t)dt + diffusion(X_t, t)dW_t
Args:
drift: Drift function a(x, t)
diffusion: Diffusion function b(x, t)
x0: Initial condition
T: Final time
n_steps: Number of time steps
Returns:
Time array and solution array
**Source:** [Line 317](Python/Stochastic/stochastic_processes.py#L317)
##### `milstein(self, drift, diffusion, diffusion_prime, x0, T, n_steps)`
Solve SDE using Milstein method (higher order accuracy).
Args:
drift: Drift function a(x, t)
diffusion: Diffusion function b(x, t)
diffusion_prime: Derivative of diffusion w.r.t. x
x0: Initial condition
T: Final time
n_steps: Number of steps
Returns:
Time array and solution array
**Source:** [Line 347](Python/Stochastic/stochastic_processes.py#L347)
##### `strong_taylor_1_5(self, drift, diffusion, drift_x, drift_t, diffusion_x, diffusion_t, x0, T, n_steps)`
Strong Taylor scheme of order 1.5.
Higher-order method for increased accuracy.
Args:
drift, diffusion: SDE coefficients
drift_x, drift_t: Partial derivatives of drift
diffusion_x, diffusion_t: Partial derivatives of diffusion
x0: Initial condition
T: Final time
n_steps: Number of steps
Returns:
Time array and solution array
**Source:** [Line 381](Python/Stochastic/stochastic_processes.py#L381)
**Class Source:** [Line 309](Python/Stochastic/stochastic_processes.py#L309)
### `OrnsteinUhlenbeck`
Ornstein-Uhlenbeck process (mean-reverting process).
Used for modeling interest rates, volatility, and other mean-reverting phenomena.
#### Methods
##### `__init__(self, theta, mu, sigma, seed)`
Initialize OU process.
dX_t = theta*(mu - X_t)dt + sigma*dW_t
Args:
theta: Mean reversion speed
mu: Long-term mean
sigma: Volatility
seed: Random seed
**Source:** [Line 443](Python/Stochastic/stochastic_processes.py#L443)
##### `generate_path(self, T, n_steps, x0)`
Generate OU process path using exact solution.
Args:
T: Total time
n_steps: Number of steps
x0: Initial value
Returns:
Time array and OU process path
**Source:** [Line 461](Python/Stochastic/stochastic_processes.py#L461)
##### `stationary_distribution(self)`
Get stationary distribution parameters.
Returns:
Mean and variance of stationary distribution
**Source:** [Line 491](Python/Stochastic/stochastic_processes.py#L491)
**Class Source:** [Line 436](Python/Stochastic/stochastic_processes.py#L436)
### `JumpDiffusion`
Jump-diffusion process (Merton model).
Combines continuous Brownian motion with discrete jumps,
used in finance and physics.
#### Methods
##### `__init__(self, drift, volatility, jump_rate, jump_mean, jump_std, seed)`
Initialize jump-diffusion process.
Args:
drift: Drift coefficient
volatility: Diffusion coefficient
jump_rate: Poisson jump intensity (λ)
jump_mean: Mean jump size
jump_std: Jump size standard deviation
seed: Random seed
**Source:** [Line 511](Python/Stochastic/stochastic_processes.py#L511)
##### `generate_path(self, T, n_steps, S0)`
Generate jump-diffusion path.
Args:
T: Total time
n_steps: Number of steps
S0: Initial value
Returns:
Time array and jump-diffusion path
**Source:** [Line 532](Python/Stochastic/stochastic_processes.py#L532)
**Class Source:** [Line 503](Python/Stochastic/stochastic_processes.py#L503)