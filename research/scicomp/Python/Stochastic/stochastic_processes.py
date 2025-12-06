"""
Stochastic Processes Module
===========================
Implementation of various stochastic processes and methods for scientific computing,
including Brownian motion, random walks, SDEs, and Monte Carlo simulations.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import scipy.stats as stats
from scipy import integrate
from typing import Tuple, Optional, Callable, Union, List, Dict, Any
import warnings
# Berkeley colors
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
class StochasticProcess:
    """
    Base class for stochastic processes.
    Provides common interface and utilities for various stochastic processes
    used in scientific computing and financial modeling.
    """
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize stochastic process.
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    def set_seed(self, seed: int):
        """Set random seed."""
        self.rng = np.random.RandomState(seed)
class BrownianMotion(StochasticProcess):
    """
    Brownian motion (Wiener process) implementation.
    Generates sample paths of standard and geometric Brownian motion
    with various extensions including fractional Brownian motion.
    """
    def __init__(self, drift: float = 0.0, volatility: float = 1.0,
                 seed: Optional[int] = None):
        """
        Initialize Brownian motion.
        Args:
            drift: Drift coefficient (μ)
            volatility: Diffusion coefficient (σ)
            seed: Random seed
        """
        super().__init__(seed)
        self.drift = drift
        self.volatility = volatility
    def generate_path(self, T: float, n_steps: int, x0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a sample path of Brownian motion.
        Args:
            T: Total time
            n_steps: Number of time steps
            x0: Initial value
        Returns:
            Time array and path array
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        # Generate increments
        dW = self.rng.normal(0, np.sqrt(dt), n_steps)
        # Cumulative sum for Brownian motion
        W = np.zeros(n_steps + 1)
        W[0] = x0
        for i in range(n_steps):
            W[i + 1] = W[i] + self.drift * dt + self.volatility * dW[i]
        return t, W
    def generate_geometric_path(self, T: float, n_steps: int,
                               S0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate geometric Brownian motion path.
        Used for modeling stock prices and other positive processes.
        Args:
            T: Total time
            n_steps: Number of steps
            S0: Initial value
        Returns:
            Time array and geometric Brownian motion path
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        # Generate standard Brownian motion
        dW = self.rng.normal(0, np.sqrt(dt), n_steps)
        # Geometric Brownian motion
        S = np.zeros(n_steps + 1)
        S[0] = S0
        for i in range(n_steps):
            S[i + 1] = S[i] * np.exp((self.drift - 0.5 * self.volatility**2) * dt +
                                     self.volatility * dW[i])
        return t, S
    def generate_bridge(self, T: float, n_steps: int,
                       x0: float = 0.0, xT: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Brownian bridge.
        Brownian motion conditioned to have specific endpoint.
        Args:
            T: Total time
            n_steps: Number of steps
            x0: Initial value
            xT: Terminal value
        Returns:
            Time array and Brownian bridge path
        """
        t = np.linspace(0, T, n_steps + 1)
        dt = T / n_steps
        # Generate standard Brownian motion
        _, W = self.generate_path(T, n_steps, 0)
        # Apply bridge transformation
        bridge = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            s = t[i]
            if s < T:
                bridge[i] = x0 + W[i] - (s/T) * (W[-1] - (xT - x0))
            else:
                bridge[i] = xT
        return t, bridge
    def fractional_brownian_motion(self, T: float, n_steps: int,
                                  H: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fractional Brownian motion using Cholesky method.
        Args:
            T: Total time
            n_steps: Number of steps
            H: Hurst parameter (0 < H < 1)
        Returns:
            Time array and fBm path
        """
        if not 0 < H < 1:
            raise ValueError("Hurst parameter must be in (0, 1)")
        t = np.linspace(0, T, n_steps + 1)
        dt = T / n_steps
        # Covariance matrix for fBm
        cov_matrix = np.zeros((n_steps + 1, n_steps + 1))
        for i in range(n_steps + 1):
            for j in range(n_steps + 1):
                ti = t[i]
                tj = t[j]
                cov_matrix[i, j] = 0.5 * (ti**(2*H) + tj**(2*H) - abs(ti - tj)**(2*H))
        # Cholesky decomposition
        L = np.linalg.cholesky(cov_matrix + 1e-10 * np.eye(n_steps + 1))
        # Generate fBm
        Z = self.rng.normal(0, 1, n_steps + 1)
        fBm = L @ Z
        return t, fBm
class RandomWalk(StochasticProcess):
    """
    Random walk processes in discrete and continuous time.
    Implements various random walk models including simple, self-avoiding,
    and Lévy walks.
    """
    def simple_walk_1d(self, n_steps: int, p: float = 0.5) -> np.ndarray:
        """
        Generate 1D random walk.
        Args:
            n_steps: Number of steps
            p: Probability of moving right
        Returns:
            Position array
        """
        steps = self.rng.choice([1, -1], size=n_steps, p=[p, 1-p])
        position = np.concatenate([[0], np.cumsum(steps)])
        return position
    def simple_walk_2d(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 2D random walk.
        Args:
            n_steps: Number of steps
        Returns:
            x and y position arrays
        """
        # Random directions
        angles = self.rng.uniform(0, 2*np.pi, n_steps)
        dx = np.cos(angles)
        dy = np.sin(angles)
        x = np.concatenate([[0], np.cumsum(dx)])
        y = np.concatenate([[0], np.cumsum(dy)])
        return x, y
    def levy_walk(self, n_steps: int, alpha: float = 2.0) -> np.ndarray:
        """
        Generate Lévy walk with power-law step sizes.
        Args:
            n_steps: Number of steps
            alpha: Lévy exponent (0 < alpha <= 2)
        Returns:
            Position array
        """
        if not 0 < alpha <= 2:
            raise ValueError("Lévy exponent must be in (0, 2]")
        # Generate Lévy-distributed step sizes
        if alpha == 2:
            # Gaussian case
            steps = self.rng.normal(0, 1, n_steps)
        else:
            # Lévy stable distribution (approximation)
            u = self.rng.uniform(-np.pi/2, np.pi/2, n_steps)
            v = self.rng.exponential(1, n_steps)
            steps = (np.sin(alpha * u) / np.cos(u)**(1/alpha) *
                    (np.cos(u - alpha * u) / v)**((1 - alpha) / alpha))
        position = np.concatenate([[0], np.cumsum(steps)])
        return position
    def self_avoiding_walk_2d(self, n_steps: int,
                             max_attempts: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate self-avoiding random walk in 2D.
        Args:
            n_steps: Desired number of steps
            max_attempts: Maximum attempts to find valid step
        Returns:
            x and y position arrays
        """
        visited = set()
        x, y = [0], [0]
        visited.add((0, 0))
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for _ in range(n_steps):
            current = (x[-1], y[-1])
            valid_moves = []
            for dx, dy in directions:
                new_pos = (current[0] + dx, current[1] + dy)
                if new_pos not in visited:
                    valid_moves.append(new_pos)
            if not valid_moves:
                break  # Trapped
            next_pos = valid_moves[self.rng.choice(len(valid_moves))]
            visited.add(next_pos)
            x.append(next_pos[0])
            y.append(next_pos[1])
        return np.array(x), np.array(y)
class StochasticDifferentialEquation(StochasticProcess):
    """
    Numerical solutions for stochastic differential equations.
    Implements various numerical schemes for solving SDEs including
    Euler-Maruyama, Milstein, and higher-order methods.
    """
    def euler_maruyama(self, drift: Callable, diffusion: Callable,
                      x0: float, T: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
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
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        t = np.linspace(0, T, n_steps + 1)
        X = np.zeros(n_steps + 1)
        X[0] = x0
        for i in range(n_steps):
            dW = self.rng.normal(0, sqrt_dt)
            X[i + 1] = X[i] + drift(X[i], t[i]) * dt + diffusion(X[i], t[i]) * dW
        return t, X
    def milstein(self, drift: Callable, diffusion: Callable,
                diffusion_prime: Callable, x0: float,
                T: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
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
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        t = np.linspace(0, T, n_steps + 1)
        X = np.zeros(n_steps + 1)
        X[0] = x0
        for i in range(n_steps):
            dW = self.rng.normal(0, sqrt_dt)
            b = diffusion(X[i], t[i])
            b_prime = diffusion_prime(X[i], t[i])
            X[i + 1] = (X[i] + drift(X[i], t[i]) * dt + b * dW +
                       0.5 * b * b_prime * (dW**2 - dt))
        return t, X
    def strong_taylor_1_5(self, drift: Callable, diffusion: Callable,
                         drift_x: Callable, drift_t: Callable,
                         diffusion_x: Callable, diffusion_t: Callable,
                         x0: float, T: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
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
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        t = np.linspace(0, T, n_steps + 1)
        X = np.zeros(n_steps + 1)
        X[0] = x0
        for i in range(n_steps):
            # Generate random increments
            dW = self.rng.normal(0, sqrt_dt)
            dZ = self.rng.normal(0, sqrt_dt) * sqrt_dt / np.sqrt(3)
            # Evaluate functions
            a = drift(X[i], t[i])
            b = diffusion(X[i], t[i])
            a_x = drift_x(X[i], t[i])
            a_t = drift_t(X[i], t[i])
            b_x = diffusion_x(X[i], t[i])
            b_t = diffusion_t(X[i], t[i])
            # Taylor expansion terms
            L0_a = a_t + a * a_x + 0.5 * b**2 * a_x
            L1_a = b * a_x
            L0_b = b_t + a * b_x + 0.5 * b**2 * b_x
            L1_b = b * b_x
            # Update
            X[i + 1] = (X[i] + a * dt + b * dW +
                       0.5 * L1_b * (dW**2 - dt) +
                       L1_a * dZ +
                       0.5 * (L0_a * dt**2 + L0_b * dW * dt))
        return t, X
class OrnsteinUhlenbeck(StochasticProcess):
    """
    Ornstein-Uhlenbeck process (mean-reverting process).
    Used for modeling interest rates, volatility, and other mean-reverting phenomena.
    """
    def __init__(self, theta: float, mu: float, sigma: float,
                 seed: Optional[int] = None):
        """
        Initialize OU process.
        dX_t = theta*(mu - X_t)dt + sigma*dW_t
        Args:
            theta: Mean reversion speed
            mu: Long-term mean
            sigma: Volatility
            seed: Random seed
        """
        super().__init__(seed)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
    def generate_path(self, T: float, n_steps: int, x0: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate OU process path using exact solution.
        Args:
            T: Total time
            n_steps: Number of steps
            x0: Initial value
        Returns:
            Time array and OU process path
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        X = np.zeros(n_steps + 1)
        X[0] = x0
        # Use exact solution for OU process
        exp_theta_dt = np.exp(-self.theta * dt)
        variance = (self.sigma**2 / (2 * self.theta)) * (1 - exp_theta_dt**2)
        std_dev = np.sqrt(variance)
        for i in range(n_steps):
            X[i + 1] = (X[i] * exp_theta_dt +
                       self.mu * (1 - exp_theta_dt) +
                       std_dev * self.rng.normal())
        return t, X
    def stationary_distribution(self) -> Tuple[float, float]:
        """
        Get stationary distribution parameters.
        Returns:
            Mean and variance of stationary distribution
        """
        mean = self.mu
        variance = self.sigma**2 / (2 * self.theta)
        return mean, variance
class JumpDiffusion(StochasticProcess):
    """
    Jump-diffusion process (Merton model).
    Combines continuous Brownian motion with discrete jumps,
    used in finance and physics.
    """
    def __init__(self, drift: float, volatility: float,
                 jump_rate: float, jump_mean: float, jump_std: float,
                 seed: Optional[int] = None):
        """
        Initialize jump-diffusion process.
        Args:
            drift: Drift coefficient
            volatility: Diffusion coefficient
            jump_rate: Poisson jump intensity (λ)
            jump_mean: Mean jump size
            jump_std: Jump size standard deviation
            seed: Random seed
        """
        super().__init__(seed)
        self.drift = drift
        self.volatility = volatility
        self.jump_rate = jump_rate
        self.jump_mean = jump_mean
        self.jump_std = jump_std
    def generate_path(self, T: float, n_steps: int, S0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate jump-diffusion path.
        Args:
            T: Total time
            n_steps: Number of steps
            S0: Initial value
        Returns:
            Time array and jump-diffusion path
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        S = np.zeros(n_steps + 1)
        S[0] = S0
        for i in range(n_steps):
            # Brownian component
            dW = self.rng.normal(0, np.sqrt(dt))
            # Jump component
            N = self.rng.poisson(self.jump_rate * dt)  # Number of jumps
            if N > 0:
                J = np.sum(self.rng.normal(self.jump_mean, self.jump_std, N))
            else:
                J = 0
            # Update (multiplicative model)
            S[i + 1] = S[i] * np.exp((self.drift - 0.5 * self.volatility**2) * dt +
                                    self.volatility * dW + J)
        return t, S
def demo_stochastic_processes():
    """Demonstrate stochastic processes."""
    import matplotlib.pyplot as plt
    print("Stochastic Processes Demo")
    print("=========================")
    # Set random seed for reproducibility
    seed = 42
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # 1. Brownian Motion
    bm = BrownianMotion(drift=0.1, volatility=0.3, seed=seed)
    t, W = bm.generate_path(T=10, n_steps=1000)
    axes[0, 0].plot(t, W, 'b-', alpha=0.7)
    axes[0, 0].set_title('Brownian Motion')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    # 2. Geometric Brownian Motion
    t, S = bm.generate_geometric_path(T=10, n_steps=1000, S0=100)
    axes[0, 1].plot(t, S, 'g-', alpha=0.7)
    axes[0, 1].set_title('Geometric Brownian Motion')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    # 3. Random Walk 2D
    rw = RandomWalk(seed=seed)
    x, y = rw.simple_walk_2d(n_steps=1000)
    axes[0, 2].plot(x, y, 'r-', alpha=0.5)
    axes[0, 2].scatter(x[0], y[0], c='green', s=100, marker='o', label='Start')
    axes[0, 2].scatter(x[-1], y[-1], c='red', s=100, marker='s', label='End')
    axes[0, 2].set_title('2D Random Walk')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    # 4. Ornstein-Uhlenbeck Process
    ou = OrnsteinUhlenbeck(theta=1.0, mu=0.0, sigma=0.3, seed=seed)
    t, X = ou.generate_path(T=20, n_steps=2000, x0=2.0)
    axes[1, 0].plot(t, X, 'purple', alpha=0.7)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('Ornstein-Uhlenbeck (Mean-Reverting)')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True, alpha=0.3)
    # 5. Jump-Diffusion Process
    jd = JumpDiffusion(drift=0.05, volatility=0.2, jump_rate=0.5,
                      jump_mean=0.0, jump_std=0.1, seed=seed)
    t, J = jd.generate_path(T=10, n_steps=1000, S0=100)
    axes[1, 1].plot(t, J, 'orange', alpha=0.7)
    axes[1, 1].set_title('Jump-Diffusion Process')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)
    # 6. Lévy Walk
    levy_pos = rw.levy_walk(n_steps=500, alpha=1.5)
    axes[1, 2].plot(levy_pos, 'brown', alpha=0.7)
    axes[1, 2].set_title('Lévy Walk (α=1.5)')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Position')
    axes[1, 2].grid(True, alpha=0.3)
    plt.tight_layout()
    print("\nStochastic process statistics:")
    print(f"Brownian Motion - Final value: {W[-1]:.4f}")
    print(f"Geometric BM - Final value: {S[-1]:.2f}")
    print(f"2D Walk - Distance from origin: {np.sqrt(x[-1]**2 + y[-1]**2):.2f}")
    print(f"OU Process - Mean: {np.mean(X):.4f}, Std: {np.std(X):.4f}")
    print(f"Jump-Diffusion - Final value: {J[-1]:.2f}")
    print("\nDemo completed successfully!")
    return fig
if __name__ == "__main__":
    demo_stochastic_processes()
    import matplotlib.pyplot as plt
    plt.show()