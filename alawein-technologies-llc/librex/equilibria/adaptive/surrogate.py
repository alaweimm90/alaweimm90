"""
Surrogate Models for Expensive Function Evaluations

This module implements Gaussian Process-based surrogate models for
expensive optimization problems with various acquisition functions.

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

logger = logging.getLogger(__name__)


class AcquisitionFunction(Enum):
    """Available acquisition functions for Bayesian optimization."""

    EI = 'expected_improvement'
    PI = 'probability_improvement'
    UCB = 'upper_confidence_bound'
    EI_MCMC = 'expected_improvement_mcmc'
    THOMPSON = 'thompson_sampling'


@dataclass
class SurrogateState:
    """State of the surrogate optimization process."""

    X_observed: np.ndarray
    y_observed: np.ndarray
    best_value: float
    best_point: np.ndarray
    n_evaluations: int
    gp_model: GaussianProcessRegressor


class GPSurrogate:
    """
    Gaussian Process surrogate model for expensive function evaluations.

    This model builds a probabilistic approximation of the objective function
    and uses acquisition functions to guide the search efficiently.
    """

    def __init__(
        self,
        kernel: Optional[Any] = None,
        alpha: float = 1e-6,
        n_restarts_optimizer: int = 10,
        normalize_y: bool = True
    ):
        """
        Initialize the GP surrogate model.

        Args:
            kernel: GP kernel (None = default Matern)
            alpha: Noise level in observations
            n_restarts_optimizer: Number of restarts for kernel optimization
            normalize_y: Whether to normalize target values
        """
        if kernel is None:
            # Default kernel: Constant * Matern + White noise
            kernel = ConstantKernel(1.0) * Matern(nu=2.5)

        self.kernel = kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y

        self.gp = None
        self.X_observed = None
        self.y_observed = None
        self.best_value = float('inf')
        self.best_point = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the GP model to observed data.

        Args:
            X: Observed input points (n_samples, n_features)
            y: Observed target values (n_samples,)
        """
        self.X_observed = X.copy()
        self.y_observed = y.copy()

        # Track best observation (assuming minimization)
        best_idx = np.argmin(y)
        self.best_value = y[best_idx]
        self.best_point = X[best_idx].copy()

        # Create and fit GP model
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y
        )
        self.gp.fit(X, y)

        logger.debug(f"GP fitted with {len(X)} observations")

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict mean and standard deviation at new points.

        Args:
            X: Points to predict (n_samples, n_features)
            return_std: Whether to return standard deviation

        Returns:
            Tuple of (mean, std) predictions
        """
        if self.gp is None:
            raise RuntimeError("Model must be fitted before prediction")

        return self.gp.predict(X, return_std=return_std)

    def expected_improvement(
        self,
        X: np.ndarray,
        xi: float = 0.01
    ) -> np.ndarray:
        """
        Calculate Expected Improvement acquisition function.

        EI(x) = E[max(f_min - f(x) - xi, 0)]

        Args:
            X: Points to evaluate (n_samples, n_features)
            xi: Exploration parameter (higher = more exploration)

        Returns:
            Expected improvement values
        """
        mu, sigma = self.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Calculate improvement
        improvement = self.best_value - mu - xi

        # Handle zero variance
        with np.errstate(divide='warn'):
            Z = improvement / (sigma + 1e-9)
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-9] = 0.0

        return ei.flatten()

    def probability_improvement(
        self,
        X: np.ndarray,
        xi: float = 0.01
    ) -> np.ndarray:
        """
        Calculate Probability of Improvement acquisition function.

        PI(x) = P(f(x) < f_min - xi)

        Args:
            X: Points to evaluate (n_samples, n_features)
            xi: Exploration parameter

        Returns:
            Probability of improvement values
        """
        mu, sigma = self.predict(X, return_std=True)

        # Calculate probability
        Z = (self.best_value - mu - xi) / (sigma + 1e-9)
        pi = norm.cdf(Z)

        return pi

    def upper_confidence_bound(
        self,
        X: np.ndarray,
        beta: float = 2.0
    ) -> np.ndarray:
        """
        Calculate Upper Confidence Bound acquisition function.

        UCB(x) = -mu(x) + beta * sigma(x)
        (Negative because we're minimizing)

        Args:
            X: Points to evaluate (n_samples, n_features)
            beta: Exploration parameter (higher = more exploration)

        Returns:
            UCB values (higher is better for acquisition)
        """
        mu, sigma = self.predict(X, return_std=True)

        # Return negative UCB for minimization
        return -mu + beta * sigma

    def thompson_sampling(
        self,
        X: np.ndarray,
        n_samples: int = 1
    ) -> np.ndarray:
        """
        Thompson sampling from the GP posterior.

        Args:
            X: Points to evaluate (n_samples, n_features)
            n_samples: Number of samples to draw

        Returns:
            Sampled function values
        """
        # Sample from the GP posterior
        samples = self.gp.sample_y(X, n_samples=n_samples, random_state=None)

        # Return mean of samples
        return np.mean(samples, axis=1)

    def next_sample(
        self,
        bounds: np.ndarray,
        acquisition: AcquisitionFunction = AcquisitionFunction.EI,
        n_candidates: int = 1000,
        **acq_kwargs
    ) -> np.ndarray:
        """
        Suggest the next point to sample based on acquisition function.

        Args:
            bounds: Variable bounds [(lower, upper), ...]
            acquisition: Acquisition function to use
            n_candidates: Number of random candidates to evaluate
            **acq_kwargs: Additional arguments for acquisition function

        Returns:
            Next point to evaluate
        """
        if self.gp is None:
            # Random sampling if no model yet
            n_dim = len(bounds)
            return np.random.uniform(
                bounds[:, 0],
                bounds[:, 1],
                size=n_dim
            )

        # Generate random candidates
        n_dim = len(bounds)
        candidates = np.random.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(n_candidates, n_dim)
        )

        # Evaluate acquisition function
        if acquisition == AcquisitionFunction.EI:
            acq_values = self.expected_improvement(candidates, **acq_kwargs)
        elif acquisition == AcquisitionFunction.PI:
            acq_values = self.probability_improvement(candidates, **acq_kwargs)
        elif acquisition == AcquisitionFunction.UCB:
            acq_values = self.upper_confidence_bound(candidates, **acq_kwargs)
        elif acquisition == AcquisitionFunction.THOMPSON:
            acq_values = -self.thompson_sampling(candidates)  # Negative for minimization
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition}")

        # Select best candidate
        best_idx = np.argmax(acq_values)
        best_candidate = candidates[best_idx]

        # Refine with local optimization
        result = minimize(
            lambda x: -self._acquisition_objective(x, acquisition, **acq_kwargs),
            best_candidate,
            bounds=bounds,
            method='L-BFGS-B'
        )

        return result.x

    def _acquisition_objective(
        self,
        x: np.ndarray,
        acquisition: AcquisitionFunction,
        **kwargs
    ) -> float:
        """Objective function for acquisition optimization."""
        x = x.reshape(1, -1)

        if acquisition == AcquisitionFunction.EI:
            return self.expected_improvement(x, **kwargs)[0]
        elif acquisition == AcquisitionFunction.PI:
            return self.probability_improvement(x, **kwargs)[0]
        elif acquisition == AcquisitionFunction.UCB:
            return self.upper_confidence_bound(x, **kwargs)[0]
        else:
            return 0.0


class SurrogateOptimizer:
    """
    Main surrogate-based optimization system.

    Combines Gaussian Process surrogate models with Bayesian optimization
    to efficiently optimize expensive objective functions.
    """

    def __init__(
        self,
        objective_func: Callable,
        bounds: np.ndarray,
        acquisition: AcquisitionFunction = AcquisitionFunction.EI,
        n_initial_points: int = 5,
        kernel: Optional[Any] = None
    ):
        """
        Initialize the surrogate optimizer.

        Args:
            objective_func: Expensive objective function to minimize
            bounds: Variable bounds [(lower, upper), ...]
            acquisition: Acquisition function strategy
            n_initial_points: Number of initial random samples
            kernel: GP kernel (None = default)
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.acquisition = acquisition
        self.n_initial_points = n_initial_points

        # Initialize surrogate model
        self.surrogate = GPSurrogate(kernel=kernel)

        # Optimization state
        self.X_observed = []
        self.y_observed = []
        self.iteration = 0

    def optimize(
        self,
        n_iterations: int = 50,
        n_parallel: int = 1,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run surrogate-based optimization.

        Args:
            n_iterations: Total number of function evaluations
            n_parallel: Number of parallel evaluations per iteration
            callback: Optional callback function(state) called each iteration

        Returns:
            Optimization result dictionary
        """
        # Initial sampling
        if len(self.X_observed) < self.n_initial_points:
            self._initial_sampling()

        # Main optimization loop
        for i in range(n_iterations - self.n_initial_points):
            # Fit surrogate model
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.surrogate.fit(X, y)

            # Get next sample point(s)
            if n_parallel > 1:
                next_points = self._parallel_acquisition(n_parallel)
            else:
                next_points = [self.surrogate.next_sample(
                    self.bounds,
                    self.acquisition
                )]

            # Evaluate objective function
            for point in next_points:
                y_new = self.objective_func(point)
                self.X_observed.append(point)
                self.y_observed.append(y_new)

                logger.debug(
                    f"Iteration {self.iteration}: "
                    f"f({point}) = {y_new:.6f} "
                    f"(best = {self.surrogate.best_value:.6f})"
                )

            self.iteration += 1

            # Call callback if provided
            if callback:
                state = SurrogateState(
                    X_observed=np.array(self.X_observed),
                    y_observed=np.array(self.y_observed),
                    best_value=self.surrogate.best_value,
                    best_point=self.surrogate.best_point,
                    n_evaluations=len(self.X_observed),
                    gp_model=self.surrogate.gp
                )
                callback(state)

        # Final fit
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.surrogate.fit(X, y)

        return {
            'best_point': self.surrogate.best_point,
            'best_value': self.surrogate.best_value,
            'n_evaluations': len(self.X_observed),
            'X_observed': X,
            'y_observed': y,
            'convergence': self._analyze_convergence()
        }

    def _initial_sampling(self):
        """Generate initial samples using Latin Hypercube or random sampling."""
        n_dim = len(self.bounds)
        n_samples = self.n_initial_points - len(self.X_observed)

        if n_samples <= 0:
            return

        # Latin Hypercube Sampling
        samples = self._latin_hypercube_sampling(n_samples, n_dim)

        # Scale to bounds
        for i in range(n_dim):
            samples[:, i] = (
                samples[:, i] * (self.bounds[i, 1] - self.bounds[i, 0]) +
                self.bounds[i, 0]
            )

        # Evaluate samples
        for point in samples:
            y = self.objective_func(point)
            self.X_observed.append(point)
            self.y_observed.append(y)

        logger.info(f"Initial sampling: evaluated {n_samples} points")

    def _latin_hypercube_sampling(
        self,
        n_samples: int,
        n_dim: int
    ) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        samples = np.zeros((n_samples, n_dim))

        for i in range(n_dim):
            # Create evenly spaced intervals
            intervals = np.linspace(0, 1, n_samples + 1)
            # Random point in each interval
            samples[:, i] = np.random.uniform(
                intervals[:-1],
                intervals[1:],
                size=n_samples
            )
            # Random permutation
            np.random.shuffle(samples[:, i])

        return samples

    def _parallel_acquisition(self, n_parallel: int) -> List[np.ndarray]:
        """
        Generate multiple acquisition points for parallel evaluation.

        Uses a strategy to ensure diversity in selected points.
        """
        points = []

        for i in range(n_parallel):
            if i == 0:
                # First point: standard acquisition
                point = self.surrogate.next_sample(
                    self.bounds,
                    self.acquisition
                )
            else:
                # Subsequent points: penalize proximity to already selected points
                point = self._diverse_acquisition(points)

            points.append(point)

        return points

    def _diverse_acquisition(
        self,
        existing_points: List[np.ndarray],
        n_candidates: int = 1000
    ) -> np.ndarray:
        """Select acquisition point with diversity penalty."""
        n_dim = len(self.bounds)

        # Generate candidates
        candidates = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_candidates, n_dim)
        )

        # Calculate acquisition values
        acq_values = self.surrogate.expected_improvement(candidates)

        # Apply diversity penalty
        for point in existing_points:
            distances = np.linalg.norm(candidates - point, axis=1)
            # Gaussian penalty
            penalty = np.exp(-distances**2 / (2 * 0.1**2))
            acq_values *= (1 - 0.5 * penalty)

        # Select best diverse candidate
        best_idx = np.argmax(acq_values)
        return candidates[best_idx]

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence of the optimization."""
        y = np.array(self.y_observed)

        # Calculate best value over time
        best_values = np.minimum.accumulate(y)

        # Calculate improvement rate
        if len(y) > 10:
            recent_improvement = best_values[-1] - best_values[-10]
            improvement_rate = abs(recent_improvement) / abs(best_values[-10] + 1e-9)
        else:
            improvement_rate = 1.0

        # Check for convergence
        converged = improvement_rate < 0.01 if len(y) > 10 else False

        return {
            'best_values': best_values.tolist(),
            'improvement_rate': improvement_rate,
            'converged': converged,
            'final_best': float(best_values[-1])
        }

    def plot_convergence(self) -> Optional[Any]:
        """Plot convergence history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            y = np.array(self.y_observed)
            best_values = np.minimum.accumulate(y)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Convergence plot
            ax1.plot(best_values, 'b-', label='Best value')
            ax1.scatter(range(len(y)), y, alpha=0.3, label='Evaluations')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Objective value')
            ax1.set_title('Convergence History')
            ax1.legend()
            ax1.grid(True)

            # GP model visualization (for 1D/2D problems)
            if self.surrogate.gp and len(self.bounds) <= 2:
                if len(self.bounds) == 1:
                    # 1D visualization
                    x_plot = np.linspace(
                        self.bounds[0, 0],
                        self.bounds[0, 1],
                        100
                    ).reshape(-1, 1)
                    mu, sigma = self.surrogate.predict(x_plot)

                    ax2.plot(x_plot, mu, 'b-', label='GP mean')
                    ax2.fill_between(
                        x_plot.flatten(),
                        mu - 2*sigma,
                        mu + 2*sigma,
                        alpha=0.3,
                        label='95% CI'
                    )
                    ax2.scatter(
                        self.X_observed,
                        self.y_observed,
                        c='r',
                        label='Observations'
                    )
                    ax2.set_xlabel('x')
                    ax2.set_ylabel('f(x)')
                    ax2.set_title('GP Surrogate Model')
                    ax2.legend()
                    ax2.grid(True)

            plt.tight_layout()
            return fig

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None