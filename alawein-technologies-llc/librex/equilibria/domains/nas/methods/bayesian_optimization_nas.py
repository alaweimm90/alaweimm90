"""Bayesian Optimization for Neural Architecture Search.

Uses Gaussian Process surrogate models to efficiently explore the architecture space.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
import warnings
from scipy.stats import norm
from scipy.optimize import minimize

from ..architecture import NASCell, MacroArchitecture
from ..nas_problem import NASProblem
from ..nas_adapter import NASAdapter


@dataclass
class ArchitectureEvaluation:
    """Record of an evaluated architecture."""
    encoding: np.ndarray
    architecture: Union[NASCell, MacroArchitecture]
    objective: float
    metrics: Dict[str, float]


class GaussianProcessSurrogate:
    """
    Gaussian Process surrogate model for architecture performance prediction.

    Simple GP implementation for demonstration purposes.
    """

    def __init__(self,
                 kernel: str = 'rbf',
                 length_scale: float = 1.0,
                 noise: float = 1e-6):
        """
        Initialize GP surrogate.

        Args:
            kernel: Kernel type ('rbf', 'matern')
            length_scale: Length scale for kernel
            noise: Observation noise
        """
        self.kernel = kernel
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF (Radial Basis Function) kernel.

        Args:
            X1: First set of points (n1, d)
            X2: Second set of points (n2, d)

        Returns:
            Kernel matrix (n1, n2)
        """
        # Ensure 2D arrays
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)

        # Compute squared Euclidean distances
        sqdist = np.sum(X1**2, axis=1, keepdims=True) + \
                np.sum(X2**2, axis=1, keepdims=True).T - \
                2 * np.dot(X1, X2.T)

        # RBF kernel
        return np.exp(-0.5 * sqdist / self.length_scale**2)

    def matern_kernel(self, X1: np.ndarray, X2: np.ndarray, nu: float = 1.5) -> np.ndarray:
        """
        Matern kernel.

        Args:
            X1: First set of points
            X2: Second set of points
            nu: Smoothness parameter

        Returns:
            Kernel matrix
        """
        # Ensure 2D arrays
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)

        # Compute distances
        dists = np.sqrt(np.sum(X1**2, axis=1, keepdims=True) +
                       np.sum(X2**2, axis=1, keepdims=True).T -
                       2 * np.dot(X1, X2.T) + 1e-8)

        # Matern kernel (nu=1.5)
        if nu == 0.5:
            K = np.exp(-dists / self.length_scale)
        elif nu == 1.5:
            K = (1 + np.sqrt(3) * dists / self.length_scale) * \
                np.exp(-np.sqrt(3) * dists / self.length_scale)
        else:  # nu == 2.5
            K = (1 + np.sqrt(5) * dists / self.length_scale +
                 5 * dists**2 / (3 * self.length_scale**2)) * \
                np.exp(-np.sqrt(5) * dists / self.length_scale)

        return K

    def compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix based on selected kernel."""
        if self.kernel == 'rbf':
            return self.rbf_kernel(X1, X2)
        elif self.kernel == 'matern':
            return self.matern_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit GP to observed data.

        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        self.X_train = X
        self.y_train = y

        # Compute kernel matrix
        K = self.compute_kernel(X, X)
        K += self.noise * np.eye(len(X))

        # Compute inverse for predictions
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Add more noise if singular
            K += 1e-4 * np.eye(len(X))
            self.K_inv = np.linalg.inv(K)

    def predict(self, X: np.ndarray, return_std: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict mean and optionally standard deviation.

        Args:
            X: Test points (n_test, n_features)
            return_std: Whether to return standard deviation

        Returns:
            Mean predictions and optionally standard deviations
        """
        if self.X_train is None:
            raise ValueError("GP not fitted yet")

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Compute kernel between test and train
        K_star = self.compute_kernel(X, self.X_train)
        K_star_star = self.compute_kernel(X, X)

        # Mean prediction
        mean = K_star @ self.K_inv @ self.y_train

        if return_std:
            # Variance prediction
            var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
            std = np.sqrt(np.maximum(var, 0))
            return mean, std
        else:
            return mean


class AcquisitionFunction:
    """Acquisition functions for Bayesian optimization."""

    @staticmethod
    def expected_improvement(mean: np.ndarray,
                           std: np.ndarray,
                           best_value: float,
                           xi: float = 0.01) -> np.ndarray:
        """
        Expected Improvement acquisition function.

        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            best_value: Current best observed value
            xi: Exploration parameter

        Returns:
            EI values
        """
        imp = mean - best_value - xi
        Z = imp / (std + 1e-9)
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0
        return ei

    @staticmethod
    def upper_confidence_bound(mean: np.ndarray,
                              std: np.ndarray,
                              beta: float = 2.0) -> np.ndarray:
        """
        Upper Confidence Bound acquisition function.

        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            beta: Exploration parameter

        Returns:
            UCB values
        """
        return mean + beta * std

    @staticmethod
    def probability_of_improvement(mean: np.ndarray,
                                  std: np.ndarray,
                                  best_value: float,
                                  xi: float = 0.01) -> np.ndarray:
        """
        Probability of Improvement acquisition function.

        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            best_value: Current best observed value
            xi: Exploration parameter

        Returns:
            PI values
        """
        Z = (mean - best_value - xi) / (std + 1e-9)
        return norm.cdf(Z)


class BayesianOptimizationNAS:
    """
    Bayesian Optimization for Neural Architecture Search.

    Uses GP surrogate to model architecture performance and acquisition
    functions to guide search.
    """

    def __init__(self,
                 problem: NASProblem,
                 acquisition: str = 'ei',
                 n_initial: int = 20,
                 kernel: str = 'matern'):
        """
        Initialize Bayesian Optimization NAS.

        Args:
            problem: NAS problem to solve
            acquisition: Acquisition function ('ei', 'ucb', 'pi')
            n_initial: Number of initial random evaluations
            kernel: GP kernel type
        """
        self.problem = problem
        self.acquisition = acquisition
        self.n_initial = n_initial

        self.adapter = NASAdapter()
        self.surrogate = GaussianProcessSurrogate(kernel=kernel)
        self.evaluations: List[ArchitectureEvaluation] = []

        # Get problem bounds
        self.bounds = np.array(problem.get_bounds())
        self.dimension = problem.get_dimension()

    def random_architecture(self) -> np.ndarray:
        """Generate random architecture encoding."""
        encoding = np.zeros(self.dimension)
        for i, (low, high) in enumerate(self.bounds):
            encoding[i] = np.random.uniform(low, high)
        return encoding

    def initial_sampling(self):
        """Perform initial random sampling."""
        print(f"Initial random sampling: {self.n_initial} architectures")

        for i in range(self.n_initial):
            # Random architecture
            encoding = self.random_architecture()

            # Decode and evaluate
            architecture = self.adapter.decode_solution(encoding, self.problem)
            metrics = self.problem.evaluate_architecture(architecture, return_all_metrics=True)

            # Store evaluation
            self.evaluations.append(ArchitectureEvaluation(
                encoding=encoding,
                architecture=architecture,
                objective=metrics.get('objective', metrics.get('accuracy', 0)),
                metrics=metrics
            ))

            print(f"  Sample {i+1}/{self.n_initial}: objective = {self.evaluations[-1].objective:.4f}")

    def optimize_acquisition(self) -> np.ndarray:
        """
        Optimize acquisition function to find next point to evaluate.

        Returns:
            Next point to evaluate
        """
        # Get current data
        X = np.array([e.encoding for e in self.evaluations])
        y = np.array([e.objective for e in self.evaluations])

        # Fit surrogate
        self.surrogate.fit(X, y)

        # Best observed value
        best_value = np.max(y)

        # Define acquisition function to maximize
        def neg_acquisition(x):
            """Negative acquisition for minimization."""
            mean, std = self.surrogate.predict(x.reshape(1, -1))

            if self.acquisition == 'ei':
                acq = AcquisitionFunction.expected_improvement(mean, std, best_value)
            elif self.acquisition == 'ucb':
                acq = AcquisitionFunction.upper_confidence_bound(mean, std)
            elif self.acquisition == 'pi':
                acq = AcquisitionFunction.probability_of_improvement(mean, std, best_value)
            else:
                raise ValueError(f"Unknown acquisition function: {self.acquisition}")

            return -acq[0]  # Negative for minimization

        # Multi-start optimization
        best_x = None
        best_acq = float('inf')

        for _ in range(10):  # 10 random starts
            x0 = self.random_architecture()

            # Optimize
            result = minimize(
                neg_acquisition,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds
            )

            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x

        return best_x

    def run(self, n_iterations: int = 100) -> Dict[str, Any]:
        """
        Run Bayesian Optimization NAS.

        Args:
            n_iterations: Total number of architecture evaluations

        Returns:
            Results dictionary
        """
        # Initial sampling
        self.initial_sampling()

        # Bayesian optimization loop
        for iteration in range(self.n_initial, n_iterations):
            print(f"\nIteration {iteration + 1}/{n_iterations}")

            # Find next architecture to evaluate
            next_encoding = self.optimize_acquisition()

            # Decode and evaluate
            architecture = self.adapter.decode_solution(next_encoding, self.problem)
            metrics = self.problem.evaluate_architecture(architecture, return_all_metrics=True)

            # Store evaluation
            self.evaluations.append(ArchitectureEvaluation(
                encoding=next_encoding,
                architecture=architecture,
                objective=metrics.get('objective', metrics.get('accuracy', 0)),
                metrics=metrics
            ))

            # Report progress
            best_so_far = max(self.evaluations, key=lambda e: e.objective)
            print(f"  New objective: {self.evaluations[-1].objective:.4f}")
            print(f"  Best so far: {best_so_far.objective:.4f}")

            # Optional: Early stopping if budget exhausted
            if self.problem.evaluation_count >= self.problem.max_evaluations:
                print("Evaluation budget exhausted")
                break

        # Find best architecture
        best_eval = max(self.evaluations, key=lambda e: e.objective)

        # Compute convergence history
        convergence = []
        best_val = float('-inf')
        for eval in self.evaluations:
            best_val = max(best_val, eval.objective)
            convergence.append(best_val)

        return {
            'best_architecture': best_eval.architecture,
            'best_encoding': best_eval.encoding,
            'best_objective': best_eval.objective,
            'best_metrics': best_eval.metrics,
            'all_evaluations': self.evaluations,
            'convergence': convergence,
            'surrogate': self.surrogate
        }


def bayesian_optimization_nas(problem: NASProblem, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Bayesian Optimization NAS.

    Args:
        problem: NAS problem to solve
        config: Configuration dictionary

    Returns:
        Results dictionary
    """
    bo_nas = BayesianOptimizationNAS(
        problem=problem,
        acquisition=config.get('acquisition', 'ei'),
        n_initial=config.get('n_initial', 20),
        kernel=config.get('kernel', 'matern')
    )

    return bo_nas.run(n_iterations=config.get('n_iterations', 100))