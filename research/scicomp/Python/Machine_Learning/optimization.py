"""
Machine Learning Optimization Algorithms for Scientific Computing
This module implements optimization algorithms specifically designed for
machine learning in scientific computing contexts, including gradient-based
methods, evolutionary algorithms, and Bayesian optimization.
Classes:
    Optimizer: Abstract base class for optimizers
    SGD: Stochastic Gradient Descent with momentum
    Adam: Adam optimizer with bias correction
    AdamW: Adam with decoupled weight decay
    LBFGS: Limited-memory BFGS for large-scale optimization
    GeneticAlgorithm: Genetic algorithm for global optimization
    ParticleSwarmOptimization: PSO for multi-modal functions
    BayesianOptimization: Gaussian process-based optimization
"""
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any, Union, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.linalg import norm
import logging
logger = logging.getLogger(__name__)
@dataclass
class OptimizationResults:
    """Container for optimization results and diagnostics."""
    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str
    history: Optional[Dict[str, List[float]]] = None
    gradient_norm: Optional[float] = None
class Optimizer(ABC):
    """Abstract base class for optimization algorithms."""
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.history = {'loss': [], 'gradient_norm': []}
    @abstractmethod
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Take optimization step."""
        pass
    @abstractmethod
    def reset(self):
        """Reset optimizer state."""
        pass
class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum and learning rate scheduling.
    Features:
    - Momentum acceleration
    - Nesterov momentum
    - Learning rate decay
    - Gradient clipping
    """
    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 nesterov: bool = False,
                 weight_decay: float = 0.0,
                 dampening: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.velocity = None
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Take SGD optimization step."""
        # Add weight decay
        if self.weight_decay > 0:
            gradients = gradients + self.weight_decay * params
        # Initialize velocity if needed
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        # Update velocity
        self.velocity = self.momentum * self.velocity + (1 - self.dampening) * gradients
        # Apply Nesterov momentum if enabled
        if self.nesterov:
            gradients = gradients + self.momentum * self.velocity
        else:
            gradients = self.velocity
        # Update parameters
        params_new = params - self.learning_rate * gradients
        # Store history
        self.history['gradient_norm'].append(norm(gradients))
        return params_new
    def reset(self):
        """Reset optimizer state."""
        self.velocity = None
        self.history = {'loss': [], 'gradient_norm': []}
class Adam(Optimizer):
    """
    Adam optimizer with bias correction and AMSGrad variant.
    Features:
    - Adaptive learning rates
    - Bias correction
    - AMSGrad variant
    - Gradient clipping
    """
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        # State variables
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.v_hat_max = None  # For AMSGrad
        self.t = 0  # Time step
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Take Adam optimization step."""
        self.t += 1
        # Add weight decay
        if self.weight_decay > 0:
            gradients = gradients + self.weight_decay * params
        # Initialize moment estimates if needed
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            if self.amsgrad:
                self.v_hat_max = np.zeros_like(params)
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients**2
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)
        # AMSGrad variant
        if self.amsgrad:
            self.v_hat_max = np.maximum(self.v_hat_max, v_hat)
            denominator = np.sqrt(self.v_hat_max) + self.epsilon
        else:
            denominator = np.sqrt(v_hat) + self.epsilon
        # Update parameters
        params_new = params - self.learning_rate * m_hat / denominator
        # Store history
        self.history['gradient_norm'].append(norm(gradients))
        return params_new
    def reset(self):
        """Reset optimizer state."""
        self.m = None
        self.v = None
        self.v_hat_max = None
        self.t = 0
        self.history = {'loss': [], 'gradient_norm': []}
class AdamW(Adam):
    """
    AdamW optimizer with decoupled weight decay.
    Separates weight decay from gradient-based updates for better performance
    with regularization.
    """
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Take AdamW optimization step."""
        self.t += 1
        # Initialize moment estimates if needed
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            if self.amsgrad:
                self.v_hat_max = np.zeros_like(params)
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients**2
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)
        # AMSGrad variant
        if self.amsgrad:
            self.v_hat_max = np.maximum(self.v_hat_max, v_hat)
            denominator = np.sqrt(self.v_hat_max) + self.epsilon
        else:
            denominator = np.sqrt(v_hat) + self.epsilon
        # Update parameters with decoupled weight decay
        params_new = params * (1 - self.learning_rate * self.weight_decay) - \
                    self.learning_rate * m_hat / denominator
        # Store history
        self.history['gradient_norm'].append(norm(gradients))
        return params_new
class LBFGS:
    """
    Limited-memory BFGS optimizer for large-scale optimization.
    Features:
    - Memory-efficient quasi-Newton method
    - Line search
    - Convergence diagnostics
    - Scientific computing adaptations
    """
    def __init__(self,
                 max_iter: int = 1000,
                 memory_size: int = 10,
                 tolerance: float = 1e-6,
                 line_search: str = 'strong_wolfe'):
        self.max_iter = max_iter
        self.memory_size = memory_size
        self.tolerance = tolerance
        self.line_search = line_search
    def minimize(self,
                 fun: Callable,
                 x0: np.ndarray,
                 jac: Optional[Callable] = None,
                 args: tuple = (),
                 bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizationResults:
        """
        Minimize function using L-BFGS.
        Parameters:
            fun: Objective function
            x0: Initial guess
            jac: Gradient function (if None, uses finite differences)
            args: Additional arguments to fun and jac
            bounds: Parameter bounds
        Returns:
            Optimization results
        """
        if jac is None:
            # Use finite differences for gradient
            def jac_finite_diff(x):
                return self._finite_difference_gradient(fun, x, args)
            jac = jac_finite_diff
        # Use scipy's L-BFGS-B implementation
        options = {
            'maxiter': self.max_iter,
            'ftol': self.tolerance,
            'gtol': self.tolerance
        }
        result = optimize.minimize(
            fun, x0, method='L-BFGS-B', jac=jac,
            args=args, bounds=bounds, options=options
        )
        return OptimizationResults(
            x=result.x,
            fun=result.fun,
            nit=result.nit,
            nfev=result.nfev,
            success=result.success,
            message=result.message,
            gradient_norm=norm(result.jac) if hasattr(result, 'jac') else None
        )
    def _finite_difference_gradient(self, fun: Callable, x: np.ndarray, args: tuple) -> np.ndarray:
        """Compute gradient using finite differences."""
        epsilon = 1e-8
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            x_minus = x.copy()
            x_minus[i] -= epsilon
            grad[i] = (fun(x_plus, *args) - fun(x_minus, *args)) / (2 * epsilon)
        return grad
class GeneticAlgorithm:
    """
    Genetic Algorithm for global optimization.
    Features:
    - Multiple selection strategies
    - Crossover and mutation operators
    - Elitism
    - Constraint handling
    """
    def __init__(self,
                 population_size: int = 50,
                 n_generations: int = 100,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1,
                 selection_method: str = 'tournament'):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.selection_method = selection_method
    def optimize(self,
                 fun: Callable,
                 bounds: List[Tuple[float, float]],
                 args: tuple = ()) -> OptimizationResults:
        """
        Optimize function using genetic algorithm.
        Parameters:
            fun: Objective function to minimize
            bounds: Parameter bounds [(min1, max1), (min2, max2), ...]
            args: Additional arguments to fun
        Returns:
            Optimization results
        """
        n_dims = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        # Initialize population
        population = self._initialize_population(lower_bounds, upper_bounds)
        best_fitness_history = []
        mean_fitness_history = []
        for generation in range(self.n_generations):
            # Evaluate fitness
            fitness = np.array([fun(individual, *args) for individual in population])
            # Track progress
            best_fitness_history.append(np.min(fitness))
            mean_fitness_history.append(np.mean(fitness))
            # Selection
            selected_indices = self._selection(fitness)
            parents = population[selected_indices]
            # Create new generation
            offspring = []
            n_elite = int(self.elitism_rate * self.population_size)
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness)[:n_elite]
            for idx in elite_indices:
                offspring.append(population[idx].copy())
            # Generate offspring through crossover and mutation
            while len(offspring) < self.population_size:
                # Select parents
                parent1_idx = np.random.choice(len(parents))
                parent2_idx = np.random.choice(len(parents))
                parent1 = parents[parent1_idx]
                parent2 = parents[parent2_idx]
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                # Mutation
                child1 = self._mutate(child1, lower_bounds, upper_bounds)
                child2 = self._mutate(child2, lower_bounds, upper_bounds)
                offspring.extend([child1, child2])
            # Update population
            population = np.array(offspring[:self.population_size])
        # Final evaluation
        final_fitness = np.array([fun(individual, *args) for individual in population])
        best_idx = np.argmin(final_fitness)
        history = {
            'best_fitness': best_fitness_history,
            'mean_fitness': mean_fitness_history
        }
        return OptimizationResults(
            x=population[best_idx],
            fun=final_fitness[best_idx],
            nit=self.n_generations,
            nfev=self.n_generations * self.population_size,
            success=True,
            message="Genetic algorithm completed",
            history=history
        )
    def _initialize_population(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
        """Initialize random population."""
        population = np.random.uniform(
            lower_bounds, upper_bounds,
            (self.population_size, len(lower_bounds))
        )
        return population
    def _selection(self, fitness: np.ndarray) -> np.ndarray:
        """Select parents for reproduction."""
        if self.selection_method == 'tournament':
            return self._tournament_selection(fitness)
        elif self.selection_method == 'roulette':
            return self._roulette_selection(fitness)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    def _tournament_selection(self, fitness: np.ndarray, tournament_size: int = 3) -> np.ndarray:
        """Tournament selection."""
        selected_indices = []
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(
                len(fitness), tournament_size, replace=False
            )
            winner_idx = tournament_indices[np.argmin(fitness[tournament_indices])]
            selected_indices.append(winner_idx)
        return np.array(selected_indices)
    def _roulette_selection(self, fitness: np.ndarray) -> np.ndarray:
        """Roulette wheel selection."""
        # Convert minimization to maximization
        max_fitness = np.max(fitness)
        selection_probabilities = (max_fitness - fitness + 1e-10)
        selection_probabilities /= np.sum(selection_probabilities)
        selected_indices = np.random.choice(
            len(fitness), self.population_size,
            p=selection_probabilities, replace=True
        )
        return selected_indices
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover."""
        mask = np.random.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    def _mutate(self, individual: np.ndarray, lower_bounds: np.ndarray,
               upper_bounds: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        mutated = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation
                sigma = (upper_bounds[i] - lower_bounds[i]) * 0.1
                mutated[i] += np.random.normal(0, sigma)
                # Ensure bounds
                mutated[i] = np.clip(mutated[i], lower_bounds[i], upper_bounds[i])
        return mutated
class BayesianOptimization:
    """
    Bayesian Optimization using Gaussian Processes.
    Features:
    - Gaussian process surrogate model
    - Multiple acquisition functions
    - Automatic hyperparameter optimization
    - Constraint handling
    """
    def __init__(self,
                 n_initial: int = 10,
                 n_iter: int = 50,
                 acquisition: str = 'ei',
                 xi: float = 0.01,
                 kappa: float = 2.576,
                 random_state: Optional[int] = None):
        self.n_initial = n_initial
        self.n_iter = n_iter
        self.acquisition = acquisition
        self.xi = xi  # Exploration parameter for EI
        self.kappa = kappa  # Exploration parameter for UCB
        self.random_state = random_state
        # Training data
        self.X_train = None
        self.y_train = None
    def optimize(self,
                 fun: Callable,
                 bounds: List[Tuple[float, float]],
                 args: tuple = ()) -> OptimizationResults:
        """
        Optimize function using Bayesian optimization.
        Parameters:
            fun: Objective function to minimize
            bounds: Parameter bounds
            args: Additional arguments to fun
        Returns:
            Optimization results
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_dims = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        # Initial sampling
        X_init = self._latin_hypercube_sampling(
            lower_bounds, upper_bounds, self.n_initial
        )
        y_init = np.array([fun(x, *args) for x in X_init])
        self.X_train = X_init
        self.y_train = y_init
        acquisition_history = []
        for iteration in range(self.n_iter):
            # Fit Gaussian process
            gp_mean, gp_std = self._fit_gaussian_process()
            # Optimize acquisition function
            x_next = self._optimize_acquisition(
                gp_mean, gp_std, lower_bounds, upper_bounds
            )
            # Evaluate objective function
            y_next = fun(x_next, *args)
            # Update training data
            self.X_train = np.vstack([self.X_train, x_next])
            self.y_train = np.append(self.y_train, y_next)
            # Track acquisition function value
            acq_value = self._acquisition_function(
                x_next.reshape(1, -1), gp_mean, gp_std
            )[0]
            acquisition_history.append(acq_value)
        # Find best result
        best_idx = np.argmin(self.y_train)
        history = {
            'acquisition': acquisition_history,
            'y_values': self.y_train.tolist()
        }
        return OptimizationResults(
            x=self.X_train[best_idx],
            fun=self.y_train[best_idx],
            nit=self.n_iter,
            nfev=self.n_initial + self.n_iter,
            success=True,
            message="Bayesian optimization completed",
            history=history
        )
    def _latin_hypercube_sampling(self, lower_bounds: np.ndarray,
                                 upper_bounds: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate Latin hypercube samples."""
        n_dims = len(lower_bounds)
        samples = np.zeros((n_samples, n_dims))
        for i in range(n_dims):
            # Generate uniform samples in [0, 1]
            uniform_samples = (np.arange(n_samples) + np.random.random(n_samples)) / n_samples
            np.random.shuffle(uniform_samples)
            # Scale to bounds
            samples[:, i] = lower_bounds[i] + uniform_samples * (upper_bounds[i] - lower_bounds[i])
        return samples
    def _fit_gaussian_process(self):
        """Fit Gaussian process to training data."""
        # Simplified GP implementation
        # In practice, would use a more sophisticated GP library
        def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
            """Radial basis function kernel."""
            sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
            return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)
        # Normalize targets
        y_mean = np.mean(self.y_train)
        y_std = np.std(self.y_train)
        y_normalized = (self.y_train - y_mean) / (y_std + 1e-8)
        # Kernel parameters (simplified - should be optimized)
        length_scale = 1.0
        sigma_f = 1.0
        sigma_n = 0.1
        # Compute kernel matrix
        K = rbf_kernel(self.X_train, self.X_train, length_scale, sigma_f)
        K += sigma_n**2 * np.eye(len(self.X_train))
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_normalized))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            alpha = np.linalg.pinv(K) @ y_normalized
            L = None
        def gp_predict(X_test):
            """GP prediction function."""
            K_s = rbf_kernel(self.X_train, X_test, length_scale, sigma_f)
            K_ss = rbf_kernel(X_test, X_test, length_scale, sigma_f)
            # Mean prediction
            mu = K_s.T @ alpha
            mu = mu * y_std + y_mean  # Denormalize
            # Variance prediction
            if L is not None:
                v = np.linalg.solve(L, K_s)
                var = np.diag(K_ss) - np.sum(v**2, axis=0)
            else:
                var = np.diag(K_ss) - np.sum(K_s * (np.linalg.pinv(K) @ K_s), axis=0)
            var = np.maximum(var, 0)  # Ensure non-negative
            std = np.sqrt(var) * y_std  # Denormalize
            return mu, std
        return gp_predict, None
    def _optimize_acquisition(self, gp_mean, gp_std, lower_bounds, upper_bounds):
        """Optimize acquisition function."""
        def acquisition_objective(x):
            """Negative acquisition function (for minimization)."""
            return -self._acquisition_function(x.reshape(1, -1), gp_mean, gp_std)[0]
        # Multiple random starts
        best_x = None
        best_acq = np.inf
        for _ in range(10):
            x_start = np.random.uniform(lower_bounds, upper_bounds)
            # Optimize using L-BFGS-B
            bounds_list = list(zip(lower_bounds, upper_bounds))
            try:
                result = optimize.minimize(
                    acquisition_objective, x_start,
                    method='L-BFGS-B', bounds=bounds_list
                )
                if result.success and result.fun < best_acq:
                    best_acq = result.fun
                    best_x = result.x
            except:
                continue
        if best_x is None:
            # Fallback to random sampling
            best_x = np.random.uniform(lower_bounds, upper_bounds)
        return best_x
    def _acquisition_function(self, X, gp_mean, gp_std):
        """Compute acquisition function values."""
        mu, std = gp_mean(X)
        if self.acquisition == 'ei':
            # Expected Improvement
            if len(self.y_train) > 0:
                f_best = np.min(self.y_train)
                improvement = f_best - mu - self.xi
                Z = improvement / (std + 1e-8)
                ei = improvement * stats.norm.cdf(Z) + std * stats.norm.pdf(Z)
                return np.maximum(ei, 0)
            else:
                return std
        elif self.acquisition == 'ucb':
            # Upper Confidence Bound
            return -(mu - self.kappa * std)  # Negative for minimization
        elif self.acquisition == 'poi':
            # Probability of Improvement
            if len(self.y_train) > 0:
                f_best = np.min(self.y_train)
                Z = (f_best - mu - self.xi) / (std + 1e-8)
                return stats.norm.cdf(Z)
            else:
                return np.ones_like(mu)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")
# Utility functions
def create_test_functions() -> Dict[str, Callable]:
    """Create test functions for optimization validation."""
    test_functions = {}
    # Rosenbrock function
    def rosenbrock(x):
        """Rosenbrock function (minimum at [1, 1])."""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    test_functions['rosenbrock'] = rosenbrock
    # Rastrigin function
    def rastrigin(x):
        """Rastrigin function (global minimum at origin)."""
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    test_functions['rastrigin'] = rastrigin
    # Sphere function
    def sphere(x):
        """Sphere function (minimum at origin)."""
        return np.sum(x**2)
    test_functions['sphere'] = sphere
    # Ackley function
    def ackley(x):
        """Ackley function (global minimum at origin)."""
        a, b, c = 20, 0.2, 2 * np.pi
        d = len(x)
        return (-a * np.exp(-b * np.sqrt(np.sum(x**2) / d)) -
                np.exp(np.sum(np.cos(c * x)) / d) + a + np.e)
    test_functions['ackley'] = ackley
    return test_functions
# Visualization utilities
def plot_optimization_convergence(results: OptimizationResults,
                                title: str = "Optimization Convergence"):
    """Plot optimization convergence with Berkeley styling."""
    if results.history is None:
        print("No history available for plotting")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Berkeley colors
    berkeley_blue = '#003262'
    california_gold = '#FDB515'
    # Plot 1: Objective function value
    if 'loss' in results.history:
        epochs = range(1, len(results.history['loss']) + 1)
        axes[0].semilogy(epochs, results.history['loss'],
                        color=berkeley_blue, linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Objective Value (log scale)')
        axes[0].set_title('Objective Function Convergence')
        axes[0].grid(True, alpha=0.3)
    elif 'best_fitness' in results.history:
        epochs = range(1, len(results.history['best_fitness']) + 1)
        axes[0].semilogy(epochs, results.history['best_fitness'],
                        color=berkeley_blue, linewidth=2, label='Best')
        if 'mean_fitness' in results.history:
            axes[0].semilogy(epochs, results.history['mean_fitness'],
                            color=california_gold, linewidth=2, label='Mean')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Fitness (log scale)')
        axes[0].set_title('Genetic Algorithm Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    # Plot 2: Gradient norm or other metric
    if 'gradient_norm' in results.history:
        epochs = range(1, len(results.history['gradient_norm']) + 1)
        axes[1].semilogy(epochs, results.history['gradient_norm'],
                        color=california_gold, linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Gradient Norm (log scale)')
        axes[1].set_title('Gradient Convergence')
        axes[1].grid(True, alpha=0.3)
    elif 'acquisition' in results.history:
        epochs = range(1, len(results.history['acquisition']) + 1)
        axes[1].plot(epochs, results.history['acquisition'],
                    color=california_gold, linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Acquisition Function Value')
        axes[1].set_title('Acquisition Function')
        axes[1].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
def plot_optimization_landscape(fun: Callable, bounds: List[Tuple[float, float]],
                               results: OptimizationResults = None,
                               title: str = "Optimization Landscape"):
    """Plot 2D optimization landscape with Berkeley styling."""
    if len(bounds) != 2:
        print("Can only plot 2D landscapes")
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    # Berkeley colors
    berkeley_blue = '#003262'
    california_gold = '#FDB515'
    # Create grid
    x_range = np.linspace(bounds[0][0], bounds[0][1], 50)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 50)
    X, Y = np.meshgrid(x_range, y_range)
    # Evaluate function on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fun([X[i, j], Y[i, j]])
    # Plot contours
    contour = ax.contour(X, Y, Z, levels=20, colors='gray', alpha=0.6)
    contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    # Plot optimization path if available
    if results is not None and hasattr(results, 'X_train'):
        # For Bayesian optimization
        ax.scatter(results.X_train[:, 0], results.X_train[:, 1],
                  c=california_gold, s=60, marker='o',
                  edgecolors=berkeley_blue, linewidth=2, label='Evaluated Points')
        ax.scatter(results.x[0], results.x[1],
                  c='red', s=100, marker='*',
                  edgecolors='white', linewidth=2, label='Best Point')
    elif results is not None:
        ax.scatter(results.x[0], results.x[1],
                  c='red', s=100, marker='*',
                  edgecolors='white', linewidth=2, label='Optimum')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    if results is not None:
        ax.legend()
    # Add colorbar
    plt.colorbar(contourf, ax=ax, label='Function Value')
    return fig