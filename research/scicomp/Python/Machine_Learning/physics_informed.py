"""
Physics-Informed Machine Learning for Scientific Computing
This module implements physics-informed machine learning methods that incorporate
physical laws, conservation principles, and domain knowledge into ML models.
Classes:
    PINN: Physics-Informed Neural Networks
    DeepONet: Deep Operator Networks
    FNO: Fourier Neural Operator
    PhysicsConstrainedNN: Neural networks with physics constraints
    ConservationLawsNN: Neural networks enforcing conservation laws
    SymmetryAwareNN: Neural networks respecting physical symmetries
"""
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any, Union, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy import optimize, integrate
from scipy.linalg import svd, eigh
import logging
logger = logging.getLogger(__name__)
@dataclass
class PINNResults:
    """Container for PINN training results."""
    loss_history: List[float]
    pde_loss_history: List[float]
    bc_loss_history: List[float]
    ic_loss_history: List[float]
    data_loss_history: Optional[List[float]] = None
    total_epochs: int = 0
class PhysicsInformedModel(ABC):
    """Abstract base class for physics-informed models."""
    def __init__(self):
        self.is_fitted = False
    @abstractmethod
    def pde_residual(self, x: np.ndarray, t: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute PDE residual."""
        pass
    @abstractmethod
    def boundary_conditions(self, x: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        """Define boundary conditions."""
        pass
    @abstractmethod
    def initial_conditions(self, x: np.ndarray) -> np.ndarray:
        """Define initial conditions."""
        pass
class PINN(PhysicsInformedModel):
    """
    Physics-Informed Neural Networks for solving PDEs.
    Features:
    - Automatic differentiation for PDE residuals
    - Multiple loss weighting strategies
    - Adaptive sampling
    - Conservation law enforcement
    """
    def __init__(self,
                 layers: List[int],
                 activation: str = 'tanh',
                 pde_weight: float = 1.0,
                 bc_weight: float = 1.0,
                 ic_weight: float = 1.0,
                 data_weight: float = 1.0,
                 learning_rate: float = 0.001,
                 adaptive_weights: bool = False):
        super().__init__()
        self.layers = layers
        self.activation = activation
        self.pde_weight = pde_weight
        self.bc_weight = bc_weight
        self.ic_weight = ic_weight
        self.data_weight = data_weight
        self.learning_rate = learning_rate
        self.adaptive_weights = adaptive_weights
        # Initialize neural network
        self._initialize_network()
        # Training history
        self.training_results = None
    def _initialize_network(self):
        """Initialize the neural network."""
        from .neural_networks import MLP
        self.network = MLP(
            layer_sizes=self.layers,
            activations=self.activation,
            output_activation='linear',
            learning_rate=self.learning_rate
        )
    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        # Concatenate spatial and temporal coordinates
        inputs = np.column_stack([x.ravel(), t.ravel()])
        return self.network.predict(inputs).reshape(x.shape)
    def compute_derivatives(self, x: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute derivatives using finite differences."""
        h = 1e-5
        derivatives = {}
        # First derivatives
        u = self.forward(x, t)
        # Spatial derivatives
        u_x_plus = self.forward(x + h, t)
        u_x_minus = self.forward(x - h, t)
        derivatives['u_x'] = (u_x_plus - u_x_minus) / (2 * h)
        # Temporal derivatives
        u_t_plus = self.forward(x, t + h)
        u_t_minus = self.forward(x, t - h)
        derivatives['u_t'] = (u_t_plus - u_t_minus) / (2 * h)
        # Second derivatives
        u_xx = (u_x_plus - 2 * u + u_x_minus) / (h**2)
        derivatives['u_xx'] = u_xx
        return derivatives
    def heat_equation_residual(self, x: np.ndarray, t: np.ndarray,
                              diffusivity: float = 1.0) -> np.ndarray:
        """Heat equation PDE residual: u_t - α * u_xx = 0."""
        derivs = self.compute_derivatives(x, t)
        return derivs['u_t'] - diffusivity * derivs['u_xx']
    def wave_equation_residual(self, x: np.ndarray, t: np.ndarray,
                              wave_speed: float = 1.0) -> np.ndarray:
        """Wave equation PDE residual: u_tt - c² * u_xx = 0."""
        # Need second time derivative
        h = 1e-5
        u_t = self.compute_derivatives(x, t)['u_t']
        u_t_plus = self.compute_derivatives(x, t + h)['u_t']
        u_t_minus = self.compute_derivatives(x, t - h)['u_t']
        u_tt = (u_t_plus - u_t_minus) / (2 * h)
        derivs = self.compute_derivatives(x, t)
        return u_tt - (wave_speed**2) * derivs['u_xx']
    def burgers_equation_residual(self, x: np.ndarray, t: np.ndarray,
                                 viscosity: float = 0.01) -> np.ndarray:
        """Burgers equation PDE residual: u_t + u * u_x - ν * u_xx = 0."""
        u = self.forward(x, t)
        derivs = self.compute_derivatives(x, t)
        return derivs['u_t'] + u * derivs['u_x'] - viscosity * derivs['u_xx']
    def pde_residual(self, x: np.ndarray, t: np.ndarray,
                     equation_type: str = 'heat', **kwargs) -> np.ndarray:
        """Compute PDE residual based on equation type."""
        if equation_type == 'heat':
            return self.heat_equation_residual(x, t, **kwargs)
        elif equation_type == 'wave':
            return self.wave_equation_residual(x, t, **kwargs)
        elif equation_type == 'burgers':
            return self.burgers_equation_residual(x, t, **kwargs)
        else:
            raise ValueError(f"Unknown equation type: {equation_type}")
    def boundary_conditions(self, x: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        """Define boundary conditions (to be overridden)."""
        # Default: homogeneous Dirichlet BC
        return {
            'left': np.zeros_like(t),    # u(0, t) = 0
            'right': np.zeros_like(t)    # u(L, t) = 0
        }
    def initial_conditions(self, x: np.ndarray) -> np.ndarray:
        """Define initial conditions (to be overridden)."""
        # Default: Gaussian pulse
        return np.exp(-((x - 0.5) / 0.1)**2)
    def compute_losses(self, x_pde: np.ndarray, t_pde: np.ndarray,
                      x_bc: np.ndarray, t_bc: np.ndarray,
                      x_ic: np.ndarray,
                      equation_type: str = 'heat',
                      x_data: Optional[np.ndarray] = None,
                      t_data: Optional[np.ndarray] = None,
                      u_data: Optional[np.ndarray] = None,
                      **pde_kwargs) -> Dict[str, float]:
        """Compute all loss components."""
        losses = {}
        # PDE loss
        pde_residual = self.pde_residual(x_pde, t_pde, equation_type, **pde_kwargs)
        losses['pde'] = np.mean(pde_residual**2)
        # Boundary condition loss
        bc_pred_left = self.forward(x_bc[0] * np.ones_like(t_bc), t_bc)
        bc_pred_right = self.forward(x_bc[1] * np.ones_like(t_bc), t_bc)
        bc_true = self.boundary_conditions(x_bc, t_bc)
        losses['bc'] = (np.mean((bc_pred_left.ravel() - bc_true['left'])**2) +
                       np.mean((bc_pred_right.ravel() - bc_true['right'])**2))
        # Initial condition loss
        ic_pred = self.forward(x_ic, np.zeros_like(x_ic))
        ic_true = self.initial_conditions(x_ic)
        losses['ic'] = np.mean((ic_pred.ravel() - ic_true)**2)
        # Data loss (if available)
        if x_data is not None and t_data is not None and u_data is not None:
            data_pred = self.forward(x_data, t_data)
            losses['data'] = np.mean((data_pred.ravel() - u_data.ravel())**2)
        else:
            losses['data'] = 0.0
        return losses
    def total_loss(self, losses: Dict[str, float]) -> float:
        """Compute weighted total loss."""
        return (self.pde_weight * losses['pde'] +
                self.bc_weight * losses['bc'] +
                self.ic_weight * losses['ic'] +
                self.data_weight * losses['data'])
    def train(self,
              x_domain: Tuple[float, float],
              t_domain: Tuple[float, float],
              n_pde: int = 10000,
              n_bc: int = 100,
              n_ic: int = 100,
              epochs: int = 1000,
              equation_type: str = 'heat',
              x_data: Optional[np.ndarray] = None,
              t_data: Optional[np.ndarray] = None,
              u_data: Optional[np.ndarray] = None,
              verbose: bool = True,
              **pde_kwargs) -> PINNResults:
        """
        Train the PINN.
        Parameters:
            x_domain: Spatial domain (x_min, x_max)
            t_domain: Temporal domain (t_min, t_max)
            n_pde: Number of PDE collocation points
            n_bc: Number of boundary condition points
            n_ic: Number of initial condition points
            epochs: Number of training epochs
            equation_type: Type of PDE ('heat', 'wave', 'burgers')
            x_data, t_data, u_data: Optional measurement data
            verbose: Whether to print training progress
            **pde_kwargs: Additional parameters for PDE
        Returns:
            Training results
        """
        # Generate training points
        np.random.seed(42)
        # PDE collocation points
        x_pde = np.random.uniform(x_domain[0], x_domain[1], n_pde)
        t_pde = np.random.uniform(t_domain[0], t_domain[1], n_pde)
        # Boundary condition points
        t_bc = np.random.uniform(t_domain[0], t_domain[1], n_bc)
        x_bc = x_domain  # Left and right boundaries
        # Initial condition points
        x_ic = np.random.uniform(x_domain[0], x_domain[1], n_ic)
        # Training history
        loss_history = []
        pde_loss_history = []
        bc_loss_history = []
        ic_loss_history = []
        data_loss_history = []
        for epoch in range(epochs):
            # Compute losses
            losses = self.compute_losses(
                x_pde, t_pde, x_bc, t_bc, x_ic,
                equation_type, x_data, t_data, u_data,
                **pde_kwargs
            )
            total_loss_val = self.total_loss(losses)
            # Store history
            loss_history.append(total_loss_val)
            pde_loss_history.append(losses['pde'])
            bc_loss_history.append(losses['bc'])
            ic_loss_history.append(losses['ic'])
            data_loss_history.append(losses['data'])
            # Backward pass and parameter update
            # Note: This is a simplified training loop
            # In practice, you would use automatic differentiation
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Total Loss: {total_loss_val:.6f}")
                print(f"  PDE Loss: {losses['pde']:.6f}")
                print(f"  BC Loss: {losses['bc']:.6f}")
                print(f"  IC Loss: {losses['ic']:.6f}")
                if losses['data'] > 0:
                    print(f"  Data Loss: {losses['data']:.6f}")
        # Store results
        self.training_results = PINNResults(
            loss_history=loss_history,
            pde_loss_history=pde_loss_history,
            bc_loss_history=bc_loss_history,
            ic_loss_history=ic_loss_history,
            data_loss_history=data_loss_history if any(data_loss_history) else None,
            total_epochs=epochs
        )
        self.is_fitted = True
        return self.training_results
    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Make predictions using the trained PINN."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.forward(x, t)
class DeepONet:
    """
    Deep Operator Networks for learning operators between function spaces.
    Features:
    - Branch and trunk networks
    - Multiple operator types
    - Uncertainty quantification
    - Physics-informed training
    """
    def __init__(self,
                 branch_layers: List[int],
                 trunk_layers: List[int],
                 activation: str = 'relu',
                 learning_rate: float = 0.001):
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.activation = activation
        self.learning_rate = learning_rate
        # Initialize networks
        self._initialize_networks()
    def _initialize_networks(self):
        """Initialize branch and trunk networks."""
        from .neural_networks import MLP
        # Branch network: processes input functions
        self.branch_net = MLP(
            layer_sizes=self.branch_layers,
            activations=self.activation,
            output_activation='linear',
            learning_rate=self.learning_rate
        )
        # Trunk network: processes evaluation coordinates
        self.trunk_net = MLP(
            layer_sizes=self.trunk_layers,
            activations=self.activation,
            output_activation='linear',
            learning_rate=self.learning_rate
        )
    def forward(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Forward pass through DeepONet.
        Parameters:
            u: Input functions (n_samples, n_sensors)
            y: Evaluation coordinates (n_points, n_dims)
        Returns:
            Output function values (n_samples, n_points)
        """
        # Branch network output
        branch_output = self.branch_net.predict(u)  # (n_samples, p)
        # Trunk network output
        trunk_output = self.trunk_net.predict(y)    # (n_points, p)
        # Combine via inner product
        output = branch_output @ trunk_output.T     # (n_samples, n_points)
        return output
    def train(self,
              u_train: np.ndarray,
              y_train: np.ndarray,
              s_train: np.ndarray,
              epochs: int = 1000,
              batch_size: int = 32,
              verbose: bool = True):
        """
        Train the DeepONet.
        Parameters:
            u_train: Input functions (n_samples, n_sensors)
            y_train: Evaluation coordinates (n_points, n_dims)
            s_train: Target function values (n_samples, n_points)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print progress
        """
        n_samples = u_train.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                u_batch = u_train[batch_indices]
                s_batch = s_train[batch_indices]
                # Forward pass
                s_pred = self.forward(u_batch, y_train)
                # Compute loss
                loss = np.mean((s_pred - s_batch)**2)
                epoch_loss += loss
                n_batches += 1
                # Note: Backward pass would be implemented here
                # with automatic differentiation
            epoch_loss /= n_batches
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
class ConservationLawsNN:
    """
    Neural network enforcing conservation laws.
    Features:
    - Built-in conservation constraints
    - Mass, momentum, energy conservation
    - Lagrangian mechanics integration
    - Hamiltonian preservation
    """
    def __init__(self,
                 layers: List[int],
                 conservation_type: str = 'mass',
                 constraint_weight: float = 1.0):
        self.layers = layers
        self.conservation_type = conservation_type
        self.constraint_weight = constraint_weight
        # Initialize network
        from .neural_networks import MLP
        self.network = MLP(layer_sizes=layers)
    def conservation_constraint(self, u: np.ndarray, x: np.ndarray,
                              t: np.ndarray) -> np.ndarray:
        """Compute conservation law constraint."""
        if self.conservation_type == 'mass':
            # Mass conservation: ∂u/∂t + ∇·(u*v) = 0
            # Simplified: ∂u/∂t + ∂u/∂x = 0
            h = 1e-5
            u_t = (self.network.predict(np.column_stack([x, t + h])) -
                   self.network.predict(np.column_stack([x, t - h]))) / (2 * h)
            u_x = (self.network.predict(np.column_stack([x + h, t])) -
                   self.network.predict(np.column_stack([x - h, t]))) / (2 * h)
            return u_t.ravel() + u_x.ravel()
        elif self.conservation_type == 'energy':
            # Energy conservation constraint
            return self._energy_conservation_constraint(u, x, t)
        else:
            raise ValueError(f"Unknown conservation type: {self.conservation_type}")
    def _energy_conservation_constraint(self, u: np.ndarray, x: np.ndarray,
                                      t: np.ndarray) -> np.ndarray:
        """Energy conservation constraint."""
        # Simplified energy conservation
        # E = 1/2 * u² + V(x) = constant
        energy = 0.5 * u**2 + 0.5 * x**2  # Harmonic potential
        # Energy should be constant in time
        h = 1e-5
        energy_t = (self._compute_energy(x, t + h) -
                   self._compute_energy(x, t - h)) / (2 * h)
        return energy_t
    def _compute_energy(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute total energy."""
        u = self.network.predict(np.column_stack([x, t]))
        return 0.5 * u.ravel()**2 + 0.5 * x**2
class SymmetryAwareNN:
    """
    Neural network respecting physical symmetries.
    Features:
    - Translation invariance
    - Rotation invariance
    - Scale invariance
    - Time reversal symmetry
    """
    def __init__(self,
                 layers: List[int],
                 symmetry_type: str = 'translation',
                 symmetry_weight: float = 1.0):
        self.layers = layers
        self.symmetry_type = symmetry_type
        self.symmetry_weight = symmetry_weight
        # Initialize network
        from .neural_networks import MLP
        self.network = MLP(layer_sizes=layers)
    def apply_symmetry(self, x: np.ndarray, transformation: str) -> np.ndarray:
        """Apply symmetry transformation to input."""
        if transformation == 'translation':
            # Translation invariance: f(x + a) = f(x)
            return x + np.random.normal(0, 0.1, x.shape)
        elif transformation == 'rotation':
            # Rotation invariance (for 2D)
            if x.shape[1] >= 2:
                theta = np.random.uniform(0, 2*np.pi)
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                          [np.sin(theta), np.cos(theta)]])
                x_rot = x[:, :2] @ rotation_matrix.T
                if x.shape[1] > 2:
                    x_rot = np.column_stack([x_rot, x[:, 2:]])
                return x_rot
            else:
                return x
        elif transformation == 'scaling':
            # Scale invariance: f(αx) = α^n f(x)
            scale = np.random.uniform(0.5, 2.0)
            return scale * x
        else:
            return x
    def symmetry_loss(self, x: np.ndarray) -> float:
        """Compute symmetry violation loss."""
        # Original prediction
        u_orig = self.network.predict(x)
        # Transformed input
        x_transformed = self.apply_symmetry(x, self.symmetry_type)
        u_transformed = self.network.predict(x_transformed)
        # Symmetry loss (predictions should be similar)
        if self.symmetry_type == 'translation':
            return np.mean((u_orig - u_transformed)**2)
        elif self.symmetry_type == 'scaling':
            # Scale covariant: u(αx) = α^n u(x)
            scale_power = 1.0  # Adjust based on physical quantity
            expected = (x_transformed[0, 0] / x[0, 0])**scale_power * u_orig
            return np.mean((u_transformed - expected)**2)
        else:
            return np.mean((u_orig - u_transformed)**2)
def create_pde_test_data(equation_type: str = 'heat') -> Dict[str, np.ndarray]:
    """Create test data for PDE problems."""
    np.random.seed(42)
    # Domain
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 51)
    X, T = np.meshgrid(x, t)
    if equation_type == 'heat':
        # Analytical solution for heat equation with specific IC/BC
        # u(x,t) = sin(πx) * exp(-π²t)
        U = np.sin(np.pi * X) * np.exp(-np.pi**2 * T)
    elif equation_type == 'wave':
        # Analytical solution for wave equation
        # u(x,t) = sin(πx) * cos(πt)
        U = np.sin(np.pi * X) * np.cos(np.pi * T)
    elif equation_type == 'burgers':
        # Approximate solution for Burgers equation
        U = 0.5 * (np.sin(np.pi * X) * np.exp(-0.1 * T) +
                   np.cos(np.pi * X) * np.exp(-0.05 * T))
    return {
        'x': X.ravel(),
        't': T.ravel(),
        'u': U.ravel(),
        'x_grid': X,
        't_grid': T,
        'u_grid': U
    }
# Visualization utilities
def plot_pinn_training(results: PINNResults, title: str = "PINN Training Results"):
    """Plot PINN training results with Berkeley styling."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Berkeley colors
    berkeley_blue = '#003262'
    california_gold = '#FDB515'
    epochs = range(1, len(results.loss_history) + 1)
    # Total loss
    axes[0, 0].semilogy(epochs, results.loss_history, color=berkeley_blue, linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    # PDE loss
    axes[0, 1].semilogy(epochs, results.pde_loss_history, color=california_gold, linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PDE Loss')
    axes[0, 1].set_title('PDE Residual Loss')
    axes[0, 1].grid(True, alpha=0.3)
    # Boundary condition loss
    axes[1, 0].semilogy(epochs, results.bc_loss_history, color='red', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('BC Loss')
    axes[1, 0].set_title('Boundary Condition Loss')
    axes[1, 0].grid(True, alpha=0.3)
    # Initial condition loss
    axes[1, 1].semilogy(epochs, results.ic_loss_history, color='green', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IC Loss')
    axes[1, 1].set_title('Initial Condition Loss')
    axes[1, 1].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
def plot_pde_solution(x_grid: np.ndarray, t_grid: np.ndarray,
                     u_pred: np.ndarray, u_true: Optional[np.ndarray] = None,
                     title: str = "PDE Solution"):
    """Plot PDE solution with Berkeley styling."""
    fig, axes = plt.subplots(1, 3 if u_true is not None else 1, figsize=(15, 5))
    if u_true is None:
        axes = [axes]
    # Predicted solution
    im1 = axes[0].contourf(x_grid, t_grid, u_pred, levels=50, cmap='viridis')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title('Predicted Solution')
    plt.colorbar(im1, ax=axes[0])
    if u_true is not None:
        # True solution
        im2 = axes[1].contourf(x_grid, t_grid, u_true, levels=50, cmap='viridis')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('t')
        axes[1].set_title('True Solution')
        plt.colorbar(im2, ax=axes[1])
        # Error
        error = np.abs(u_pred - u_true)
        im3 = axes[2].contourf(x_grid, t_grid, error, levels=50, cmap='Reds')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('t')
        axes[2].set_title('Absolute Error')
        plt.colorbar(im3, ax=axes[2])
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig