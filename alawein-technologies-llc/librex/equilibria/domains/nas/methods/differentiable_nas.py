"""Differentiable NAS methods (DARTS, GDAS, etc.).

Implements gradient-based architecture search using continuous relaxation
of the discrete search space.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
import warnings

from ..architecture import NASCell, MacroArchitecture, Operation, OperationType
from ..nas_problem import NASProblem, SearchSpace


@dataclass
class DARTSCell:
    """Cell representation for DARTS with continuous architecture parameters."""
    n_nodes: int
    n_inputs: int
    operations: List[OperationType]
    alpha: np.ndarray  # Architecture parameters (softmax weights)
    weights: Optional[np.ndarray] = None  # Network weights


class DARTS:
    """
    Differentiable Architecture Search (DARTS).

    Uses continuous relaxation and bi-level optimization to find architectures.
    """

    def __init__(self,
                 problem: NASProblem,
                 learning_rate_arch: float = 3e-4,
                 learning_rate_weights: float = 0.025,
                 weight_decay: float = 3e-4,
                 momentum: float = 0.9):
        """
        Initialize DARTS.

        Args:
            problem: NAS problem to solve
            learning_rate_arch: Learning rate for architecture parameters
            learning_rate_weights: Learning rate for network weights
            weight_decay: Weight decay coefficient
            momentum: Momentum for SGD
        """
        self.problem = problem
        self.lr_arch = learning_rate_arch
        self.lr_weights = learning_rate_weights
        self.weight_decay = weight_decay
        self.momentum = momentum

        # Initialize architecture parameters
        self.cell = self._initialize_cell()

    def _initialize_cell(self) -> DARTSCell:
        """Initialize DARTS cell with random architecture parameters."""
        if self.problem.search_space != SearchSpace.CELL:
            warnings.warn("DARTS is designed for cell-based search space")

        n_nodes = self.problem.cell_config['n_nodes']
        n_inputs = self.problem.cell_config['n_inputs']
        operations = list(OperationType)

        # Initialize architecture parameters (alpha)
        # For each edge, we have a softmax over operations
        n_edges = sum(i for i in range(n_inputs, n_inputs + n_nodes))
        alpha = np.random.randn(n_edges, len(operations)) * 0.001

        return DARTSCell(
            n_nodes=n_nodes,
            n_inputs=n_inputs,
            operations=operations,
            alpha=alpha
        )

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def mixed_operation(self, inputs: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Compute mixed operation output.

        In real implementation, this would apply actual operations.
        Here we simulate with weighted combination.

        Args:
            inputs: Input tensor
            alpha: Softmax weights for operations

        Returns:
            Mixed operation output
        """
        # Simulate mixed operation
        # In practice, each operation would process inputs differently
        output = np.zeros_like(inputs)
        weights = self.softmax(alpha)

        for i, weight in enumerate(weights):
            # Simulate different operations having different effects
            if i < len(self.cell.operations):
                op_type = self.cell.operations[i]
                if op_type == OperationType.NONE:
                    op_output = np.zeros_like(inputs)
                elif op_type == OperationType.SKIP_CONNECT:
                    op_output = inputs
                else:
                    # Simulate convolution effect
                    op_output = inputs * np.random.randn(*inputs.shape) * 0.1 + inputs

                output += weight * op_output

        return output

    def forward_cell(self, inputs: List[np.ndarray]) -> np.ndarray:
        """
        Forward pass through the cell.

        Args:
            inputs: List of input tensors

        Returns:
            Output tensor
        """
        states = inputs.copy()
        edge_idx = 0

        for i in range(self.cell.n_inputs, self.cell.n_inputs + self.cell.n_nodes):
            # Collect inputs for this node
            node_inputs = []
            for j in range(i):
                # Mixed operation from j to i
                mixed_out = self.mixed_operation(
                    states[j],
                    self.cell.alpha[edge_idx]
                )
                node_inputs.append(mixed_out)
                edge_idx += 1

            # Combine inputs (sum in DARTS)
            states.append(np.sum(node_inputs, axis=0))

        # Concatenate all intermediate nodes for output
        output_states = states[self.cell.n_inputs:]
        return np.concatenate(output_states, axis=-1)

    def update_architecture_gradient(self,
                                    train_data: np.ndarray,
                                    val_data: np.ndarray) -> float:
        """
        Update architecture parameters using validation gradient.

        Args:
            train_data: Training data batch
            val_data: Validation data batch

        Returns:
            Validation loss
        """
        # Simulate architecture gradient computation
        # In practice, this would:
        # 1. Compute validation loss
        # 2. Backprop to get gradients w.r.t. alpha
        # 3. Update alpha using gradient descent

        # Simulate validation loss
        val_loss = np.random.random()

        # Simulate gradient
        grad_alpha = np.random.randn(*self.cell.alpha.shape) * 0.01

        # Update architecture parameters
        self.cell.alpha -= self.lr_arch * grad_alpha

        return val_loss

    def update_weights_gradient(self, train_data: np.ndarray) -> float:
        """
        Update network weights using training gradient.

        Args:
            train_data: Training data batch

        Returns:
            Training loss
        """
        # Simulate weight update
        # In practice, this would update the actual network weights

        train_loss = np.random.random()

        if self.cell.weights is None:
            # Initialize weights
            self.cell.weights = np.random.randn(100) * 0.01

        # Simulate gradient
        grad_weights = np.random.randn(*self.cell.weights.shape) * 0.01

        # Update with momentum SGD
        if not hasattr(self, 'velocity'):
            self.velocity = np.zeros_like(self.cell.weights)

        self.velocity = self.momentum * self.velocity - self.lr_weights * grad_weights
        self.cell.weights += self.velocity

        # Weight decay
        self.cell.weights *= (1 - self.weight_decay)

        return train_loss

    def derive_discrete_architecture(self) -> NASCell:
        """
        Derive discrete architecture from continuous parameters.

        Returns:
            Discrete NASCell architecture
        """
        cell = NASCell(
            n_nodes=self.cell.n_nodes,
            n_inputs=self.cell.n_inputs
        )
        cell.edges = []

        edge_idx = 0
        for i in range(self.cell.n_inputs, self.cell.n_inputs + self.cell.n_nodes):
            # For each node, select top-k incoming edges
            edge_strengths = []
            edge_info = []

            for j in range(i):
                # Get operation with highest weight
                op_weights = self.softmax(self.cell.alpha[edge_idx])
                max_op_idx = np.argmax(op_weights)
                max_weight = op_weights[max_op_idx]

                edge_strengths.append(max_weight)
                edge_info.append((j, self.cell.operations[max_op_idx]))
                edge_idx += 1

            # Select top-2 strongest edges
            top_edges_idx = np.argsort(edge_strengths)[-2:]

            for idx in top_edges_idx:
                if idx < len(edge_info):
                    from_node, op_type = edge_info[idx]
                    if op_type != OperationType.NONE:
                        op = Operation(
                            op_type=op_type,
                            channels=64  # Default channels
                        )
                        cell.add_edge(from_node, i, op)

        return cell

    def run(self, n_epochs: int = 50) -> Dict[str, Any]:
        """
        Run DARTS search.

        Args:
            n_epochs: Number of epochs

        Returns:
            Results dictionary
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'arch_params': []
        }

        for epoch in range(n_epochs):
            # Simulate data batches
            train_data = np.random.randn(32, 3, 32, 32)  # Batch of images
            val_data = np.random.randn(32, 3, 32, 32)

            # Update architecture (using validation data)
            val_loss = self.update_architecture_gradient(train_data, val_data)

            # Update weights (using training data)
            train_loss = self.update_weights_gradient(train_data)

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['arch_params'].append(self.cell.alpha.copy())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: "
                     f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # Derive final discrete architecture
        final_architecture = self.derive_discrete_architecture()

        # Evaluate final architecture
        metrics = self.problem.evaluate_architecture(final_architecture, return_all_metrics=True)

        return {
            'architecture': final_architecture,
            'metrics': metrics,
            'continuous_params': self.cell.alpha,
            'history': history
        }


class GDAS:
    """
    Gradient-based search using Differentiable Architecture Sampler (GDAS).

    Uses Gumbel-Softmax for differentiable sampling instead of mixed operations.
    """

    def __init__(self,
                 problem: NASProblem,
                 tau: float = 1.0,
                 tau_min: float = 0.1,
                 tau_decay: float = 0.97):
        """
        Initialize GDAS.

        Args:
            problem: NAS problem to solve
            tau: Temperature for Gumbel-Softmax
            tau_min: Minimum temperature
            tau_decay: Temperature decay rate
        """
        self.problem = problem
        self.tau = tau
        self.tau_min = tau_min
        self.tau_decay = tau_decay

        # Initialize like DARTS
        self.darts = DARTS(problem)

    def gumbel_softmax_sample(self, logits: np.ndarray, tau: float) -> np.ndarray:
        """
        Sample from Gumbel-Softmax distribution.

        Args:
            logits: Logits for categorical distribution
            tau: Temperature parameter

        Returns:
            Soft sample (approximation of one-hot)
        """
        # Add Gumbel noise
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, logits.shape)))
        y = logits + gumbel_noise

        # Apply softmax with temperature
        return self.darts.softmax(y / tau)

    def sample_architecture(self) -> np.ndarray:
        """
        Sample architecture using Gumbel-Softmax.

        Returns:
            Sampled architecture weights
        """
        sampled_alpha = np.zeros_like(self.darts.cell.alpha)

        for i in range(len(self.darts.cell.alpha)):
            sampled_alpha[i] = self.gumbel_softmax_sample(
                self.darts.cell.alpha[i],
                self.tau
            )

        return sampled_alpha

    def run(self, n_epochs: int = 50) -> Dict[str, Any]:
        """
        Run GDAS search.

        Args:
            n_epochs: Number of epochs

        Returns:
            Results dictionary
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'temperature': []
        }

        for epoch in range(n_epochs):
            # Sample architecture
            sampled_alpha = self.sample_architecture()

            # Store original alpha and use sampled
            original_alpha = self.darts.cell.alpha.copy()
            self.darts.cell.alpha = sampled_alpha

            # Update using sampled architecture
            train_data = np.random.randn(32, 3, 32, 32)
            val_data = np.random.randn(32, 3, 32, 32)

            val_loss = self.darts.update_architecture_gradient(train_data, val_data)
            train_loss = self.darts.update_weights_gradient(train_data)

            # Restore and update original alpha
            gradient = self.darts.cell.alpha - sampled_alpha
            self.darts.cell.alpha = original_alpha - self.darts.lr_arch * gradient

            # Decay temperature
            self.tau = max(self.tau_min, self.tau * self.tau_decay)

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['temperature'].append(self.tau)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: "
                     f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                     f"Tau: {self.tau:.4f}")

        # Derive final architecture
        final_architecture = self.darts.derive_discrete_architecture()
        metrics = self.problem.evaluate_architecture(final_architecture, return_all_metrics=True)

        return {
            'architecture': final_architecture,
            'metrics': metrics,
            'history': history
        }


def differentiable_nas(problem: NASProblem, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run differentiable NAS (DARTS).

    Args:
        problem: NAS problem to solve
        config: Configuration dictionary

    Returns:
        Results dictionary
    """
    darts = DARTS(
        problem=problem,
        learning_rate_arch=config.get('lr_arch', 3e-4),
        learning_rate_weights=config.get('lr_weights', 0.025),
        weight_decay=config.get('weight_decay', 3e-4),
        momentum=config.get('momentum', 0.9)
    )

    return darts.run(n_epochs=config.get('n_epochs', 50))


def gdas(problem: NASProblem, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run GDAS (Gradient-based Differentiable Architecture Search).

    Args:
        problem: NAS problem to solve
        config: Configuration dictionary

    Returns:
        Results dictionary
    """
    gdas_search = GDAS(
        problem=problem,
        tau=config.get('tau', 1.0),
        tau_min=config.get('tau_min', 0.1),
        tau_decay=config.get('tau_decay', 0.97)
    )

    return gdas_search.run(n_epochs=config.get('n_epochs', 50))