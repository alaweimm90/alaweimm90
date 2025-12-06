"""Efficient Neural Architecture Search (ENAS) implementation.

ENAS uses weight sharing and reinforcement learning to dramatically
speed up architecture search.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass

from ..architecture import NASCell, MacroArchitecture, Operation, OperationType
from ..nas_problem import NASProblem, SearchSpace


@dataclass
class SharedWeights:
    """Shared weights across all architectures."""
    operation_weights: Dict[str, np.ndarray]
    embedding_dim: int = 100


class Controller:
    """
    LSTM controller for generating architectures.

    The controller learns to sample good architectures using REINFORCE.
    """

    def __init__(self,
                 hidden_size: int = 100,
                 n_layers: int = 2,
                 learning_rate: float = 3.5e-4):
        """
        Initialize controller.

        Args:
            hidden_size: LSTM hidden size
            n_layers: Number of LSTM layers
            learning_rate: Learning rate for REINFORCE
        """
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.learning_rate = learning_rate

        # Initialize LSTM parameters (simplified)
        self.lstm_weights = np.random.randn(hidden_size, hidden_size) * 0.01
        self.output_weights = np.random.randn(hidden_size, 20) * 0.01  # 20 possible decisions

        # For REINFORCE
        self.baseline = 0.0
        self.baseline_decay = 0.999

    def sample_architecture(self, problem: NASProblem) -> Tuple[Any, List[int], List[float]]:
        """
        Sample an architecture using the controller.

        Args:
            problem: NAS problem

        Returns:
            Tuple of (architecture, decisions, log_probs)
        """
        decisions = []
        log_probs = []

        # Initialize LSTM state
        hidden = np.zeros(self.hidden_size)
        cell = np.zeros(self.hidden_size)

        if problem.search_space == SearchSpace.CELL:
            # Sample cell architecture
            architecture = NASCell(
                n_nodes=problem.cell_config['n_nodes'],
                n_inputs=problem.cell_config['n_inputs']
            )

            # For each node, decide connections and operations
            for to_node in range(architecture.n_inputs,
                               architecture.n_inputs + architecture.n_nodes):
                # Sample 2 input connections
                for _ in range(2):
                    # LSTM step
                    hidden, cell = self._lstm_step(hidden, cell)

                    # Sample from node
                    from_node_logits = hidden @ self.output_weights[:, :to_node]
                    from_node_probs = self._softmax(from_node_logits)
                    from_node = np.random.choice(to_node, p=from_node_probs)

                    decisions.append(from_node)
                    log_probs.append(np.log(from_node_probs[from_node] + 1e-10))

                    # Sample operation
                    hidden, cell = self._lstm_step(hidden, cell)
                    op_logits = hidden @ self.output_weights[:, :len(OperationType)]
                    op_probs = self._softmax(op_logits)
                    op_idx = np.random.choice(len(OperationType), p=op_probs)

                    decisions.append(op_idx)
                    log_probs.append(np.log(op_probs[op_idx] + 1e-10))

                    # Add edge
                    op = Operation(
                        op_type=list(OperationType)[op_idx],
                        channels=64
                    )
                    if from_node < to_node:  # Ensure DAG
                        architecture.add_edge(from_node, to_node, op)

        else:
            # Sample macro architecture
            architecture = MacroArchitecture(
                max_layers=problem.macro_config['max_layers'],
                input_channels=problem.macro_config['input_channels'],
                num_classes=problem.macro_config['num_classes']
            )
            architecture.layers = []

            # Sample depth
            hidden, cell = self._lstm_step(hidden, cell)
            depth_logits = hidden @ self.output_weights[:, :15]  # Max 15 layers
            depth_probs = self._softmax(depth_logits)
            depth = np.random.choice(15, p=depth_probs) + 3  # 3-17 layers

            decisions.append(depth)
            log_probs.append(np.log(depth_probs[depth - 3] + 1e-10))

            # Sample each layer
            for i in range(depth):
                hidden, cell = self._lstm_step(hidden, cell)

                if i == depth - 1:
                    # Final layer is always FC
                    from ..architecture import Layer
                    architecture.layers.append(
                        Layer('fc', problem.macro_config['num_classes'])
                    )
                else:
                    # Sample layer type
                    type_logits = hidden @ self.output_weights[:, :6]
                    type_probs = self._softmax(type_logits)
                    layer_type_idx = np.random.choice(6, p=type_probs)
                    layer_type = ['conv', 'fc', 'pool', 'residual', 'inception', 'attention'][layer_type_idx]

                    decisions.append(layer_type_idx)
                    log_probs.append(np.log(type_probs[layer_type_idx] + 1e-10))

                    # Sample channels
                    hidden, cell = self._lstm_step(hidden, cell)
                    channel_logits = hidden @ self.output_weights[:, :5]
                    channel_probs = self._softmax(channel_logits)
                    channel_idx = np.random.choice(5, p=channel_probs)
                    channels = [32, 64, 128, 256, 512][channel_idx]

                    decisions.append(channel_idx)
                    log_probs.append(np.log(channel_probs[channel_idx] + 1e-10))

                    from ..architecture import Layer
                    architecture.layers.append(
                        Layer(layer_type, channels, kernel_size=3 if layer_type == 'conv' else None)
                    )

        return architecture, decisions, log_probs

    def _lstm_step(self, hidden: np.ndarray, cell: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple LSTM step (simplified for demonstration)."""
        # Simplified LSTM equations
        gates = hidden @ self.lstm_weights
        forget_gate = self._sigmoid(gates[:self.hidden_size // 4])
        input_gate = self._sigmoid(gates[self.hidden_size // 4:self.hidden_size // 2])
        cell_candidate = np.tanh(gates[self.hidden_size // 2:3 * self.hidden_size // 4])
        output_gate = self._sigmoid(gates[3 * self.hidden_size // 4:])

        cell = forget_gate * cell + input_gate * cell_candidate
        hidden = output_gate * np.tanh(cell)

        return hidden, cell

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def update(self, log_probs: List[float], reward: float):
        """
        Update controller using REINFORCE.

        Args:
            log_probs: Log probabilities of decisions
            reward: Reward (validation accuracy)
        """
        # Update baseline
        self.baseline = self.baseline_decay * self.baseline + \
                       (1 - self.baseline_decay) * reward

        # REINFORCE gradient
        advantage = reward - self.baseline
        policy_loss = -sum(log_probs) * advantage

        # Simplified gradient update
        # In practice, this would use proper backpropagation
        gradient_lstm = np.random.randn(*self.lstm_weights.shape) * policy_loss * 0.01
        gradient_output = np.random.randn(*self.output_weights.shape) * policy_loss * 0.01

        self.lstm_weights -= self.learning_rate * gradient_lstm
        self.output_weights -= self.learning_rate * gradient_output


class ENAS:
    """
    Efficient Neural Architecture Search.

    Uses weight sharing and RL to find architectures efficiently.
    """

    def __init__(self,
                 problem: NASProblem,
                 controller_hidden_size: int = 100,
                 controller_learning_rate: float = 3.5e-4,
                 child_learning_rate: float = 0.05,
                 entropy_weight: float = 0.01):
        """
        Initialize ENAS.

        Args:
            problem: NAS problem
            controller_hidden_size: Controller LSTM hidden size
            controller_learning_rate: Controller learning rate
            child_learning_rate: Child model learning rate
            entropy_weight: Entropy regularization weight
        """
        self.problem = problem
        self.entropy_weight = entropy_weight
        self.child_lr = child_learning_rate

        # Initialize controller
        self.controller = Controller(
            hidden_size=controller_hidden_size,
            learning_rate=controller_learning_rate
        )

        # Initialize shared weights
        self.shared_weights = SharedWeights(
            operation_weights={
                op.value: np.random.randn(100, 100) * 0.01
                for op in OperationType
            }
        )

    def train_child(self, architecture: Any, n_steps: int = 50) -> float:
        """
        Train child model with shared weights.

        Args:
            architecture: Architecture to train
            n_steps: Number of training steps

        Returns:
            Validation accuracy
        """
        # Simulate training with shared weights
        # In practice, this would:
        # 1. Build model using architecture and shared weights
        # 2. Train for n_steps on training data
        # 3. Return validation accuracy

        # For demonstration, simulate improvement
        base_acc = 0.5
        architecture_quality = np.random.random() * 0.3
        training_gain = min(0.2, n_steps * 0.004)

        val_acc = base_acc + architecture_quality + training_gain + np.random.normal(0, 0.05)
        return np.clip(val_acc, 0, 1)

    def train_controller(self, n_architectures: int = 10):
        """
        Train controller by sampling architectures and getting rewards.

        Args:
            n_architectures: Number of architectures to sample per iteration
        """
        rewards = []
        log_probs_batch = []

        for _ in range(n_architectures):
            # Sample architecture
            arch, decisions, log_probs = self.controller.sample_architecture(self.problem)

            # Train child and get validation accuracy as reward
            val_acc = self.train_child(arch, n_steps=50)

            rewards.append(val_acc)
            log_probs_batch.append(log_probs)

        # Update controller with all samples
        for log_probs, reward in zip(log_probs_batch, rewards):
            self.controller.update(log_probs, reward)

        return np.mean(rewards)

    def run(self, n_iterations: int = 100) -> Dict[str, Any]:
        """
        Run ENAS.

        Args:
            n_iterations: Number of controller training iterations

        Returns:
            Results dictionary
        """
        history = {
            'rewards': [],
            'best_architectures': [],
            'controller_baseline': []
        }

        best_architecture = None
        best_reward = 0

        print(f"Running ENAS for {n_iterations} iterations")

        for iteration in range(n_iterations):
            # Train controller
            avg_reward = self.train_controller(n_architectures=10)

            history['rewards'].append(avg_reward)
            history['controller_baseline'].append(self.controller.baseline)

            # Sample best architecture
            best_arch_iter = None
            best_reward_iter = 0

            for _ in range(5):  # Sample 5 architectures and pick best
                arch, _, _ = self.controller.sample_architecture(self.problem)
                reward = self.train_child(arch, n_steps=100)  # Train longer for evaluation

                if reward > best_reward_iter:
                    best_reward_iter = reward
                    best_arch_iter = arch

            if best_reward_iter > best_reward:
                best_reward = best_reward_iter
                best_architecture = best_arch_iter

            history['best_architectures'].append(best_arch_iter)

            # Progress report
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{n_iterations}: "
                     f"Avg reward: {avg_reward:.4f}, "
                     f"Best: {best_reward:.4f}, "
                     f"Baseline: {self.controller.baseline:.4f}")

        # Final evaluation
        if best_architecture:
            final_metrics = self.problem.evaluate_architecture(
                best_architecture, return_all_metrics=True)
        else:
            final_metrics = {}

        return {
            'best_architecture': best_architecture,
            'best_reward': best_reward,
            'final_metrics': final_metrics,
            'history': history,
            'controller': self.controller,
            'shared_weights': self.shared_weights
        }


def enas(problem: NASProblem, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Efficient Neural Architecture Search (ENAS).

    Args:
        problem: NAS problem
        config: Configuration dictionary

    Returns:
        Results dictionary
    """
    enas_search = ENAS(
        problem=problem,
        controller_hidden_size=config.get('controller_hidden_size', 100),
        controller_learning_rate=config.get('controller_learning_rate', 3.5e-4),
        child_learning_rate=config.get('child_learning_rate', 0.05),
        entropy_weight=config.get('entropy_weight', 0.01)
    )

    return enas_search.run(n_iterations=config.get('n_iterations', 100))