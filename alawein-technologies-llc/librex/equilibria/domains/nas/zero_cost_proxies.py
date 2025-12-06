"""Zero-cost proxies for fast architecture evaluation.

These proxies estimate architecture quality without training, enabling
rapid evaluation of thousands of architectures in seconds.
"""

from typing import Dict, Any, Optional, List, Callable, Union
import numpy as np
from abc import ABC, abstractmethod
import warnings

from .architecture import NASCell, MacroArchitecture, Operation, OperationType


class ZeroCostProxy(ABC):
    """Base class for zero-cost proxies."""

    @abstractmethod
    def compute(self,
               architecture: Union[NASCell, MacroArchitecture],
               data_batch: Optional[np.ndarray] = None) -> float:
        """
        Compute proxy score for an architecture.

        Args:
            architecture: Architecture to evaluate
            data_batch: Optional data batch for computation

        Returns:
            Proxy score (higher is better)
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return proxy name."""
        pass


class GradientNormProxy(ZeroCostProxy):
    """
    Gradient norm proxy.

    Architectures with higher gradient norms at initialization tend to train better.
    Based on "Picking Winning Tickets Before Training by Preserving Gradient Flow".
    """

    def compute(self,
               architecture: Union[NASCell, MacroArchitecture],
               data_batch: Optional[np.ndarray] = None) -> float:
        """
        Compute gradient norm score.

        In practice, this would:
        1. Initialize model with random weights
        2. Forward pass with data batch
        3. Backward pass
        4. Compute gradient norm

        For demonstration, we simulate based on architecture properties.
        """
        if isinstance(architecture, NASCell):
            # More connections and diverse operations -> higher gradient flow
            score = 0
            for edge in architecture.edges:
                if edge.operation.op_type != OperationType.NONE:
                    # Different operations contribute differently
                    if edge.operation.op_type == OperationType.SKIP_CONNECT:
                        score += 1.5  # Skip connections preserve gradients
                    elif 'conv' in edge.operation.op_type.value:
                        score += 1.0
                    elif 'pool' in edge.operation.op_type.value:
                        score += 0.5
                    else:
                        score += 0.8

            # Normalize by number of nodes
            score /= (architecture.n_nodes + architecture.n_inputs)

        else:  # MacroArchitecture
            score = 0
            depth = len(architecture.layers)

            for i, layer in enumerate(architecture.layers):
                # Gradient flow depends on layer type and position
                if layer.layer_type == 'conv':
                    score += 1.0 * np.exp(-i / depth)  # Decaying contribution
                elif layer.layer_type == 'fc':
                    score += 0.8 * np.exp(-i / depth)
                elif layer.layer_type == 'residual':
                    score += 1.5  # Residual connections preserve gradients
                elif layer.layer_type == 'attention':
                    score += 1.2
                else:
                    score += 0.5

            # Skip connections improve gradient flow
            score += len(architecture.skip_connections) * 0.5

            # Normalize by depth
            score /= max(1, depth)

        return score

    def name(self) -> str:
        return "grad_norm"


class JacobianCovarianceProxy(ZeroCostProxy):
    """
    Jacobian covariance proxy.

    Measures the expressivity of the architecture through the covariance
    of Jacobian matrices. Based on "Neural Architecture Search without Training".
    """

    def compute(self,
               architecture: Union[NASCell, MacroArchitecture],
               data_batch: Optional[np.ndarray] = None) -> float:
        """
        Compute Jacobian covariance score.

        In practice, this would:
        1. Compute Jacobian matrices for multiple inputs
        2. Calculate covariance of Jacobians
        3. Return log determinant of covariance matrix

        For demonstration, we estimate based on architecture diversity.
        """
        if isinstance(architecture, NASCell):
            # Diversity of operations increases expressivity
            operation_types = set()
            channel_sizes = set()

            for edge in architecture.edges:
                operation_types.add(edge.operation.op_type)
                if edge.operation.channels:
                    channel_sizes.add(edge.operation.channels)

            # More diverse operations -> higher covariance
            diversity_score = len(operation_types) / len(OperationType)
            channel_diversity = len(channel_sizes) / max(1, len(architecture.edges))

            score = diversity_score * 10 + channel_diversity * 5

        else:  # MacroArchitecture
            # Layer diversity and depth contribute to expressivity
            layer_types = set()
            channel_sizes = set()

            for layer in architecture.layers:
                layer_types.add(layer.layer_type)
                channel_sizes.add(layer.channels)

            diversity_score = len(layer_types) / 6  # 6 possible layer types
            channel_diversity = len(channel_sizes) / max(1, len(architecture.layers))
            depth_factor = np.log(len(architecture.layers) + 1)

            score = diversity_score * 10 + channel_diversity * 5 + depth_factor

        return score

    def name(self) -> str:
        return "jacob_cov"


class NTKProxy(ZeroCostProxy):
    """
    Neural Tangent Kernel condition number proxy.

    Lower condition number of NTK indicates better trainability.
    Based on "Neural Tangent Kernel: Convergence and Generalization in Neural Networks".
    """

    def compute(self,
               architecture: Union[NASCell, MacroArchitecture],
               data_batch: Optional[np.ndarray] = None) -> float:
        """
        Compute NTK condition number score.

        In practice, this would:
        1. Compute Neural Tangent Kernel matrix
        2. Calculate condition number (ratio of largest to smallest eigenvalue)
        3. Return negative condition number (lower is better)

        For demonstration, we estimate based on architecture balance.
        """
        if isinstance(architecture, NASCell):
            # Balanced architectures have better conditioning
            node_in_degree = {i: 0 for i in range(
                architecture.n_inputs + architecture.n_nodes)}
            node_out_degree = {i: 0 for i in range(
                architecture.n_inputs + architecture.n_nodes)}

            for edge in architecture.edges:
                node_out_degree[edge.from_node] += 1
                node_in_degree[edge.to_node] += 1

            # Compute balance score
            in_variance = np.var(list(node_in_degree.values()))
            out_variance = np.var(list(node_out_degree.values()))

            # Lower variance means better balance -> lower condition number
            condition_estimate = 1 + in_variance + out_variance

        else:  # MacroArchitecture
            # Check layer width balance
            channel_sizes = [layer.channels for layer in architecture.layers]

            if channel_sizes:
                # Large jumps in width hurt conditioning
                ratios = []
                for i in range(1, len(channel_sizes)):
                    ratio = channel_sizes[i] / max(1, channel_sizes[i-1])
                    ratios.append(max(ratio, 1/ratio))

                if ratios:
                    condition_estimate = np.mean(ratios)
                else:
                    condition_estimate = 1.0
            else:
                condition_estimate = float('inf')

        # Return negative since lower condition number is better
        return -condition_estimate

    def name(self) -> str:
        return "ntk"


class SynFlowProxy(ZeroCostProxy):
    """
    SynFlow proxy.

    Computes synaptic flow to avoid layer collapse.
    Based on "Pruning neural networks without any data by iteratively conserving synaptic flow".
    """

    def compute(self,
               architecture: Union[NASCell, MacroArchitecture],
               data_batch: Optional[np.ndarray] = None) -> float:
        """
        Compute SynFlow score.

        In practice, this would:
        1. Initialize with all positive weights
        2. Compute product of weights along paths
        3. Return path strength score

        For demonstration, we compute based on connectivity.
        """
        if isinstance(architecture, NASCell):
            # Compute path strength through the cell
            # More paths and stronger operations -> higher score
            path_score = 0

            # Count paths from inputs to outputs
            for input_node in range(architecture.n_inputs):
                for output_node in range(architecture.n_inputs,
                                       architecture.n_inputs + architecture.n_nodes):
                    # Check if path exists (simplified)
                    if any(e.from_node == input_node and e.to_node == output_node
                          for e in architecture.edges):
                        path_score += 1
                    # Check indirect paths (2-hop)
                    for mid_node in range(input_node + 1, output_node):
                        has_first = any(e.from_node == input_node and e.to_node == mid_node
                                       for e in architecture.edges)
                        has_second = any(e.from_node == mid_node and e.to_node == output_node
                                       for e in architecture.edges)
                        if has_first and has_second:
                            path_score += 0.5

            # Weight by operation strength
            for edge in architecture.edges:
                if edge.operation.op_type != OperationType.NONE:
                    if 'conv' in edge.operation.op_type.value:
                        path_score *= 1.1
                    elif edge.operation.op_type == OperationType.SKIP_CONNECT:
                        path_score *= 1.2

        else:  # MacroArchitecture
            # Product of layer capacities
            path_score = 1.0
            prev_channels = architecture.input_channels

            for layer in architecture.layers:
                if layer.layer_type != 'pool':
                    # Capacity increases with channels
                    capacity = layer.channels / prev_channels
                    path_score *= (1 + np.log(capacity + 1))
                    prev_channels = layer.channels

            # Skip connections increase flow
            path_score *= (1 + 0.1 * len(architecture.skip_connections))

        return path_score

    def name(self) -> str:
        return "synflow"


class ZenScoreProxy(ZeroCostProxy):
    """
    Zen score proxy.

    Measures network expressivity through Gaussian complexity.
    Based on "Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition".
    """

    def compute(self,
               architecture: Union[NASCell, MacroArchitecture],
               data_batch: Optional[np.ndarray] = None) -> float:
        """
        Compute Zen score.

        In practice, this would:
        1. Pass Gaussian noise through network
        2. Measure output complexity
        3. Return complexity score

        For demonstration, we estimate based on architecture properties.
        """
        if isinstance(architecture, NASCell):
            # Complexity increases with diverse operations and connections
            complexity = 0

            # Operation diversity
            operations_used = len(set(e.operation.op_type for e in architecture.edges))
            complexity += operations_used * 2

            # Connectivity complexity
            avg_connections = len(architecture.edges) / architecture.n_nodes
            complexity += avg_connections * 3

            # Channel diversity
            if architecture.edges:
                channel_variance = np.var([
                    e.operation.channels for e in architecture.edges
                    if e.operation.channels is not None
                ])
                complexity += np.sqrt(channel_variance + 1)

        else:  # MacroArchitecture
            complexity = 0

            # Depth complexity
            complexity += np.log(len(architecture.layers) + 1) * 5

            # Width variation
            channel_sizes = [layer.channels for layer in architecture.layers]
            if len(channel_sizes) > 1:
                complexity += np.std(channel_sizes) / np.mean(channel_sizes) * 10

            # Layer type diversity
            layer_types = set(layer.layer_type for layer in architecture.layers)
            complexity += len(layer_types) * 2

            # Non-linearity from skip connections
            complexity += len(architecture.skip_connections) * 1.5

        return complexity

    def name(self) -> str:
        return "zen_score"


# Convenience functions
def grad_norm_proxy(architecture: Union[NASCell, MacroArchitecture],
                   data_batch: Optional[np.ndarray] = None) -> float:
    """Compute gradient norm proxy score."""
    return GradientNormProxy().compute(architecture, data_batch)


def jacob_cov_proxy(architecture: Union[NASCell, MacroArchitecture],
                   data_batch: Optional[np.ndarray] = None) -> float:
    """Compute Jacobian covariance proxy score."""
    return JacobianCovarianceProxy().compute(architecture, data_batch)


def ntk_proxy(architecture: Union[NASCell, MacroArchitecture],
             data_batch: Optional[np.ndarray] = None) -> float:
    """Compute NTK condition number proxy score."""
    return NTKProxy().compute(architecture, data_batch)


def synflow_proxy(architecture: Union[NASCell, MacroArchitecture],
                 data_batch: Optional[np.ndarray] = None) -> float:
    """Compute SynFlow proxy score."""
    return SynFlowProxy().compute(architecture, data_batch)


def zen_score(architecture: Union[NASCell, MacroArchitecture],
             data_batch: Optional[np.ndarray] = None) -> float:
    """Compute Zen score proxy."""
    return ZenScoreProxy().compute(architecture, data_batch)


class ProxyEnsemble:
    """
    Ensemble of zero-cost proxies for robust evaluation.

    Combines multiple proxies to get more reliable architecture ranking.
    """

    def __init__(self,
                proxies: Optional[List[ZeroCostProxy]] = None,
                weights: Optional[List[float]] = None):
        """
        Initialize proxy ensemble.

        Args:
            proxies: List of proxy instances to use
            weights: Weights for each proxy (default: equal weights)
        """
        if proxies is None:
            # Use all available proxies by default
            proxies = [
                GradientNormProxy(),
                JacobianCovarianceProxy(),
                NTKProxy(),
                SynFlowProxy(),
                ZenScoreProxy()
            ]

        self.proxies = proxies

        if weights is None:
            weights = [1.0 / len(proxies)] * len(proxies)
        elif len(weights) != len(proxies):
            raise ValueError("Number of weights must match number of proxies")

        # Normalize weights
        weight_sum = sum(weights)
        self.weights = [w / weight_sum for w in weights]

    def compute(self,
               architecture: Union[NASCell, MacroArchitecture],
               data_batch: Optional[np.ndarray] = None,
               return_all: bool = False) -> Union[float, Dict[str, float]]:
        """
        Compute ensemble score.

        Args:
            architecture: Architecture to evaluate
            data_batch: Optional data batch
            return_all: If True, return individual scores too

        Returns:
            Ensemble score or dictionary with all scores
        """
        scores = {}
        for proxy in self.proxies:
            scores[proxy.name()] = proxy.compute(architecture, data_batch)

        # Normalize scores (z-score normalization would be better with history)
        normalized_scores = {}
        for name, score in scores.items():
            # Simple min-max normalization for now
            normalized_scores[name] = score

        # Compute weighted average
        ensemble_score = sum(
            normalized_scores[proxy.name()] * weight
            for proxy, weight in zip(self.proxies, self.weights)
        )

        if return_all:
            scores['ensemble'] = ensemble_score
            return scores
        return ensemble_score


def rank_correlation(proxy_scores: List[float],
                     true_scores: List[float]) -> float:
    """
    Compute Spearman rank correlation between proxy and true scores.

    Args:
        proxy_scores: List of proxy scores
        true_scores: List of true accuracy/performance scores

    Returns:
        Spearman correlation coefficient
    """
    from scipy.stats import spearmanr

    if len(proxy_scores) != len(true_scores):
        raise ValueError("Score lists must have same length")

    if len(proxy_scores) < 2:
        return 0.0

    correlation, _ = spearmanr(proxy_scores, true_scores)
    return correlation