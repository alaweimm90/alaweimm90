"""Neural architecture representations for NAS.

This module defines various search space representations including:
- Cell-based search (DARTS, ENAS style)
- Macro search (layer-wise architecture)
- Operation definitions
- Encoding/decoding methods for optimization
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib


class OperationType(Enum):
    """Available operations in the search space."""
    NONE = 'none'
    SKIP_CONNECT = 'skip'
    AVG_POOL_3x3 = 'avg_pool3x3'
    MAX_POOL_3x3 = 'max_pool3x3'
    CONV_1x1 = 'conv1x1'
    CONV_3x3 = 'conv3x3'
    CONV_5x5 = 'conv5x5'
    SEP_CONV_3x3 = 'sep_conv3x3'
    SEP_CONV_5x5 = 'sep_conv5x5'
    SEP_CONV_7x7 = 'sep_conv7x7'
    DIL_CONV_3x3 = 'dil_conv3x3'
    DIL_CONV_5x5 = 'dil_conv5x5'
    DEPTHWISE_CONV = 'depthwise_conv'
    GROUPED_CONV = 'grouped_conv'
    ATTENTION = 'attention'
    SQUEEZE_EXCITE = 'squeeze_excite'
    INVERTED_BOTTLENECK = 'inverted_bottleneck'


@dataclass
class Operation:
    """Single operation in the architecture."""
    op_type: OperationType
    stride: int = 1
    channels: Optional[int] = None
    kernel_size: Optional[int] = None
    dilation: Optional[int] = 1
    groups: Optional[int] = 1
    activation: str = 'relu'
    normalization: str = 'batch_norm'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'op_type': self.op_type.value,
            'stride': self.stride,
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'dilation': self.dilation,
            'groups': self.groups,
            'activation': self.activation,
            'normalization': self.normalization
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Operation':
        """Create from dictionary representation."""
        data['op_type'] = OperationType(data['op_type'])
        return cls(**data)

    def get_params(self) -> int:
        """Estimate parameter count for this operation."""
        if self.op_type in [OperationType.NONE, OperationType.SKIP_CONNECT,
                            OperationType.AVG_POOL_3x3, OperationType.MAX_POOL_3x3]:
            return 0

        if self.channels is None:
            return 0

        params = 0
        if 'conv' in self.op_type.value:
            kernel_size = self.kernel_size or 3
            if 'sep' in self.op_type.value:
                # Separable convolution: depthwise + pointwise
                params = kernel_size * kernel_size * self.channels  # Depthwise
                params += self.channels * self.channels  # Pointwise
            elif 'depthwise' in self.op_type.value:
                params = kernel_size * kernel_size * self.channels
            else:
                # Regular convolution
                effective_groups = self.groups or 1
                params = (kernel_size * kernel_size * self.channels * self.channels) // effective_groups

        # Add normalization parameters
        if self.normalization == 'batch_norm':
            params += 2 * self.channels  # Scale and shift

        return params

    def get_flops(self, input_size: int) -> int:
        """Estimate FLOPs for this operation."""
        if self.op_type in [OperationType.NONE, OperationType.SKIP_CONNECT]:
            return 0

        if self.channels is None:
            return 0

        output_size = input_size // self.stride
        flops = 0

        if 'pool' in self.op_type.value:
            kernel_size = 3  # Pooling kernel size
            flops = kernel_size * kernel_size * self.channels * output_size * output_size

        elif 'conv' in self.op_type.value:
            kernel_size = self.kernel_size or 3
            if 'sep' in self.op_type.value:
                # Separable convolution
                flops = kernel_size * kernel_size * self.channels * output_size * output_size
                flops += self.channels * self.channels * output_size * output_size
            elif 'depthwise' in self.op_type.value:
                flops = kernel_size * kernel_size * self.channels * output_size * output_size
            else:
                # Regular convolution
                effective_groups = self.groups or 1
                flops = (kernel_size * kernel_size * self.channels * self.channels *
                        output_size * output_size) // effective_groups

        return flops


@dataclass
class Edge:
    """Edge in the cell DAG."""
    from_node: int
    to_node: int
    operation: Operation

    def to_dict(self) -> Dict[str, Any]:
        return {
            'from_node': self.from_node,
            'to_node': self.to_node,
            'operation': self.operation.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Edge':
        data['operation'] = Operation.from_dict(data['operation'])
        return cls(**data)


class NASCell:
    """
    Cell-based architecture representation (like DARTS, ENAS).
    Each cell is a directed acyclic graph (DAG) of operations.
    """

    def __init__(self, n_nodes: int = 4, n_inputs: int = 2, initialize_random: bool = False):
        """
        Initialize a NAS cell.

        Args:
            n_nodes: Number of intermediate nodes in the cell
            n_inputs: Number of input nodes (typically 2: previous cells)
            initialize_random: Whether to initialize with random edges
        """
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.edges: List[Edge] = []
        self.operations = list(OperationType)

        # Initialize with random edges if requested
        if initialize_random:
            self._initialize_random()

    def _initialize_random(self):
        """Initialize with random valid connections."""
        # Each intermediate node connects to 2 previous nodes
        for i in range(self.n_inputs, self.n_inputs + self.n_nodes):
            # Connect to two random previous nodes
            prev_nodes = list(range(i))
            if len(prev_nodes) >= 2:
                selected = np.random.choice(prev_nodes, size=2, replace=False)
                for from_node in selected:
                    op = Operation(
                        op_type=np.random.choice(self.operations),
                        channels=np.random.choice([32, 64, 128])
                    )
                    self.add_edge(from_node, i, op)

    def add_edge(self, from_node: int, to_node: int, operation: Operation):
        """Add an edge to the cell."""
        if from_node >= to_node:
            raise ValueError("Cell must be a DAG: from_node must be less than to_node")
        self.edges.append(Edge(from_node, to_node, operation))

    def remove_edge(self, from_node: int, to_node: int):
        """Remove an edge from the cell."""
        self.edges = [e for e in self.edges
                     if not (e.from_node == from_node and e.to_node == to_node)]

    def to_encoding(self) -> np.ndarray:
        """
        Encode cell as a vector for optimization.

        Returns:
            Encoding as concatenation of:
            - Operation indices for each possible edge
            - Channel sizes
        """
        # Count total possible edges
        total_edges = 0
        for i in range(self.n_inputs, self.n_inputs + self.n_nodes):
            total_edges += i  # Each node can connect to all previous nodes

        encoding = np.zeros(total_edges * 2)  # Operation + channels for each edge

        edge_idx = 0
        for i in range(self.n_inputs, self.n_inputs + self.n_nodes):
            for j in range(i):
                # Find edge from j to i
                edge = self._find_edge(j, i)
                if edge_idx * 2 + 1 < len(encoding):
                    if edge:
                        encoding[edge_idx * 2] = self.operations.index(edge.operation.op_type)
                        encoding[edge_idx * 2 + 1] = edge.operation.channels or 64
                    else:
                        encoding[edge_idx * 2] = 0  # None operation
                        encoding[edge_idx * 2 + 1] = 64  # Default channels
                edge_idx += 1

        return encoding

    def from_encoding(self, encoding: np.ndarray) -> 'NASCell':
        """
        Decode optimization solution to architecture.

        Args:
            encoding: Flattened vector encoding the cell

        Returns:
            Self for chaining
        """
        self.edges = []

        edge_idx = 0
        for i in range(self.n_inputs, self.n_inputs + self.n_nodes):
            for j in range(i):
                if edge_idx * 2 + 1 < len(encoding):
                    op_idx = int(encoding[edge_idx * 2])
                    channels = int(encoding[edge_idx * 2 + 1])

                    if op_idx > 0 and op_idx < len(self.operations):
                        op = Operation(
                            op_type=self.operations[op_idx],
                            channels=channels
                        )
                        self.add_edge(j, i, op)
                edge_idx += 1

        return self

    def _find_edge(self, from_node: int, to_node: int) -> Optional[Edge]:
        """Find edge between two nodes."""
        for edge in self.edges:
            if edge.from_node == from_node and edge.to_node == to_node:
                return edge
        return None

    def to_genotype(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Convert to genotype format (compact representation).

        Returns:
            Dictionary with normal and reduce cell specifications
        """
        gene = []
        for i in range(self.n_inputs, self.n_inputs + self.n_nodes):
            edges_to_i = [(e.operation.op_type.value, e.from_node)
                         for e in self.edges if e.to_node == i]
            gene.append(edges_to_i)
        return {'normal': gene}

    def get_hash(self) -> str:
        """Get unique hash for this architecture."""
        genotype_str = json.dumps(self.to_genotype(), sort_keys=True)
        return hashlib.md5(genotype_str.encode()).hexdigest()

    def get_params(self) -> int:
        """Calculate total parameter count."""
        return sum(edge.operation.get_params() for edge in self.edges)

    def get_flops(self, input_size: int = 32) -> int:
        """Calculate total FLOPs."""
        return sum(edge.operation.get_flops(input_size) for edge in self.edges)


@dataclass
class Layer:
    """Single layer in macro architecture."""
    layer_type: str
    channels: int
    kernel_size: Optional[int] = None
    stride: int = 1
    padding: Optional[int] = None
    activation: str = 'relu'
    dropout: float = 0.0
    normalization: str = 'batch_norm'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'layer_type': self.layer_type,
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'activation': self.activation,
            'dropout': self.dropout,
            'normalization': self.normalization
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Layer':
        """Create from dictionary."""
        return cls(**data)

    def get_params(self, input_channels: int) -> int:
        """Estimate parameters for this layer."""
        params = 0

        if self.layer_type == 'conv':
            kernel_size = self.kernel_size or 3
            params = kernel_size * kernel_size * input_channels * self.channels

        elif self.layer_type == 'fc':
            params = input_channels * self.channels

        elif self.layer_type in ['residual', 'inception']:
            # Simplified estimation
            params = 3 * 3 * input_channels * self.channels * 2

        elif self.layer_type == 'attention':
            # Multi-head attention parameters
            params = 3 * input_channels * self.channels  # Q, K, V projections

        # Add normalization parameters
        if self.normalization == 'batch_norm':
            params += 2 * self.channels

        return params


class MacroArchitecture:
    """
    Layer-wise architecture definition.
    Variable depth, width, and skip connections.
    """

    def __init__(self, max_layers: int = 20, input_channels: int = 3, num_classes: int = 10):
        """
        Initialize macro architecture.

        Args:
            max_layers: Maximum number of layers
            input_channels: Number of input channels
            num_classes: Number of output classes
        """
        self.max_layers = max_layers
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.layers: List[Layer] = []
        self.skip_connections: List[Tuple[int, int]] = []

        self.layer_types = ['conv', 'fc', 'pool', 'residual', 'inception', 'attention']

        # Initialize with simple architecture
        self._initialize_default()

    def _initialize_default(self):
        """Initialize with default simple architecture."""
        # Conv layers
        self.layers.append(Layer('conv', 64, kernel_size=3))
        self.layers.append(Layer('conv', 128, kernel_size=3))
        self.layers.append(Layer('pool', 128))
        self.layers.append(Layer('conv', 256, kernel_size=3))
        self.layers.append(Layer('fc', self.num_classes))

    def add_layer(self, layer: Layer):
        """Add a layer to the architecture."""
        if len(self.layers) < self.max_layers:
            self.layers.append(layer)

    def add_skip_connection(self, from_idx: int, to_idx: int):
        """Add a skip connection between layers."""
        if from_idx < to_idx and to_idx < len(self.layers):
            self.skip_connections.append((from_idx, to_idx))

    def to_encoding(self) -> Dict[str, np.ndarray]:
        """
        Multi-component encoding for optimization.

        Returns:
            Dictionary containing:
            - depth: Number of layers
            - types: Categorical encoding of layer types
            - channels: Channel sizes for each layer
            - kernel_sizes: Kernel sizes for convolutional layers
            - skip_connections: Binary matrix of skip connections
        """
        n_layers = len(self.layers)

        # Encode layer types
        types = np.zeros(self.max_layers)
        channels = np.zeros(self.max_layers)
        kernel_sizes = np.zeros(self.max_layers)

        for i, layer in enumerate(self.layers):
            types[i] = self.layer_types.index(layer.layer_type)
            channels[i] = layer.channels
            kernel_sizes[i] = layer.kernel_size if layer.kernel_size else 0

        # Encode skip connections as binary matrix
        skip_matrix = np.zeros((self.max_layers, self.max_layers))
        for from_idx, to_idx in self.skip_connections:
            skip_matrix[from_idx, to_idx] = 1

        return {
            'depth': n_layers,
            'types': types,
            'channels': channels,
            'kernel_sizes': kernel_sizes,
            'skip_connections': skip_matrix
        }

    def from_encoding(self, encoding: Dict[str, np.ndarray]) -> 'MacroArchitecture':
        """
        Decode optimization solution to architecture.

        Args:
            encoding: Dictionary with encoded components

        Returns:
            Self for chaining
        """
        self.layers = []
        self.skip_connections = []

        depth = int(encoding.get('depth', 5))
        depth = min(depth, self.max_layers)

        types = encoding.get('types', np.zeros(depth))
        channels = encoding.get('channels', np.ones(depth) * 64)
        kernel_sizes = encoding.get('kernel_sizes', np.ones(depth) * 3)

        for i in range(depth):
            layer_type_idx = int(types[i]) % len(self.layer_types)
            layer_type = self.layer_types[layer_type_idx]

            layer = Layer(
                layer_type=layer_type,
                channels=int(channels[i]),
                kernel_size=int(kernel_sizes[i]) if kernel_sizes[i] > 0 else None
            )
            self.layers.append(layer)

        # Decode skip connections
        if 'skip_connections' in encoding:
            skip_matrix = encoding['skip_connections']
            for i in range(depth):
                for j in range(i + 1, depth):
                    if skip_matrix[i, j] > 0.5:
                        self.skip_connections.append((i, j))

        return self

    def to_flat_encoding(self) -> np.ndarray:
        """
        Flatten encoding to single vector for certain optimizers.

        Returns:
            Flattened encoding vector
        """
        encoding = self.to_encoding()
        flat = [encoding['depth']]
        flat.extend(encoding['types'].flatten())
        flat.extend(encoding['channels'].flatten())
        flat.extend(encoding['kernel_sizes'].flatten())
        flat.extend(encoding['skip_connections'].flatten())
        return np.array(flat)

    def from_flat_encoding(self, flat: np.ndarray) -> 'MacroArchitecture':
        """
        Decode from flattened vector.

        Args:
            flat: Flattened encoding vector

        Returns:
            Self for chaining
        """
        idx = 0
        depth = int(flat[idx])
        idx += 1

        types = flat[idx:idx + self.max_layers]
        idx += self.max_layers

        channels = flat[idx:idx + self.max_layers]
        idx += self.max_layers

        kernel_sizes = flat[idx:idx + self.max_layers]
        idx += self.max_layers

        skip_connections = flat[idx:].reshape((self.max_layers, self.max_layers))

        encoding = {
            'depth': depth,
            'types': types,
            'channels': channels,
            'kernel_sizes': kernel_sizes,
            'skip_connections': skip_connections
        }

        return self.from_encoding(encoding)

    def get_params(self) -> int:
        """Calculate total parameter count."""
        total_params = 0
        prev_channels = self.input_channels

        for layer in self.layers:
            total_params += layer.get_params(prev_channels)
            if layer.layer_type != 'pool':
                prev_channels = layer.channels

        return total_params

    def get_depth(self) -> int:
        """Get the depth of the architecture."""
        return len(self.layers)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'layers': [layer.to_dict() for layer in self.layers],
            'skip_connections': self.skip_connections,
            'max_layers': self.max_layers,
            'input_channels': self.input_channels,
            'num_classes': self.num_classes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MacroArchitecture':
        """Create from dictionary representation."""
        arch = cls(
            max_layers=data['max_layers'],
            input_channels=data['input_channels'],
            num_classes=data['num_classes']
        )
        arch.layers = [Layer.from_dict(layer_data) for layer_data in data['layers']]
        arch.skip_connections = data['skip_connections']
        return arch