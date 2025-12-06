"""Hardware-aware NAS for optimizing architectures with device constraints.

This module provides tools for:
- Latency prediction on different devices
- Energy consumption estimation
- Memory footprint calculation
- Multi-objective optimization with hardware constraints
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import warnings

from .architecture import NASCell, MacroArchitecture, Operation, OperationType
from .nas_problem import NASProblem, NASObjective, NASConstraint, SearchSpace
from .nas_adapter import NASAdapter, StandardizedProblem


class DeviceType(Enum):
    """Target device types."""
    CPU = 'cpu'
    GPU = 'gpu'
    TPU = 'tpu'
    MOBILE = 'mobile'
    EDGE = 'edge'
    FPGA = 'fpga'


@dataclass
class DeviceProfile:
    """Profile of target device characteristics."""
    device_type: DeviceType
    compute_capability: float  # TFLOPS
    memory_bandwidth: float  # GB/s
    memory_capacity: float  # GB
    power_budget: float  # Watts
    precision: str = 'fp32'  # fp32, fp16, int8


class LatencyPredictor:
    """
    Predict latency of architectures on target devices.

    Uses lookup tables or learned models to estimate inference time.
    """

    def __init__(self, device_type: Union[str, DeviceType]):
        """
        Initialize latency predictor.

        Args:
            device_type: Target device type
        """
        self.device_type = DeviceType(device_type) if isinstance(device_type, str) else device_type
        self.operation_latency = self._load_operation_latencies()

    def _load_operation_latencies(self) -> Dict[str, float]:
        """
        Load operation latency lookup table for device.

        In practice, these would be measured or learned from real hardware.
        Returns latency in microseconds per operation.
        """
        if self.device_type == DeviceType.GPU:
            return {
                'conv1x1': 0.5,
                'conv3x3': 2.0,
                'conv5x5': 5.0,
                'sep_conv3x3': 1.5,
                'sep_conv5x5': 3.0,
                'sep_conv7x7': 5.5,
                'dil_conv3x3': 2.5,
                'dil_conv5x5': 5.5,
                'depthwise_conv': 1.0,
                'grouped_conv': 1.8,
                'avg_pool3x3': 0.8,
                'max_pool3x3': 0.8,
                'skip': 0.1,
                'fc': 1.0,
                'attention': 10.0,
                'squeeze_excite': 2.0,
                'inverted_bottleneck': 3.0,
                'none': 0.0
            }
        elif self.device_type == DeviceType.MOBILE:
            # Mobile devices have different characteristics
            return {
                'conv1x1': 1.5,
                'conv3x3': 5.0,
                'conv5x5': 12.0,
                'sep_conv3x3': 3.0,
                'sep_conv5x5': 7.0,
                'sep_conv7x7': 12.0,
                'dil_conv3x3': 6.0,
                'dil_conv5x5': 13.0,
                'depthwise_conv': 2.0,
                'grouped_conv': 4.0,
                'avg_pool3x3': 1.5,
                'max_pool3x3': 1.5,
                'skip': 0.2,
                'fc': 3.0,
                'attention': 25.0,
                'squeeze_excite': 5.0,
                'inverted_bottleneck': 6.0,
                'none': 0.0
            }
        elif self.device_type == DeviceType.CPU:
            return {
                'conv1x1': 2.0,
                'conv3x3': 8.0,
                'conv5x5': 20.0,
                'sep_conv3x3': 5.0,
                'sep_conv5x5': 12.0,
                'sep_conv7x7': 20.0,
                'dil_conv3x3': 10.0,
                'dil_conv5x5': 22.0,
                'depthwise_conv': 3.0,
                'grouped_conv': 7.0,
                'avg_pool3x3': 2.0,
                'max_pool3x3': 2.0,
                'skip': 0.1,
                'fc': 5.0,
                'attention': 40.0,
                'squeeze_excite': 8.0,
                'inverted_bottleneck': 10.0,
                'none': 0.0
            }
        else:
            # Default/Edge devices
            return {k: v * 2 for k, v in
                   self._load_operation_latencies_for_device(DeviceType.CPU).items()}

    def _load_operation_latencies_for_device(self, device: DeviceType) -> Dict[str, float]:
        """Helper to avoid recursion."""
        if device == DeviceType.CPU:
            return {
                'conv1x1': 2.0,
                'conv3x3': 8.0,
                'conv5x5': 20.0,
                'sep_conv3x3': 5.0,
                'sep_conv5x5': 12.0,
                'sep_conv7x7': 20.0,
                'dil_conv3x3': 10.0,
                'dil_conv5x5': 22.0,
                'depthwise_conv': 3.0,
                'grouped_conv': 7.0,
                'avg_pool3x3': 2.0,
                'max_pool3x3': 2.0,
                'skip': 0.1,
                'fc': 5.0,
                'attention': 40.0,
                'squeeze_excite': 8.0,
                'inverted_bottleneck': 10.0,
                'none': 0.0
            }
        return {}

    def predict(self,
               architecture: Union[NASCell, MacroArchitecture],
               batch_size: int = 1,
               input_size: int = 224) -> float:
        """
        Predict latency in milliseconds.

        Args:
            architecture: Architecture to evaluate
            batch_size: Batch size for inference
            input_size: Input image size

        Returns:
            Predicted latency in milliseconds
        """
        total_latency = 0

        if isinstance(architecture, NASCell):
            # Sum latencies of all operations
            for edge in architecture.edges:
                op_type = edge.operation.op_type.value
                if op_type in self.operation_latency:
                    op_latency = self.operation_latency[op_type]

                    # Scale by channels and input size
                    if edge.operation.channels:
                        channel_factor = edge.operation.channels / 64  # Normalized to 64
                    else:
                        channel_factor = 1.0

                    size_factor = (input_size / 224) ** 2  # Quadratic in spatial size
                    batch_factor = batch_size

                    total_latency += op_latency * channel_factor * size_factor * batch_factor

        else:  # MacroArchitecture
            current_size = input_size

            for layer in architecture.layers:
                if layer.layer_type == 'conv':
                    # Map to operation type
                    if layer.kernel_size == 1:
                        op_key = 'conv1x1'
                    elif layer.kernel_size == 3:
                        op_key = 'conv3x3'
                    elif layer.kernel_size == 5:
                        op_key = 'conv5x5'
                    else:
                        op_key = 'conv3x3'

                    op_latency = self.operation_latency.get(op_key, 2.0)
                elif layer.layer_type == 'fc':
                    op_latency = self.operation_latency.get('fc', 1.0)
                elif layer.layer_type == 'pool':
                    op_latency = self.operation_latency.get('avg_pool3x3', 0.8)
                    current_size = current_size // 2  # Pooling reduces size
                elif layer.layer_type == 'attention':
                    op_latency = self.operation_latency.get('attention', 10.0)
                else:
                    op_latency = 2.0  # Default

                # Scale by channels and size
                channel_factor = layer.channels / 64
                size_factor = (current_size / 224) ** 2
                batch_factor = batch_size

                total_latency += op_latency * channel_factor * size_factor * batch_factor

                # Update size based on stride
                if layer.stride > 1:
                    current_size = current_size // layer.stride

        # Convert from microseconds to milliseconds
        return total_latency / 1000

    def build_lookup_table(self,
                          architectures: List[Union[NASCell, MacroArchitecture]]) -> Dict[str, float]:
        """
        Build lookup table of latencies for a set of architectures.

        Args:
            architectures: List of architectures to profile

        Returns:
            Dictionary mapping architecture hash to latency
        """
        lookup_table = {}

        for arch in architectures:
            if isinstance(arch, NASCell):
                arch_hash = arch.get_hash()
            else:
                arch_hash = str(arch.to_dict())

            latency = self.predict(arch)
            lookup_table[arch_hash] = latency

        return lookup_table


class EnergyEstimator:
    """
    Estimate energy consumption of architectures.

    Energy depends on operations, memory access patterns, and device characteristics.
    """

    def __init__(self, device_profile: DeviceProfile):
        """
        Initialize energy estimator.

        Args:
            device_profile: Target device profile
        """
        self.device_profile = device_profile
        self.operation_energy = self._load_operation_energy()

    def _load_operation_energy(self) -> Dict[str, float]:
        """
        Load operation energy lookup table.

        Returns energy in millijoules per operation.
        """
        # Base energy costs (scaled by device power budget)
        base_scale = self.device_profile.power_budget / 10  # Normalized to 10W baseline

        return {
            'conv1x1': 0.1 * base_scale,
            'conv3x3': 0.4 * base_scale,
            'conv5x5': 1.0 * base_scale,
            'sep_conv3x3': 0.3 * base_scale,
            'sep_conv5x5': 0.6 * base_scale,
            'sep_conv7x7': 1.1 * base_scale,
            'dil_conv3x3': 0.5 * base_scale,
            'dil_conv5x5': 1.1 * base_scale,
            'depthwise_conv': 0.2 * base_scale,
            'grouped_conv': 0.35 * base_scale,
            'avg_pool3x3': 0.15 * base_scale,
            'max_pool3x3': 0.15 * base_scale,
            'skip': 0.01 * base_scale,
            'fc': 0.2 * base_scale,
            'attention': 2.0 * base_scale,
            'squeeze_excite': 0.4 * base_scale,
            'inverted_bottleneck': 0.6 * base_scale,
            'none': 0.0
        }

    def estimate(self,
                architecture: Union[NASCell, MacroArchitecture],
                input_size: int = 224) -> float:
        """
        Estimate energy consumption in millijoules.

        Args:
            architecture: Architecture to evaluate
            input_size: Input size

        Returns:
            Estimated energy in millijoules
        """
        total_energy = 0

        if isinstance(architecture, NASCell):
            for edge in architecture.edges:
                op_type = edge.operation.op_type.value
                if op_type in self.operation_energy:
                    op_energy = self.operation_energy[op_type]

                    # Scale by channels and FLOPs
                    if edge.operation.channels:
                        flops = edge.operation.get_flops(input_size)
                        energy_scale = flops / 1e6  # Normalize to MFLOPs
                    else:
                        energy_scale = 1.0

                    total_energy += op_energy * energy_scale

        else:  # MacroArchitecture
            current_size = input_size

            for layer in architecture.layers:
                # Get operation energy
                if layer.layer_type == 'conv':
                    if layer.kernel_size == 1:
                        op_key = 'conv1x1'
                    elif layer.kernel_size == 3:
                        op_key = 'conv3x3'
                    elif layer.kernel_size == 5:
                        op_key = 'conv5x5'
                    else:
                        op_key = 'conv3x3'
                    op_energy = self.operation_energy.get(op_key, 0.4)
                elif layer.layer_type == 'fc':
                    op_energy = self.operation_energy.get('fc', 0.2)
                elif layer.layer_type == 'pool':
                    op_energy = self.operation_energy.get('avg_pool3x3', 0.15)
                    current_size = current_size // 2
                elif layer.layer_type == 'attention':
                    op_energy = self.operation_energy.get('attention', 2.0)
                else:
                    op_energy = 0.3  # Default

                # Scale by operations
                ops = layer.channels * current_size * current_size
                energy_scale = ops / 1e6  # Normalize to millions of operations

                total_energy += op_energy * energy_scale

                # Update size
                if layer.stride > 1:
                    current_size = current_size // layer.stride

        # Add memory access energy (proportional to parameters)
        if isinstance(architecture, NASCell):
            params = architecture.get_params()
        else:
            params = architecture.get_params()

        memory_energy = (params / 1e6) * 0.1 * self.device_profile.power_budget

        return total_energy + memory_energy


class MemoryEstimator:
    """Estimate memory footprint of architectures."""

    def __init__(self, precision: str = 'fp32'):
        """
        Initialize memory estimator.

        Args:
            precision: Numerical precision ('fp32', 'fp16', 'int8')
        """
        self.precision = precision
        self.bytes_per_param = {
            'fp32': 4,
            'fp16': 2,
            'int8': 1,
            'int4': 0.5
        }[precision]

    def estimate_parameters_memory(self,
                                  architecture: Union[NASCell, MacroArchitecture]) -> float:
        """
        Estimate memory for parameters in MB.

        Args:
            architecture: Architecture to evaluate

        Returns:
            Parameter memory in megabytes
        """
        if isinstance(architecture, NASCell):
            params = architecture.get_params()
        else:
            params = architecture.get_params()

        return (params * self.bytes_per_param) / (1024 * 1024)

    def estimate_activation_memory(self,
                                  architecture: Union[NASCell, MacroArchitecture],
                                  batch_size: int = 1,
                                  input_size: int = 224) -> float:
        """
        Estimate memory for activations in MB.

        Args:
            architecture: Architecture to evaluate
            batch_size: Batch size
            input_size: Input size

        Returns:
            Activation memory in megabytes
        """
        # Simplified estimation based on architecture depth and width
        if isinstance(architecture, NASCell):
            # Estimate based on number of edges and channels
            total_activations = 0
            for edge in architecture.edges:
                if edge.operation.channels:
                    # Activation size: batch * channels * height * width
                    act_size = batch_size * edge.operation.channels * input_size * input_size
                    total_activations += act_size

        else:  # MacroArchitecture
            total_activations = 0
            current_size = input_size

            for layer in architecture.layers:
                # Activation size for this layer
                act_size = batch_size * layer.channels * current_size * current_size
                total_activations += act_size

                # Update size based on pooling/stride
                if layer.layer_type == 'pool' or layer.stride > 1:
                    current_size = current_size // 2

        return (total_activations * self.bytes_per_param) / (1024 * 1024)

    def estimate_total_memory(self,
                            architecture: Union[NASCell, MacroArchitecture],
                            batch_size: int = 1,
                            input_size: int = 224) -> float:
        """
        Estimate total memory footprint in MB.

        Args:
            architecture: Architecture to evaluate
            batch_size: Batch size
            input_size: Input size

        Returns:
            Total memory in megabytes
        """
        param_memory = self.estimate_parameters_memory(architecture)
        activation_memory = self.estimate_activation_memory(architecture, batch_size, input_size)

        # Add overhead (gradients, optimizer states, etc.)
        overhead = (param_memory + activation_memory) * 0.2

        return param_memory + activation_memory + overhead


class HardwareAwareNAS:
    """
    Hardware-aware Neural Architecture Search.

    Optimizes architectures considering hardware constraints and objectives.
    """

    def __init__(self,
                target_device: Union[str, DeviceType] = 'gpu',
                device_profile: Optional[DeviceProfile] = None):
        """
        Initialize hardware-aware NAS.

        Args:
            target_device: Target device type
            device_profile: Optional detailed device profile
        """
        self.target_device = DeviceType(target_device) if isinstance(
            target_device, str) else target_device

        if device_profile is None:
            # Create default profile for device
            self.device_profile = self._get_default_profile(self.target_device)
        else:
            self.device_profile = device_profile

        self.latency_predictor = LatencyPredictor(self.target_device)
        self.energy_estimator = EnergyEstimator(self.device_profile)
        self.memory_estimator = MemoryEstimator(self.device_profile.precision)

    def _get_default_profile(self, device_type: DeviceType) -> DeviceProfile:
        """Get default device profile."""
        profiles = {
            DeviceType.GPU: DeviceProfile(
                device_type=DeviceType.GPU,
                compute_capability=10.0,  # TFLOPS
                memory_bandwidth=500.0,  # GB/s
                memory_capacity=16.0,  # GB
                power_budget=250.0,  # Watts
                precision='fp16'
            ),
            DeviceType.MOBILE: DeviceProfile(
                device_type=DeviceType.MOBILE,
                compute_capability=0.5,
                memory_bandwidth=30.0,
                memory_capacity=4.0,
                power_budget=5.0,
                precision='int8'
            ),
            DeviceType.CPU: DeviceProfile(
                device_type=DeviceType.CPU,
                compute_capability=1.0,
                memory_bandwidth=100.0,
                memory_capacity=32.0,
                power_budget=100.0,
                precision='fp32'
            ),
            DeviceType.EDGE: DeviceProfile(
                device_type=DeviceType.EDGE,
                compute_capability=0.2,
                memory_bandwidth=10.0,
                memory_capacity=2.0,
                power_budget=2.0,
                precision='int8'
            ),
            DeviceType.TPU: DeviceProfile(
                device_type=DeviceType.TPU,
                compute_capability=100.0,
                memory_bandwidth=900.0,
                memory_capacity=32.0,
                power_budget=450.0,
                precision='bfloat16'
            ),
            DeviceType.FPGA: DeviceProfile(
                device_type=DeviceType.FPGA,
                compute_capability=2.0,
                memory_bandwidth=50.0,
                memory_capacity=8.0,
                power_budget=50.0,
                precision='int8'
            )
        }
        return profiles.get(device_type, profiles[DeviceType.CPU])

    def evaluate_architecture(self,
                            architecture: Union[NASCell, MacroArchitecture],
                            batch_size: int = 1,
                            input_size: int = 224) -> Dict[str, float]:
        """
        Evaluate architecture with hardware metrics.

        Args:
            architecture: Architecture to evaluate
            batch_size: Batch size
            input_size: Input size

        Returns:
            Dictionary with hardware metrics
        """
        metrics = {
            'latency_ms': self.latency_predictor.predict(architecture, batch_size, input_size),
            'energy_mj': self.energy_estimator.estimate(architecture, input_size),
            'memory_mb': self.memory_estimator.estimate_total_memory(
                architecture, batch_size, input_size),
            'param_memory_mb': self.memory_estimator.estimate_parameters_memory(architecture),
            'activation_memory_mb': self.memory_estimator.estimate_activation_memory(
                architecture, batch_size, input_size)
        }

        # Add throughput (images/second)
        if metrics['latency_ms'] > 0:
            metrics['throughput'] = (batch_size * 1000) / metrics['latency_ms']
        else:
            metrics['throughput'] = float('inf')

        # Add efficiency metrics
        if isinstance(architecture, NASCell):
            flops = architecture.get_flops(input_size)
        else:
            flops = sum(layer.channels * input_size * input_size
                       for layer in architecture.layers)

        metrics['flops'] = flops
        if metrics['energy_mj'] > 0:
            metrics['flops_per_joule'] = flops / (metrics['energy_mj'] / 1000)
        else:
            metrics['flops_per_joule'] = float('inf')

        return metrics

    def optimize(self,
                dataset: str,
                constraints: Dict[str, float],
                search_space: str = 'cell',
                optimization_method: str = 'evolutionary',
                max_evaluations: int = 1000) -> Dict[str, Any]:
        """
        Optimize architecture with hardware constraints.

        Args:
            dataset: Dataset name
            constraints: Hardware constraints (e.g., {'max_latency_ms': 10})
            search_space: Search space type
            optimization_method: Optimization algorithm to use
            max_evaluations: Maximum evaluations

        Returns:
            Optimization result with best architecture
        """
        # Create hardware-aware problem
        objectives = [
            NASObjective('accuracy', weight=0.7, minimize=False),
            NASObjective('latency_ms', weight=0.2, minimize=True),
            NASObjective('energy_mj', weight=0.1, minimize=True)
        ]

        problem = NASProblem(
            dataset=dataset,
            search_space=search_space,
            objectives=objectives,
            constraints=constraints,
            evaluation_strategy='proxy',
            max_evaluations=max_evaluations
        )

        # Create adapter
        adapter = NASAdapter(scalarization_method='weighted_sum')

        # Encode problem
        standardized = adapter.encode_problem(problem)

        # Run optimization (simplified for demonstration)
        # In practice, this would call Librex's optimize function
        best_solution = self._run_optimization(
            standardized, optimization_method, max_evaluations)

        # Decode result
        best_architecture = adapter.decode_solution(best_solution, problem)

        # Evaluate final architecture
        hw_metrics = self.evaluate_architecture(best_architecture)

        return {
            'architecture': best_architecture,
            'solution_vector': best_solution,
            'hardware_metrics': hw_metrics,
            'satisfies_constraints': self._check_constraints(hw_metrics, constraints)
        }

    def _run_optimization(self,
                         problem: StandardizedProblem,
                         method: str,
                         max_evaluations: int) -> np.ndarray:
        """
        Run optimization (placeholder for actual implementation).

        In practice, this would use Librex's optimization algorithms.
        """
        # Simple random search for demonstration
        best_solution = None
        best_score = float('-inf')

        for _ in range(min(100, max_evaluations)):
            # Generate random solution
            solution = np.random.uniform(0, 1, problem.dimension)

            # Scale to bounds
            if problem.bounds:
                for i, (low, high) in enumerate(problem.bounds):
                    solution[i] = low + solution[i] * (high - low)

            # Evaluate
            score = problem.objective_function(solution)

            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution

    def _check_constraints(self,
                          metrics: Dict[str, float],
                          constraints: Dict[str, float]) -> bool:
        """Check if metrics satisfy constraints."""
        for constraint_key, constraint_value in constraints.items():
            if constraint_key.startswith('max_'):
                metric_key = constraint_key[4:]  # Remove 'max_' prefix
                if metric_key in metrics:
                    if metrics[metric_key] > constraint_value:
                        return False
            elif constraint_key.startswith('min_'):
                metric_key = constraint_key[4:]  # Remove 'min_' prefix
                if metric_key in metrics:
                    if metrics[metric_key] < constraint_value:
                        return False

        return True