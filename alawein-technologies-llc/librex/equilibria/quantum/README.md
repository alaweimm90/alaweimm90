# Quantum Computing Integration for Librex

## Overview

The quantum module provides quantum optimization methods and converters for solving combinatorial optimization problems using quantum computing. This module is designed to be completely optional and provides graceful fallback when quantum libraries are not installed.

## Features

### 1. Problem Converters
- **QUBO Converter**: Converts classical problems to Quadratic Unconstrained Binary Optimization format
- **Ising Encoder**: Encodes problems as Ising Hamiltonians for quantum annealing
- **Quantum Adapter**: High-level interface for problem conversion

### 2. Quantum Methods
- **Quantum Annealing**: D-Wave style quantum annealing (simulated)
- **QAOA**: Quantum Approximate Optimization Algorithm
- **VQE**: Variational Quantum Eigensolver

### 3. Validators & Estimators
- **Problem Validator**: Checks if problems are suitable for quantum computing
- **Qubit Estimator**: Estimates quantum resource requirements
- **Feasibility Analysis**: Determines NISQ device compatibility

### 4. Utilities
- **State Decoder**: Converts quantum measurements to classical solutions
- **Hamiltonian Builder**: Constructs quantum Hamiltonians
- **Result Converter**: Translates quantum results to standard format

## Installation

### Basic Installation (Classical Only)
```bash
pip install Librex
```

### With Quantum Support
```bash
pip install Librex[quantum]
```

This installs:
- `qiskit>=0.40.0` - IBM's quantum computing framework
- `pennylane>=0.30.0` - Cross-platform quantum machine learning library

## Usage Examples

### 1. Check Quantum Availability

```python
from Librex.quantum import check_quantum_availability

# Check if any quantum library is available
if check_quantum_availability():
    print("Quantum features available!")

# Check specific library
if check_quantum_availability('qiskit'):
    print("Qiskit is available")
```

### 2. Convert Problem to QUBO

```python
from Librex.quantum.adapters import QUBOConverter
import numpy as np

# Example: Convert QAP to QUBO
flow_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
distance_matrix = np.array([[0, 2, 3], [2, 0, 1], [3, 1, 0]])

converter = QUBOConverter(penalty_weight=1000.0)
qubo_problem = converter.convert_qap_to_qubo(flow_matrix, distance_matrix)

print(f"QUBO size: {qubo_problem.num_variables} variables")
print(f"QUBO matrix shape: {qubo_problem.Q.shape}")
```

### 3. Validate Problem for Quantum

```python
from Librex.quantum.validators import QuantumProblemValidator
from Librex.core.interfaces import StandardizedProblem

# Create a problem
problem = StandardizedProblem(
    dimension=16,
    objective_matrix=np.random.randn(16, 16),
    problem_metadata={'problem_type': 'QAP', 'n_facilities': 4}
)

# Validate for NISQ devices
validator = QuantumProblemValidator(hardware_type='nisq')
validation = validator.validate_problem(problem)

print(f"Suitable for quantum: {validation['is_suitable']}")
print(f"Required qubits: {validation['required_qubits']}")
print(f"Recommendations: {validation['recommendations']}")
```

### 4. Estimate Qubit Requirements

```python
from Librex.quantum.validators import QubitEstimator

estimator = QubitEstimator()
estimate = estimator.estimate_qubits(problem, encoding='binary')

print(f"Logical qubits: {estimate['logical_qubits']}")
print(f"Ancilla qubits: {estimate['ancilla_qubits']}")
print(f"Total qubits: {estimate['total_qubits']}")
print(f"NISQ feasible: {estimate['feasibility']['nisq_feasible']}")
```

### 5. Run Quantum Optimization (QAOA)

```python
from Librex.quantum.methods import qaoa_optimize

# Configure QAOA
config = {
    'p_layers': 3,          # Number of QAOA layers
    'optimizer': 'COBYLA',  # Classical optimizer
    'max_iter': 100,        # Optimization iterations
    'shots': 1024,          # Measurement shots
    'seed': 42
}

# Run QAOA (will use simulation if no quantum hardware)
result = qaoa_optimize(problem, config)

print(f"Best solution: {result['solution']}")
print(f"Objective value: {result['objective']}")
print(f"Valid solution: {result['is_valid']}")
```

### 6. Quantum Annealing

```python
from Librex.quantum.methods import quantum_annealing_optimize

# Configure annealing
config = {
    'num_reads': 1000,      # Number of annealing runs
    'annealing_time': 20,   # Microseconds
    'auto_scale': True,     # Auto-scale coefficients
    'seed': 42
}

# Run quantum annealing (simulated)
result = quantum_annealing_optimize(problem, config)

print(f"Best solution: {result['solution']}")
print(f"Energy: {result['objective']}")
```

### 7. VQE Optimization

```python
from Librex.quantum.methods import vqe_optimize

# Configure VQE
config = {
    'ansatz': 'hardware_efficient',
    'ansatz_layers': 2,
    'optimizer': 'SLSQP',
    'max_iter': 200,
    'shots': 1024,
    'seed': 42
}

# Run VQE
result = vqe_optimize(problem, config)

print(f"Ground state energy: {result['objective']}")
print(f"Optimal parameters: {result['optimal_params']}")
```

### 8. Complete Workflow Example

```python
import numpy as np
from Librex.quantum import (
    QuantumProblemAdapter,
    QuantumProblemValidator,
    QubitEstimator,
    qaoa_optimize
)
from Librex.core.interfaces import StandardizedProblem

# 1. Create problem
n = 4  # 4x4 QAP
flow = np.random.randn(n, n)
distance = np.random.randn(n, n)

problem = StandardizedProblem(
    dimension=n*n,
    problem_metadata={
        'problem_type': 'QAP',
        'flow_matrix': flow,
        'distance_matrix': distance,
        'n_facilities': n
    }
)

# 2. Validate for quantum
validator = QuantumProblemValidator('nisq')
validation = validator.validate_problem(problem)

if validation['is_suitable']:
    print("✓ Problem suitable for quantum optimization")

    # 3. Estimate resources
    estimator = QubitEstimator()
    resources = estimator.estimate_qubits(problem)
    print(f"  Required qubits: {resources['total_qubits']}")

    # 4. Get algorithm recommendation
    recommendation = validator.recommend_quantum_algorithm(problem)
    print(f"  Recommended algorithm: {recommendation['algorithm']}")

    # 5. Convert to quantum format
    adapter = QuantumProblemAdapter()
    quantum_problem = adapter.convert_to_quantum(problem, target_format='ising')

    # 6. Run quantum optimization
    result = qaoa_optimize(problem, {
        'p_layers': recommendation['parameters'].get('p_layers', 3),
        'shots': 1024
    })

    print(f"\nSolution found:")
    print(f"  Assignment: {result['solution']}")
    print(f"  Objective: {result['objective']:.4f}")
    print(f"  Valid: {result['is_valid']}")
else:
    print("✗ Problem not suitable for current quantum hardware")
    for issue in validation['issues']:
        print(f"  - {issue}")
```

## Architecture

### Module Structure

```
Librex/quantum/
├── __init__.py              # Main entry, availability checking
├── adapters/                # Problem converters
│   ├── qubo_converter.py   # QUBO conversion
│   ├── ising_encoder.py    # Ising model encoding
│   └── quantum_adapter.py  # High-level adapter
├── methods/                 # Quantum algorithms
│   ├── quantum_annealing.py # D-Wave style annealing
│   ├── qaoa.py             # QAOA implementation
│   └── vqe.py              # VQE implementation
├── validators/              # Feasibility checking
│   ├── problem_validator.py # Problem suitability
│   └── qubit_estimator.py  # Resource estimation
└── utils/                   # Helper utilities
    ├── state_decoder.py     # Measurement decoding
    ├── hamiltonian_builder.py # Hamiltonian construction
    └── result_converter.py  # Result translation
```

### Design Principles

1. **Optional Dependencies**: Quantum features are completely optional
2. **Graceful Degradation**: Clear error messages when libraries missing
3. **Hardware Agnostic**: Supports multiple quantum platforms
4. **Future Ready**: Prepared for real quantum hardware integration
5. **Mathematical Rigor**: Correct quantum formulations

## Problem Support

### Currently Supported Problems

| Problem Type | QUBO | Ising | Quantum Annealing | QAOA | VQE |
|-------------|------|-------|------------------|------|-----|
| QAP | ✓ | ✓ | ✓ | ✓ | ✓ |
| TSP | ✓ | ✓ | ✓ | ✓ | ✓ |
| Max-Cut | ✓ | ✓ | ✓ | ✓ | ✓ |
| Generic Quadratic | ✓ | ✓ | ✓ | ✓ | ✓ |

### Qubit Requirements

| Problem | Size | Binary Variables | Qubits Required | NISQ Feasible |
|---------|------|-----------------|-----------------|---------------|
| QAP | n=4 | 16 | 16 | ✓ |
| QAP | n=5 | 25 | 25 | ✓ |
| QAP | n=7 | 49 | 49 | ✓ |
| QAP | n=8 | 64 | 64 | ✗ |
| TSP | n=4 | 16 | 16 | ✓ |
| TSP | n=6 | 36 | 36 | ✓ |
| TSP | n=8 | 64 | 64 | ✗ |
| Max-Cut | n=20 | 20 | 20 | ✓ |
| Max-Cut | n=50 | 50 | 50 | ✓ |
| Max-Cut | n=100 | 100 | 100 | ✗ |

## Hardware Compatibility

### Current Support
- **Simulation**: Full support via statevector simulation
- **NISQ Devices**: Prepared for ~50 qubit devices
- **Quantum Annealers**: Ready for D-Wave integration (up to 5000 qubits)

### Future Integration (TODOs)
- Real Qiskit circuit implementation
- D-Wave Ocean SDK integration
- PennyLane device support
- AWS Braket compatibility
- Azure Quantum support
- Google Cirq integration

## Performance Considerations

### Classical Simulation Limits
- **Exact simulation**: Up to ~30 qubits
- **Approximate methods**: Up to ~50 qubits
- **Tensor networks**: Potentially hundreds of qubits (structure dependent)

### Quantum Advantage Threshold
Problems need to be large enough to benefit from quantum:
- QAP: n ≥ 10 (100+ variables)
- TSP: n ≥ 15 (225+ variables)
- Max-Cut: Dense graphs with 100+ vertices

### Optimization Tips
1. Use problem-specific encodings to minimize qubits
2. Apply coefficient scaling to improve numerical stability
3. Choose appropriate penalty weights for constraints
4. Consider hybrid quantum-classical approaches
5. Use error mitigation for NISQ devices

## Troubleshooting

### Common Issues

1. **ImportError: Quantum features require quantum libraries**
   - Solution: Install with `pip install Librex[quantum]`

2. **Problem too large for NISQ**
   - Solution: Use problem decomposition or classical methods

3. **Poor solution quality**
   - Solution: Increase shots, adjust penalty weights, try different algorithms

4. **Numerical instability**
   - Solution: Enable auto-scaling, rescale coefficients

## Contributing

We welcome contributions to improve quantum integration:

1. Implement real quantum circuits with Qiskit/PennyLane
2. Add support for more problem types
3. Improve encoding efficiency
4. Add hardware-specific optimizations
5. Implement error mitigation techniques

## References

- [QAOA Original Paper](https://arxiv.org/abs/1411.4028)
- [VQE Algorithm](https://arxiv.org/abs/1304.3061)
- [Quantum Annealing](https://arxiv.org/abs/cond-mat/9804280)
- [QUBO Formulations](https://arxiv.org/abs/1811.11538)

## License

Apache 2.0 - See LICENSE file for details

## Author

Meshal Alawein (meshal@berkeley.edu)

---

*Quantum optimization: Bridging classical algorithms with quantum computing*