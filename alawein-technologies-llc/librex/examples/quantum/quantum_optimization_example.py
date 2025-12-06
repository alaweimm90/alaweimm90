#!/usr/bin/env python3
"""
Quantum Optimization Example

Demonstrates using quantum computing methods for solving optimization problems
with Librex's quantum integration.

Author: Meshal Alawein
Date: 2025-11-18
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def demonstrate_quantum_availability():
    """Check and display quantum library availability."""
    print("=" * 60)
    print("QUANTUM LIBRARY AVAILABILITY CHECK")
    print("=" * 60)

    from Librex.quantum import (
        check_quantum_availability,
        QISKIT_AVAILABLE,
        PENNYLANE_AVAILABLE
    )

    print(f"Qiskit available: {QISKIT_AVAILABLE}")
    print(f"PennyLane available: {PENNYLANE_AVAILABLE}")
    print(f"Any quantum library: {check_quantum_availability()}")
    print()


def demonstrate_qubo_conversion():
    """Demonstrate converting optimization problems to QUBO format."""
    print("=" * 60)
    print("QUBO CONVERSION DEMONSTRATION")
    print("=" * 60)

    from Librex.quantum.adapters import QUBOConverter

    # Small QAP instance
    n = 3
    flow_matrix = np.array([
        [0, 3, 2],
        [3, 0, 4],
        [2, 4, 0]
    ])
    distance_matrix = np.array([
        [0, 2, 5],
        [2, 0, 3],
        [5, 3, 0]
    ])

    print(f"QAP Problem Size: {n}x{n}")
    print("Flow Matrix:")
    print(flow_matrix)
    print("\nDistance Matrix:")
    print(distance_matrix)

    # Convert to QUBO
    converter = QUBOConverter(penalty_weight=100.0)
    qubo_problem = converter.convert_qap_to_qubo(flow_matrix, distance_matrix)

    print(f"\nQUBO Conversion Results:")
    print(f"  Number of binary variables: {qubo_problem.num_variables}")
    print(f"  QUBO matrix shape: {qubo_problem.Q.shape}")
    print(f"  Matrix sparsity: {1 - np.count_nonzero(qubo_problem.Q) / qubo_problem.Q.size:.2%}")
    print(f"  Offset: {qubo_problem.offset}")

    # Test solution evaluation
    test_solution = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])  # Identity permutation
    objective = qubo_problem.evaluate(test_solution)
    print(f"\nTest solution objective: {objective}")
    print()


def demonstrate_ising_encoding():
    """Demonstrate Ising model encoding."""
    print("=" * 60)
    print("ISING MODEL ENCODING")
    print("=" * 60)

    from Librex.quantum.adapters import QUBOConverter, IsingEncoder

    # Create a Max-Cut problem
    adjacency = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])

    print("Max-Cut Graph (Adjacency Matrix):")
    print(adjacency)
    print(f"Number of vertices: {len(adjacency)}")
    print(f"Number of edges: {np.count_nonzero(adjacency) // 2}")

    # Convert to QUBO then Ising
    converter = QUBOConverter()
    encoder = IsingEncoder()

    qubo_problem = converter.convert_maxcut_to_qubo(adjacency)
    ising_problem = encoder.qubo_to_ising(qubo_problem)

    print(f"\nIsing Model:")
    print(f"  Number of spins: {ising_problem.num_spins}")
    print(f"  Local fields (h): {ising_problem.h}")
    print(f"  Couplings (J):")
    print(ising_problem.J)
    print()


def demonstrate_problem_validation():
    """Demonstrate quantum problem validation."""
    print("=" * 60)
    print("QUANTUM PROBLEM VALIDATION")
    print("=" * 60)

    from Librex.quantum.validators import QuantumProblemValidator
    from Librex.core.interfaces import StandardizedProblem

    # Create problems of different sizes
    sizes = [3, 5, 8, 10]

    for n in sizes:
        problem = StandardizedProblem(
            dimension=n*n,
            objective_matrix=np.random.randn(n*n, n*n),
            problem_metadata={
                'problem_type': 'QAP',
                'n_facilities': n,
                'flow_matrix': np.random.randn(n, n),
                'distance_matrix': np.random.randn(n, n)
            }
        )

        validator = QuantumProblemValidator(hardware_type='nisq')
        validation = validator.validate_problem(problem)

        print(f"\nQAP n={n} ({n*n} qubits):")
        print(f"  NISQ feasible: {validation['size_feasible']}")
        print(f"  Ideal size: {validation['ideal_size']}")
        print(f"  Classically simulatable: {validation['classically_simulatable']}")

        if validation['recommendations']:
            print(f"  Recommendations:")
            for rec in validation['recommendations'][:2]:
                print(f"    - {rec}")


def demonstrate_qubit_estimation():
    """Demonstrate qubit requirement estimation."""
    print("=" * 60)
    print("QUBIT REQUIREMENT ESTIMATION")
    print("=" * 60)

    from Librex.quantum.validators import QubitEstimator
    from Librex.core.interfaces import StandardizedProblem

    estimator = QubitEstimator()

    # Compare different problem types
    problem_configs = [
        ('QAP', 5, 25),
        ('TSP', 6, 36),
        ('Max-Cut', 20, 20)
    ]

    for prob_type, size, dim in problem_configs:
        metadata = {'problem_type': prob_type}
        if prob_type in ['QAP', 'TSP']:
            metadata['n_facilities'] = size
            metadata['n_cities'] = size
        else:
            metadata['n_vertices'] = size

        problem = StandardizedProblem(
            dimension=dim,
            problem_metadata=metadata
        )

        # Estimate for different encodings
        print(f"\n{prob_type} (size={size}):")
        for encoding in ['binary', 'gray']:
            try:
                estimate = estimator.estimate_qubits(problem, encoding)
                print(f"  {encoding:8s} encoding: {estimate['total_qubits']:3d} qubits "
                      f"(efficiency: {estimate['encoding_efficiency']:.2%})")
            except:
                print(f"  {encoding:8s} encoding: Not supported")

        # Circuit depth estimation
        depth_info = estimator.estimate_circuit_depth(problem, algorithm='qaoa')
        print(f"  QAOA circuit depth: {depth_info.get('total_depth', 'N/A')}")


def demonstrate_quantum_methods():
    """Demonstrate quantum optimization methods."""
    print("=" * 60)
    print("QUANTUM OPTIMIZATION METHODS")
    print("=" * 60)

    from Librex.core.interfaces import StandardizedProblem

    # Create a small TSP instance
    n = 4
    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])

    problem = StandardizedProblem(
        dimension=n*n,
        objective_matrix=distance_matrix,
        problem_metadata={
            'problem_type': 'TSP',
            'n_cities': n,
            'distance_matrix': distance_matrix
        }
    )

    print(f"TSP Problem with {n} cities")
    print("Distance Matrix:")
    print(distance_matrix)
    print()

    # Try each quantum method (will use simulation)
    methods = [
        ('quantum_annealing', {
            'num_reads': 100,
            'annealing_time': 20,
            'seed': 42
        }),
        ('qaoa', {
            'p_layers': 2,
            'max_iter': 50,
            'shots': 512,
            'seed': 42
        }),
        ('vqe', {
            'ansatz_layers': 1,
            'max_iter': 50,
            'shots': 512,
            'seed': 42
        })
    ]

    for method_name, config in methods:
        print(f"\n{method_name.upper()}:")
        try:
            if method_name == 'quantum_annealing':
                from Librex.quantum.methods import quantum_annealing_optimize
                result = quantum_annealing_optimize(problem, config)
            elif method_name == 'qaoa':
                from Librex.quantum.methods import qaoa_optimize
                result = qaoa_optimize(problem, config)
            elif method_name == 'vqe':
                from Librex.quantum.methods import vqe_optimize
                result = vqe_optimize(problem, config)

            print(f"  Solution: {result['solution']}")
            print(f"  Objective: {result['objective']:.2f}")
            print(f"  Valid: {result['is_valid']}")
            if 'convergence' in result:
                print(f"  Converged: {result['convergence'].get('converged', 'N/A')}")

        except ImportError as e:
            print(f"  Skipped: {e}")
        except Exception as e:
            print(f"  Error: {e}")


def demonstrate_complete_workflow():
    """Demonstrate complete quantum optimization workflow."""
    print("=" * 60)
    print("COMPLETE QUANTUM WORKFLOW")
    print("=" * 60)

    from Librex.quantum import check_quantum_availability

    if not check_quantum_availability():
        print("Note: Quantum libraries not installed.")
        print("Using classical simulation of quantum methods.")
        print("Install with: pip install Librex[quantum]")
        print()

    from Librex.quantum.adapters import QuantumProblemAdapter
    from Librex.quantum.validators import QuantumProblemValidator, QubitEstimator
    from Librex.core.interfaces import StandardizedProblem

    # Create a QAP problem
    n = 4
    np.random.seed(42)
    flow = np.random.randint(0, 10, (n, n))
    flow = (flow + flow.T) / 2  # Symmetric
    np.fill_diagonal(flow, 0)

    distance = np.random.randint(1, 10, (n, n))
    distance = (distance + distance.T) / 2  # Symmetric
    np.fill_diagonal(distance, 0)

    print(f"Generated {n}x{n} QAP instance")

    # Create standardized problem
    problem = StandardizedProblem(
        dimension=n*n,
        problem_metadata={
            'problem_type': 'QAP',
            'flow_matrix': flow,
            'distance_matrix': distance,
            'n_facilities': n
        }
    )

    # Step 1: Validate problem
    print("\n1. Validating problem for quantum...")
    validator = QuantumProblemValidator('nisq')
    validation = validator.validate_problem(problem)
    print(f"   Suitable: {validation['is_suitable']}")
    print(f"   Qubits required: {validation['required_qubits']}")

    # Step 2: Estimate resources
    print("\n2. Estimating quantum resources...")
    estimator = QubitEstimator()
    resources = estimator.estimate_qubits(problem)
    circuit_depth = estimator.estimate_circuit_depth(problem, 'qaoa')
    print(f"   Total qubits: {resources['total_qubits']}")
    print(f"   QAOA depth: {circuit_depth.get('total_depth', 'N/A')}")
    print(f"   Encoding efficiency: {resources['encoding_efficiency']:.1%}")

    # Step 3: Get algorithm recommendation
    print("\n3. Getting algorithm recommendation...")
    recommendation = validator.recommend_quantum_algorithm(problem)
    print(f"   Recommended: {recommendation['algorithm']}")
    for reason in recommendation['reasoning']:
        print(f"   - {reason}")

    # Step 4: Convert to quantum format
    print("\n4. Converting to quantum format...")
    adapter = QuantumProblemAdapter()

    # Try both formats
    qubo_problem = adapter.convert_to_quantum(problem, target_format='qubo')
    print(f"   QUBO variables: {qubo_problem.num_variables}")

    ising_problem = adapter.convert_to_quantum(problem, target_format='ising')
    print(f"   Ising spins: {ising_problem.num_spins}")

    # Step 5: Validate quantum problem
    print("\n5. Validating quantum problem...")
    quantum_validation = adapter.validate_quantum_problem(qubo_problem)
    print(f"   Valid: {quantum_validation['valid']}")
    if quantum_validation['warnings']:
        print(f"   Warnings: {quantum_validation['warnings'][0]}")

    # Step 6: Run optimization (if suitable)
    if validation['is_suitable']:
        print("\n6. Running quantum optimization...")
        try:
            from Librex.quantum.methods import qaoa_optimize

            result = qaoa_optimize(problem, {
                'p_layers': 2,
                'max_iter': 30,
                'shots': 256,
                'seed': 42
            })

            print(f"   Solution found: {result['solution']}")
            print(f"   Objective: {result['objective']:.2f}")
            print(f"   Valid: {result['is_valid']}")

            # Verify solution
            assignment = result['solution']
            total_cost = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        total_cost += flow[i, j] * distance[assignment[i], assignment[j]]
            print(f"   Verified cost: {total_cost}")

        except ImportError as e:
            print(f"   Skipped: Quantum libraries not available")
    else:
        print("\n6. Problem too large for current quantum hardware")
        print("   Would use classical optimization instead")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Librex QUANTUM INTEGRATION DEMONSTRATION")
    print("=" * 60)
    print("Author: Meshal Alawein")
    print("Date: 2025-11-18")
    print()

    # Run demonstrations
    demonstrate_quantum_availability()
    demonstrate_qubo_conversion()
    demonstrate_ising_encoding()
    demonstrate_problem_validation()
    demonstrate_qubit_estimation()
    demonstrate_quantum_methods()
    demonstrate_complete_workflow()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Quantum features are completely optional")
    print("2. Problems are validated for quantum feasibility")
    print("3. Multiple quantum algorithms are supported")
    print("4. Graceful fallback when libraries not installed")
    print("5. Ready for future quantum hardware integration")
    print()


if __name__ == "__main__":
    main()