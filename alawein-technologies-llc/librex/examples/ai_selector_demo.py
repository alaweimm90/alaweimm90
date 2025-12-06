#!/usr/bin/env python3
"""
Demo of AI-powered method selection in Librex

This script demonstrates how the AI selector automatically chooses
the best optimization method based on problem characteristics.
"""

import logging
import time
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from Librex import optimize
from Librex.adapters.qap import QAPAdapter
from Librex.ai import MethodSelector, ProblemFeatureExtractor
from Librex.benchmarks import load_qap_instance
from Librex.core.interfaces import StandardizedProblem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_automatic_selection():
    """Demonstrate automatic method selection with method='auto'"""
    print("\n" + "="*60)
    print("DEMO 1: Automatic Method Selection with method='auto'")
    print("="*60)

    # Create different problem types
    problems = [
        {
            'name': 'Small QAP (n=12)',
            'problem': load_qap_instance('tai12a'),
            'adapter': QAPAdapter()
        },
        {
            'name': 'Medium QAP (n=20)',
            'problem': load_qap_instance('tai20a'),
            'adapter': QAPAdapter()
        },
        {
            'name': 'Random Dense Matrix (n=15)',
            'problem': StandardizedProblem(
                dimension=15,
                objective_matrix=np.random.rand(15, 15) * 100,
                objective_function=lambda x: np.sum(x ** 2)
            ),
            'adapter': None
        },
        {
            'name': 'Sparse Problem (n=50)',
            'problem': StandardizedProblem(
                dimension=50,
                objective_matrix=np.random.choice([0, 1], size=(50, 50), p=[0.9, 0.1]) * 10,
                objective_function=lambda x: np.sum(x)
            ),
            'adapter': None
        }
    ]

    for prob_info in problems:
        print(f"\n--- {prob_info['name']} ---")

        # Run with automatic method selection
        start = time.time()
        result = optimize(
            problem=prob_info['problem'],
            adapter=prob_info['adapter'],
            method='auto',  # AI will choose the best method
            config={'iterations': 1000}  # Override iterations for demo speed
        )
        elapsed = time.time() - start

        print(f"  AI selected method: Check logs above")
        print(f"  Best objective: {result['objective']:.2f}")
        print(f"  Time taken: {elapsed:.2f}s")
        print(f"  Iterations: {result.get('iterations', 'N/A')}")


def demo_method_comparison():
    """Compare AI recommendation with all available methods"""
    print("\n" + "="*60)
    print("DEMO 2: Method Comparison and Recommendations")
    print("="*60)

    # Load a benchmark problem
    problem = load_qap_instance('tai15a')
    adapter = QAPAdapter()

    # Get AI recommendations
    selector = MethodSelector()

    print("\n--- AI Method Rankings for tai15a (n=15) ---")
    recommendations = selector.recommend_top_k(
        problem=problem,
        adapter=adapter,
        k=5
    )

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.method_name.upper()}")
        print(f"   Confidence: {rec.confidence:.1%}")
        print(f"   Reasoning: {rec.explanation}")
        print(f"   Evidence:")
        for evidence in rec.supporting_evidence[:3]:  # Show top 3 pieces of evidence
            print(f"     - {evidence}")

    # Now actually run the top 2 methods
    print("\n--- Running Top 2 Recommended Methods ---")
    methods_to_test = [recommendations[0].method_name, recommendations[1].method_name]

    for method in methods_to_test:
        print(f"\n{method.upper()}:")
        start = time.time()
        result = optimize(
            problem=problem,
            adapter=adapter,
            method=method,
            config={'iterations': 5000}  # Fixed iterations for fair comparison
        )
        elapsed = time.time() - start

        print(f"  Objective: {result['objective']:.2f}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Valid: {result.get('is_valid', True)}")


def demo_feature_extraction():
    """Demonstrate problem feature extraction"""
    print("\n" + "="*60)
    print("DEMO 3: Problem Feature Extraction")
    print("="*60)

    extractor = ProblemFeatureExtractor()

    # Different problem types
    problems = [
        ('Small symmetric', np.array([[0, 5, 3], [5, 0, 2], [3, 2, 0]])),
        ('Large sparse', np.random.choice([0, 1], size=(100, 100), p=[0.95, 0.05])),
        ('Dense random', np.random.rand(20, 20) * 100)
    ]

    for name, matrix in problems:
        problem = StandardizedProblem(
            dimension=matrix.shape[0],
            objective_matrix=matrix,
            objective_function=lambda x: np.sum(x)
        )

        features = extractor.extract_features(
            problem=None,
            standardized_problem=problem
        )

        print(f"\n--- {name} (n={features.dimension}) ---")
        print(f"  Sparsity: {features.matrix_sparsity:.1%}")
        print(f"  Symmetry: {features.matrix_symmetry:.1%}")
        print(f"  Connectivity: {features.connectivity:.2f}")
        print(f"  Value range: {features.value_range:.2f}")
        print(f"  Estimated difficulty: {features.estimated_difficulty}")


def demo_recommendation_explanation():
    """Demonstrate detailed recommendation explanations"""
    print("\n" + "="*60)
    print("DEMO 4: Recommendation Explanations")
    print("="*60)

    selector = MethodSelector()

    # Create different problem scenarios
    scenarios = [
        {
            'name': 'Quick Solution Needed',
            'problem': StandardizedProblem(
                dimension=30,
                objective_matrix=np.random.rand(30, 30),
                objective_function=lambda x: np.sum(x)
            ),
            'time_budget': 0.5,
            'quality': 'fast'
        },
        {
            'name': 'Best Quality Required',
            'problem': StandardizedProblem(
                dimension=25,
                objective_matrix=np.random.rand(25, 25),
                objective_function=lambda x: np.sum(x ** 2),
                constraint_matrix=np.random.rand(10, 25)
            ),
            'time_budget': None,
            'quality': 'best'
        },
        {
            'name': 'Very Large Problem',
            'problem': StandardizedProblem(
                dimension=500,
                objective_matrix=None,
                objective_function=lambda x: np.sum(x)
            ),
            'time_budget': 10.0,
            'quality': 'balanced'
        }
    ]

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['name']} ---")

        method, config, confidence = selector.recommend_method(
            problem=None,
            standardized_problem=scenario['problem'],
            time_budget=scenario.get('time_budget'),
            quality_requirement=scenario['quality']
        )

        print(f"Recommended method: {method.upper()}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Suggested iterations: {config.get('iterations', 'default')}")

        # Get detailed explanation
        explanation = selector.explain_recommendation(
            problem=None,
            standardized_problem=scenario['problem'],
            method=method
        )
        print("\nDetailed Explanation:")
        print(explanation)


def demo_performance_tracking():
    """Demonstrate performance tracking for ML training"""
    print("\n" + "="*60)
    print("DEMO 5: Performance Tracking for Future ML Training")
    print("="*60)

    import tempfile
    log_file = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
    logger.info(f"Performance log: {log_file.name}")

    selector = MethodSelector(performance_log_path=log_file.name)

    # Run several optimizations and track performance
    problems = [
        load_qap_instance('tai12a'),
        load_qap_instance('tai12b'),
        load_qap_instance('tai15a')
    ]

    adapter = QAPAdapter()
    methods = ['simulated_annealing', 'genetic_algorithm', 'tabu_search']

    print("\nRunning optimizations and tracking performance...")
    for problem in problems:
        for method in methods:
            print(f"  Testing {method} on problem...")

            start = time.time()
            result = optimize(
                problem=problem,
                adapter=adapter,
                method=method,
                config={'iterations': 1000}  # Quick test
            )
            runtime = time.time() - start

            # Record performance
            selector.record_performance(
                problem=problem,
                adapter=adapter,
                method=method,
                config={'iterations': 1000},
                result=result,
                runtime_seconds=runtime
            )

    print(f"\nPerformance data saved to: {log_file.name}")
    print("This data can be used to train ML models for better recommendations")

    # Demonstrate data analysis
    from Librex.ai.models import DataCollector
    collector = DataCollector(log_file.name)
    best_methods = collector.get_best_methods(min_samples=1)

    print("\nAnalysis of collected data:")
    for problem_type, methods in best_methods.items():
        print(f"  {problem_type}: Best methods are {', '.join(methods)}")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════╗")
    print("║    Librex AI Method Selector Demonstration        ║")
    print("╚════════════════════════════════════════════════════════╝")

    # Run all demos
    demo_automatic_selection()
    demo_method_comparison()
    demo_feature_extraction()
    demo_recommendation_explanation()
    demo_performance_tracking()

    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60)