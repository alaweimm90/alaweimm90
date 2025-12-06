#!/usr/bin/env python3
"""
Quick start guide for using AI method selection in Librex
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from Librex import optimize
from Librex.adapters.qap import QAPAdapter
from Librex.ai import MethodSelector
from Librex.benchmarks import load_qap_instance


def example1_automatic_selection():
    """Example 1: Let AI choose the best method automatically"""
    print("Example 1: Automatic Method Selection")
    print("-" * 40)

    # Load a QAP problem
    problem = load_qap_instance('tai12a')
    adapter = QAPAdapter()

    # Use method='auto' to let AI choose
    result = optimize(
        problem=problem,
        adapter=adapter,
        method='auto'  # AI will choose the best method!
    )

    print(f"Best solution found: {result['solution']}")
    print(f"Objective value: {result['objective']}")
    print()


def example2_get_recommendations():
    """Example 2: Get method recommendations with explanations"""
    print("Example 2: Get Method Recommendations")
    print("-" * 40)

    # Create a problem
    problem = {
        'flow_matrix': np.array([[0, 5, 3], [5, 0, 2], [3, 2, 0]]),
        'distance_matrix': np.array([[0, 8, 4], [8, 0, 6], [4, 6, 0]])
    }
    adapter = QAPAdapter()

    # Create selector
    selector = MethodSelector()

    # Get top 3 recommendations
    recommendations = selector.recommend_top_k(
        problem=problem,
        adapter=adapter,
        k=3
    )

    print("Top 3 recommended methods:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.method_name} (confidence: {rec.confidence:.1%})")
        print(f"   Reason: {rec.supporting_evidence[0] if rec.supporting_evidence else 'Default heuristic'}")
    print()


def example3_time_constrained():
    """Example 3: Get recommendation for time-constrained optimization"""
    print("Example 3: Time-Constrained Optimization")
    print("-" * 40)

    from Librex.core.interfaces import StandardizedProblem

    # Create a large problem
    problem = StandardizedProblem(
        dimension=100,
        objective_matrix=np.random.rand(100, 100),
        objective_function=lambda x: np.sum(x ** 2)
    )

    selector = MethodSelector()

    # Get fast recommendation (0.5 second time budget)
    method, config, confidence = selector.recommend_method(
        problem=None,
        standardized_problem=problem,
        time_budget=0.5,
        quality_requirement='fast'
    )

    print(f"For quick results, use: {method}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Suggested iterations: {config.get('iterations', 'default')}")
    print()


def example4_explain_recommendation():
    """Example 4: Get detailed explanation for a recommendation"""
    print("Example 4: Detailed Explanation")
    print("-" * 40)

    from Librex.core.interfaces import StandardizedProblem

    # Create a problem with constraints
    problem = StandardizedProblem(
        dimension=20,
        objective_matrix=np.random.rand(20, 20),
        objective_function=lambda x: np.sum(x),
        constraint_matrix=np.random.rand(5, 20)  # Has constraints
    )

    selector = MethodSelector()

    # Get explanation
    explanation = selector.explain_recommendation(
        problem=None,
        standardized_problem=problem
    )

    print(explanation)
    print()


def example5_compare_methods():
    """Example 5: Compare AI recommendation with manual selection"""
    print("Example 5: AI vs Manual Method Selection")
    print("-" * 40)

    # Load problem
    problem = load_qap_instance('tai15a')
    adapter = QAPAdapter()

    # Run with AI selection
    print("Running with AI selection (method='auto')...")
    result_auto = optimize(
        problem=problem,
        adapter=adapter,
        method='auto',
        config={'iterations': 5000}
    )

    # Run with manual selection
    print("Running with manual selection (simulated_annealing)...")
    result_manual = optimize(
        problem=problem,
        adapter=adapter,
        method='simulated_annealing',
        config={'iterations': 5000}
    )

    print(f"\nResults:")
    print(f"AI selection objective: {result_auto['objective']:.2f}")
    print(f"Manual selection objective: {result_manual['objective']:.2f}")

    if result_auto['objective'] < result_manual['objective']:
        improvement = (result_manual['objective'] - result_auto['objective']) / result_manual['objective'] * 100
        print(f"AI selection was {improvement:.1f}% better!")
    else:
        print("Manual selection performed better this time")


if __name__ == "__main__":
    print("═══════════════════════════════════════")
    print("  Librex AI Selector Quick Start  ")
    print("═══════════════════════════════════════")
    print()

    example1_automatic_selection()
    example2_get_recommendations()
    example3_time_constrained()
    example4_explain_recommendation()
    example5_compare_methods()

    print("═══════════════════════════════════════")
    print("Quick start examples completed!")
    print("═══════════════════════════════════════")