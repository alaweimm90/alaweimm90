#!/usr/bin/env python3
"""
ML Recommender Integration Demo
--------------------------------
Demonstrates how to use the ML-based solver recommendation engine
with ORCHEX optimization workflows.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas_core.ml_recommender import (
    SolverRecommender,
    FeatureExtractor,
    ProblemFeatures,
    generate_synthetic_training_data
)


def main():
    """Main demo function."""
    print("=" * 60)
    print("MEZAN ML-Based Solver Recommendation Demo")
    print("=" * 60)

    # Initialize the ML recommender
    print("\n1. Initializing ML Recommender...")
    recommender = SolverRecommender(model_type='random_forest')
    print("   ✓ Recommender initialized with Random Forest model")

    # Generate synthetic training data
    print("\n2. Generating synthetic training data...")
    training_data = generate_synthetic_training_data(n_samples=500)
    print(f"   ✓ Generated {len(training_data)} training samples")

    # Split data
    train_size = int(0.8 * len(training_data))
    train_data = training_data[:train_size]
    test_data = training_data[train_size:]
    print(f"   ✓ Split into {len(train_data)} train, {len(test_data)} test samples")

    # Train the model
    print("\n3. Training the ML model...")
    train_metrics = recommender.train(train_data, perform_hyperopt=False, cv_folds=3)
    print(f"   ✓ Training accuracy: {train_metrics['accuracy']:.3f}")
    print(f"   ✓ Cross-validation accuracy: {train_metrics['cv_accuracy_mean']:.3f}")

    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_metrics = recommender.evaluate(test_data)
    print(f"   ✓ Test accuracy: {test_metrics['accuracy']:.3f}")

    # Example: Combinatorial optimization problem
    print("\n5. Example Predictions:")
    print("-" * 40)

    # Example 1: TSP-like problem
    tsp_problem = {
        'name': 'TSP-100',
        'type': 'combinatorial',
        'num_variables': 100,
        'num_constraints': 99,
        'num_objectives': 1,
        'constraints': {
            'linear': 99,
            'nonlinear': 0,
            'equality': 99,
        },
        'variables': {
            'continuous': 0,
            'discrete': 0,
            'binary': 100,
        },
        'objective': {
            'linearity': 1.0,
            'convexity': 0.0,
            'smoothness': 0.0,
            'multimodality': 0.8,
        },
        'graph': {
            'density': 1.0,  # Complete graph
            'connectivity': 1.0,
            'clustering': 0.33,
            'avg_degree': 99.0,
        }
    }

    print("\n   TSP Problem (100 cities):")
    solver, confidence = recommender.predict(tsp_problem, return_confidence=True)
    print(f"   Recommended solver: {solver}")
    print(f"   Confidence: {confidence:.2%}")

    # Get top-3 recommendations
    top_3 = recommender.predict_top_k(tsp_problem, k=3)
    print("   Top 3 solvers:")
    for i, (s, c) in enumerate(top_3, 1):
        print(f"     {i}. {s}: {c:.2%}")

    # Example 2: Convex optimization problem
    convex_problem = {
        'name': 'ConvexQP',
        'type': 'continuous',
        'num_variables': 50,
        'num_constraints': 30,
        'num_objectives': 1,
        'constraints': {
            'linear': 30,
            'nonlinear': 0,
            'equality': 10,
        },
        'variables': {
            'continuous': 50,
            'discrete': 0,
            'binary': 0,
        },
        'objective': {
            'linearity': 0.0,
            'convexity': 1.0,  # Fully convex
            'smoothness': 1.0,  # Smooth
            'multimodality': 0.0,  # Single optimum
        },
        'graph': {
            'density': 0.0,
            'connectivity': 0.0,
            'clustering': 0.0,
            'avg_degree': 0.0,
        }
    }

    print("\n   Convex Quadratic Problem:")
    solver, confidence = recommender.predict(convex_problem, return_confidence=True)
    print(f"   Recommended solver: {solver}")
    print(f"   Confidence: {confidence:.2%}")

    # Top-3 for convex problem
    top_3 = recommender.predict_top_k(convex_problem, k=3)
    print("   Top 3 solvers:")
    for i, (s, c) in enumerate(top_3, 1):
        print(f"     {i}. {s}: {c:.2%}")

    # Example 3: Non-convex multimodal problem
    multimodal_problem = {
        'name': 'Rastrigin',
        'type': 'continuous',
        'num_variables': 100,
        'num_constraints': 0,
        'num_objectives': 1,
        'constraints': {
            'linear': 0,
            'nonlinear': 0,
            'equality': 0,
        },
        'variables': {
            'continuous': 100,
            'discrete': 0,
            'binary': 0,
            'bounded': 100,
        },
        'objective': {
            'linearity': 0.0,
            'convexity': 0.0,  # Non-convex
            'smoothness': 0.8,
            'multimodality': 1.0,  # Highly multimodal
        },
        'graph': {
            'density': 0.0,
            'connectivity': 0.0,
            'clustering': 0.0,
            'avg_degree': 0.0,
        }
    }

    print("\n   Non-convex Multimodal Problem (Rastrigin):")
    solver, confidence = recommender.predict(multimodal_problem, return_confidence=True)
    print(f"   Recommended solver: {solver}")
    print(f"   Confidence: {confidence:.2%}")

    # Top-3 for multimodal problem
    top_3 = recommender.predict_top_k(multimodal_problem, k=3)
    print("   Top 3 solvers:")
    for i, (s, c) in enumerate(top_3, 1):
        print(f"     {i}. {s}: {c:.2%}")

    # Feature importance analysis
    print("\n6. Feature Importance Analysis:")
    print("-" * 40)
    importance = train_metrics.get('feature_importance', {})
    if importance:
        print("   Top 10 most important features:")
        for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
            print(f"     {i:2}. {feature:30s}: {score:.4f}")

    # Online learning demo
    print("\n7. Online Learning Capability:")
    print("-" * 40)
    print("   Simulating feedback from solver execution...")

    # Simulate performance feedback
    recommender.update_online(tsp_problem, 'genetic_algorithm', performance=0.85)
    recommender.update_online(convex_problem, 'gradient_descent', performance=0.95)
    recommender.update_online(multimodal_problem, 'particle_swarm', performance=0.78)
    print("   ✓ Model updated with 3 new performance observations")

    # Save the trained model
    print("\n8. Model Persistence:")
    print("-" * 40)
    model_path = Path("models/demo_recommender.pkl")
    model_path.parent.mkdir(exist_ok=True)
    recommender.save_model(model_path)
    print(f"   ✓ Model saved to {model_path}")

    # Load and verify
    new_recommender = SolverRecommender()
    new_recommender.load_model(model_path)
    print(f"   ✓ Model loaded successfully")

    # Verify predictions are consistent
    solver2, _ = new_recommender.predict(tsp_problem, return_confidence=True)
    assert solver == solver2, "Model persistence verification failed"
    print("   ✓ Model persistence verified")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()