"""Random search baseline for Neural Architecture Search.

Simple but effective baseline that randomly samples architectures.
"""

from typing import Dict, List, Optional, Any
import numpy as np

from ..architecture import NASCell, MacroArchitecture
from ..nas_problem import NASProblem
from ..nas_adapter import NASAdapter
from .evolutionary_nas import EvolutionaryNAS


def random_search_nas(problem: NASProblem, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Random search for NAS.

    Randomly samples architectures and evaluates them.

    Args:
        problem: NAS problem to solve
        config: Configuration dictionary with:
            - n_samples: Number of architectures to sample
            - early_stopping: Whether to stop if budget exhausted

    Returns:
        Results dictionary
    """
    n_samples = config.get('n_samples', 100)
    early_stopping = config.get('early_stopping', True)

    # Use evolutionary NAS helper for random generation
    evo_helper = EvolutionaryNAS(problem=problem, population_size=1)
    adapter = NASAdapter()

    evaluations = []
    best_architecture = None
    best_metrics = None
    best_objective = float('-inf')

    print(f"Random search: sampling {n_samples} architectures")

    for i in range(n_samples):
        # Generate random architecture
        architecture = problem.create_architecture()

        if problem.search_space.value == 'cell':
            evo_helper._random_cell_architecture(architecture)
        else:
            evo_helper._random_macro_architecture(architecture)

        # Evaluate
        metrics = problem.evaluate_architecture(architecture, return_all_metrics=True)
        objective = metrics.get('objective', metrics.get('accuracy', 0))

        evaluations.append({
            'architecture': architecture,
            'metrics': metrics,
            'objective': objective
        })

        # Update best
        if objective > best_objective:
            best_objective = objective
            best_architecture = architecture
            best_metrics = metrics
            print(f"  Sample {i+1}/{n_samples}: New best = {best_objective:.4f}")
        else:
            if (i + 1) % 10 == 0:
                print(f"  Sample {i+1}/{n_samples}: objective = {objective:.4f}")

        # Early stopping
        if early_stopping and problem.evaluation_count >= problem.max_evaluations:
            print(f"Stopping early: evaluation budget exhausted at sample {i+1}")
            break

    # Compute statistics
    objectives = [e['objective'] for e in evaluations]
    stats = {
        'mean': np.mean(objectives),
        'std': np.std(objectives),
        'min': np.min(objectives),
        'max': np.max(objectives),
        'median': np.median(objectives)
    }

    return {
        'best_architecture': best_architecture,
        'best_metrics': best_metrics,
        'best_objective': best_objective,
        'all_evaluations': evaluations,
        'statistics': stats,
        'n_evaluated': len(evaluations)
    }