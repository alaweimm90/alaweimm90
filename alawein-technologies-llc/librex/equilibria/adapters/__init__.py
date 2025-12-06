"""
Domain adapters for various optimization problems

Provides domain-specific adapters for various optimization problem types.
"""

from Librex.adapters.qap import QAPAdapter
from Librex.adapters.tsp import TSPAdapter
from Librex.domains.portfolio import PortfolioAdapter


def get_adapter(problem_type: str):
    """
    Get adapter for a specific problem type

    Args:
        problem_type: Type of optimization problem

    Returns:
        Domain adapter instance
    """
    adapters = {
        "qap": QAPAdapter,
        "tsp": TSPAdapter,
        "portfolio": PortfolioAdapter,
        # Add more adapters as they are implemented
    }

    if problem_type not in adapters:
        raise ValueError(f"Unknown problem type: {problem_type}. Available: {list(adapters.keys())}")

    return adapters[problem_type]()


__all__ = [
    "QAPAdapter",
    "TSPAdapter",
    "PortfolioAdapter",
    "get_adapter",
]
