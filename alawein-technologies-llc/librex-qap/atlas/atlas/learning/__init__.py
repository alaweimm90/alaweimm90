"""
ORCHEX Learning Module

Advanced learning algorithms for agent selection and optimization.

Components:
- AdvancedBanditSelector: Multi-algorithm bandit selection
- BanditAlgorithm: Available algorithms
- EnsembleBandit: Meta-learning ensemble

Cycle 25-26: Custom Learning Algorithms
"""

__version__ = "0.1.0"

from ORCHEX.learning.advanced_bandits import (
    AdvancedBanditSelector,
    BanditAlgorithm,
    ArmStats,
    EnsembleBandit
)

__all__ = [
    "AdvancedBanditSelector",
    "BanditAlgorithm",
    "ArmStats",
    "EnsembleBandit",
]
