"""
TalAI Multi-Provider Cross-Validation System

Comprehensive cross-validation framework that runs hypotheses through multiple
LLM providers to detect bias, ensure agreement, and optimize cost-benefit.
"""

from .cross_validation_engine import CrossValidationEngine
from .provider_manager import ProviderManager
from .agreement_analyzer import AgreementAnalyzer
from .bias_detector import BiasDetector
from .cost_optimizer import CostOptimizer

__version__ = "1.0.0"

__all__ = [
    "CrossValidationEngine",
    "ProviderManager",
    "AgreementAnalyzer",
    "BiasDetector",
    "CostOptimizer",
]