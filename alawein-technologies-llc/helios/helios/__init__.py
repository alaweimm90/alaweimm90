"""
HELIOS - Hypothesis Exploration & Learning Intelligence Orchestration System

A unified platform for autonomous research discovery, validation, and learning.

Quick Start:
    from helios import HypothesisGenerator, TuringValidator
    from helios.domains import DOMAINS

    generator = HypothesisGenerator()
    hypotheses = generator.generate("Your research topic")

    validator = TuringValidator()
    results = validator.validate(hypotheses)
"""

__version__ = "0.1.0"
__author__ = "HELIOS Team"
__license__ = "MIT"

# Core Discovery Module
try:
    from helios.core.discovery import HypothesisGenerator
except ImportError:
    HypothesisGenerator = None

# Core Validation Module
try:
    from helios.core.validation.turing import TuringValidator
except ImportError:
    TuringValidator = None

# Core Learning Module
try:
    from helios.core.learning import MetaLearner, HallOfFailures
except ImportError:
    MetaLearner = None
    HallOfFailures = None

# Core Orchestration Module
try:
    from helios.core.orchestration import WorkflowOrchestrator
except ImportError:
    WorkflowOrchestrator = None

# Domains
try:
    from helios.domains import DOMAINS
except ImportError:
    DOMAINS = {}

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    # Discovery
    'HypothesisGenerator',
    # Validation
    'TuringValidator',
    # Learning
    'MetaLearner',
    'HallOfFailures',
    # Orchestration
    'WorkflowOrchestrator',
    # Domains
    'DOMAINS',
]
