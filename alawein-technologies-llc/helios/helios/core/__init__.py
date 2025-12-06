"""
HELIOS Core: Discovery, Validation, Learning, and Orchestration

Core modules:
- discovery: Hypothesis generation from literature
- validation: Turing validation suite (falsification, interrogation, etc.)
- learning: Meta-learning with personality agents
- orchestration: ORCHEX research workflow engine
"""

try:
    from helios.core.discovery import HypothesisGenerator
    from helios.core.validation import TuringValidator
    from helios.core.learning import MetaLearner, HallOfFailures, PersonalityAgent
    from helios.core.orchestration import ATLASEngine, ResearchWorkflow
except ImportError:
    pass

__all__ = [
    'HypothesisGenerator',
    'TuringValidator',
    'MetaLearner',
    'HallOfFailures',
    'PersonalityAgent',
    'ATLASEngine',
    'ResearchWorkflow',
]
