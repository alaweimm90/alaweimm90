"""
Validation Module: Turing validation suite (falsification, interrogation, etc.)
"""

try:
    from helios.core.validation.turing import TuringValidator, SelfRefutation, Interrogation
except ImportError:
    pass

__all__ = ['TuringValidator', 'SelfRefutation', 'Interrogation']
