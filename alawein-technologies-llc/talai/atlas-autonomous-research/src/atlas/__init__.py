"""
ORCHEX - Autonomous Theorist & Laboratory Autonomous System

Full autonomous research: Topic → Hypotheses → Validation → Experiments → Paper

⚠️ NOT the Nobel Turing Challenge - computational research prototype only.
"""

__version__ = "0.1.0"

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used only for type checkers
    from .protocol import ATLASProtocol, ResearchProject
    from .hypothesis_generator import HypothesisGenerator, HypothesisCandidate

__all__ = ["ATLASProtocol", "ResearchProject", "HypothesisGenerator", "HypothesisCandidate"]


def __getattr__(name):
    """Lazy-load heavy modules to avoid importing optional dependencies at init time."""
    if name == "ATLASProtocol":
        from .protocol import ATLASProtocol as _ATLASProtocol

        return _ATLASProtocol
    if name == "ResearchProject":
        from .protocol import ResearchProject as _ResearchProject

        return _ResearchProject
    if name == "HypothesisGenerator":
        from .hypothesis_generator import HypothesisGenerator as _HypothesisGenerator

        return _HypothesisGenerator
    if name == "HypothesisCandidate":
        from .hypothesis_generator import HypothesisCandidate as _HypothesisCandidate

        return _HypothesisCandidate
    raise AttributeError(name)
