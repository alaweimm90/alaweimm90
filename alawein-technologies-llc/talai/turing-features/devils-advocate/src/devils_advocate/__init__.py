"""
Devil's Advocate Agent

Dedicated adversarial testing of all hypotheses.

Usage:
    from devils_advocate import DevilsAdvocateProtocol

    protocol = DevilsAdvocateProtocol()
    result = await protocol.attack(hypothesis)
"""

__version__ = "0.1.0"

from devils_advocate.protocol import DevilsAdvocateProtocol
from devils_advocate.core.models import (
    AttackResult,
    AttackStrategy,
    Flaw,
    EdgeCase,
)

__all__ = [
    "DevilsAdvocateProtocol",
    "AttackResult",
    "AttackStrategy",
    "Flaw",
    "EdgeCase",
]
