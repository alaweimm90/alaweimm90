"""
Quantum Problem Adapters

Converts classical optimization problems to quantum formulations.
"""

from Librex.quantum.adapters.qubo_converter import QUBOConverter
from Librex.quantum.adapters.ising_encoder import IsingEncoder
from Librex.quantum.adapters.quantum_adapter import QuantumProblemAdapter

__all__ = [
    'QUBOConverter',
    'IsingEncoder',
    'QuantumProblemAdapter',
]