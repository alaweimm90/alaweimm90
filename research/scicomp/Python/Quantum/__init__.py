"""
Quantum module for SciComp.
Provides comprehensive quantum mechanics tools including:
- Quantum states and operators
- Quantum algorithms
- Entanglement measures
- Time evolution
"""
from .core.quantum_states import (
    QuantumState,
    EntanglementMeasures,
    BellStates,
    GHZStates,
    QuantumStateTomography
)
from .core.quantum_operators import (
    PauliOperators,
    QuantumGates,
    HamiltonianOperators,
    TimeEvolution,
    OperatorMeasurements
)
from .core.quantum_algorithms import (
    QuantumFourierTransform,
    PhaseEstimation,
    AmplitudeAmplification,
    QuantumWalk
)
__all__ = [
    # States
    'QuantumState',
    'EntanglementMeasures',
    'BellStates',
    'GHZStates',
    'QuantumStateTomography',
    # Operators
    'PauliOperators',
    'QuantumGates',
    'HamiltonianOperators',
    'TimeEvolution',
    'OperatorMeasurements',
    # Algorithms
    'QuantumFourierTransform',
    'PhaseEstimation',
    'AmplitudeAmplification',
    'QuantumWalk'
]
__version__ = '1.0.0'
__author__ = 'UC Berkeley SciComp Team'