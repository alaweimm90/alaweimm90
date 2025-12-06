"""
Quantum Optics module for SciComp.
Provides comprehensive quantum optics tools including:
- Cavity QED simulations
- Quantum light states
- Photon statistics
- Jaynes-Cummings model
"""
from .core.cavity_qed import (
    JaynesCummings,
    DissipativeJaynesCummings,
    CavityModes,
    PulseShaping
)
from .core.quantum_light import (
    FockStates,
    CoherentStates,
    SqueezedStates,
    PhotonStatistics,
    WignerFunction
)
__all__ = [
    # Cavity QED
    'JaynesCummings',
    'DissipativeJaynesCummings',
    'CavityModes',
    'PulseShaping',
    # Quantum Light
    'FockStates',
    'CoherentStates',
    'SqueezedStates',
    'PhotonStatistics',
    'WignerFunction'
]
__version__ = '1.0.0'
__author__ = 'UC Berkeley SciComp Team'