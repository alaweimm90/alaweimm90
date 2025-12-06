"""
Spintronics module for SciComp.
Provides comprehensive spintronics simulation tools including:
- Spin dynamics and magnetization evolution
- Spin transport phenomena
- Magnetoresistive devices
- Spin-orbit coupling effects
"""
from .core.spin_dynamics import (
    LandauLifshitzGilbert,
    SpinWaves,
    MagneticDomains,
    SpinTorque,
    MagnetocrystallineAnisotropy
)
from .core.spin_transport import (
    SpinDiffusion,
    MagnetoresistiveDevices,
    SpinHallEffect,
    RashbaEffect,
    MagnonTransport,
    SpinValve
)
__all__ = [
    # Spin Dynamics
    'LandauLifshitzGilbert',
    'SpinWaves',
    'MagneticDomains',
    'SpinTorque',
    'MagnetocrystallineAnisotropy',
    # Spin Transport
    'SpinDiffusion',
    'MagnetoresistiveDevices',
    'SpinHallEffect',
    'RashbaEffect',
    'MagnonTransport',
    'SpinValve'
]
__version__ = '1.0.0'
__author__ = 'UC Berkeley SciComp Team'