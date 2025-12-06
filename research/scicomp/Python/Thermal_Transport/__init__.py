"""
Thermal Transport module for SciComp.
Provides comprehensive thermal transport simulation tools including:
- Heat conduction and diffusion
- Phonon transport modeling
- Thermoelectric effects
- Nanoscale heat transfer
"""
from .core.heat_conduction import (
    HeatEquation,
    PhononTransport,
    ThermoelectricEffects,
    HeatExchanger,
    NanoscaleHeatTransfer
)
__all__ = [
    'HeatEquation',
    'PhononTransport',
    'ThermoelectricEffects',
    'HeatExchanger',
    'NanoscaleHeatTransfer'
]
__version__ = '1.0.0'
__author__ = 'UC Berkeley SciComp Team'