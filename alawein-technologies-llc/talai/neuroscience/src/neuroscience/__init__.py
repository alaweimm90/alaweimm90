"""
Neuroscience Research Module for TalAI.

fMRI/EEG analysis, neural circuit hypothesis generation, connectome analysis,
cognitive modeling, and brain-computer interface research.
"""

from .models import (
    NeuralData,
    BrainRegion,
    ConnectomeGraph,
    CognitiveModel,
    BCIExperiment,
    NeuroscienceHypothesis,
    FMRIData,
    EEGData,
    NeuralCircuit,
)
from .protocol import NeuroscienceProtocol
from .analyzers import (
    FMRIAnalyzer,
    EEGAnalyzer,
    ConnectomeAnalyzer,
    CognitiveModelTester,
    BCIDesigner,
    CircuitHypothesisGenerator,
)
from .integrations import AllenBrainAPI, NWBInterface

__version__ = "1.0.0"

__all__ = [
    "NeuralData",
    "BrainRegion",
    "ConnectomeGraph",
    "CognitiveModel",
    "BCIExperiment",
    "NeuroscienceHypothesis",
    "FMRIData",
    "EEGData",
    "NeuralCircuit",
    "NeuroscienceProtocol",
    "FMRIAnalyzer",
    "EEGAnalyzer",
    "ConnectomeAnalyzer",
    "CognitiveModelTester",
    "BCIDesigner",
    "CircuitHypothesisGenerator",
    "AllenBrainAPI",
    "NWBInterface",
]