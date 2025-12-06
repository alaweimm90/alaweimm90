"""
Materials Science Research Module for TalAI.

Crystal structure prediction, property analysis, synthesis planning,
and DFT workflow automation for materials discovery.
"""

from .models import (
    CrystalStructure,
    MaterialProperties,
    SynthesisRoute,
    PhaseDiagram,
    DFTCalculation,
    MaterialsHypothesis,
)
from .protocol import MaterialsScienceProtocol
from .analyzers import (
    StructurePredictor,
    PropertyPredictor,
    RetrosynthesisPlanner,
    PhaseDiagramGenerator,
    DFTWorkflowManager,
)
from .integrations import MaterialsProjectAPI, PeriodicTableReasoner

__version__ = "1.0.0"

__all__ = [
    "CrystalStructure",
    "MaterialProperties",
    "SynthesisRoute",
    "PhaseDiagram",
    "DFTCalculation",
    "MaterialsHypothesis",
    "MaterialsScienceProtocol",
    "StructurePredictor",
    "PropertyPredictor",
    "RetrosynthesisPlanner",
    "PhaseDiagramGenerator",
    "DFTWorkflowManager",
    "MaterialsProjectAPI",
    "PeriodicTableReasoner",
]