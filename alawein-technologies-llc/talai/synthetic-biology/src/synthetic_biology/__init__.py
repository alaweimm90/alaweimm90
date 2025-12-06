"""
Synthetic Biology Research Module for TalAI.

Gene circuit design, metabolic pathway engineering, protein engineering,
CRISPR guide RNA design, and BioBricks integration.
"""

from .models import (
    GeneCircuit,
    MetabolicPathway,
    ProteinDesign,
    CRISPRTarget,
    PlasmidConstruct,
    SynBioHypothesis,
    BioBrick,
    DirectedEvolution,
    SafetyScreen,
)
from .protocol import SyntheticBiologyProtocol
from .analyzers import (
    CircuitDesigner,
    PathwayEngineer,
    ProteinEvolution,
    CRISPRDesigner,
    PlasmidAssembler,
)
from .integrations import BioBricksRegistry, IGSCCompliance

__version__ = "1.0.0"

__all__ = [
    "GeneCircuit",
    "MetabolicPathway",
    "ProteinDesign",
    "CRISPRTarget",
    "PlasmidConstruct",
    "SynBioHypothesis",
    "BioBrick",
    "DirectedEvolution",
    "SafetyScreen",
    "SyntheticBiologyProtocol",
    "CircuitDesigner",
    "PathwayEngineer",
    "ProteinEvolution",
    "CRISPRDesigner",
    "PlasmidAssembler",
    "BioBricksRegistry",
    "IGSCCompliance",
]