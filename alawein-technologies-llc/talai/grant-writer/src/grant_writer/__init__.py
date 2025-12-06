"""GrantWriter - AI-Powered Grant Proposal Assistant

Generate competitive grant proposals with AI assistance.
Includes significance statements, budgets, timelines, and NIH/NSF formatting.
"""

__version__ = "1.0.0"
__author__ = "TalAI"

from .main import GrantWriter, GrantProposal, BudgetItem, TimelinePhase, Personnel

__all__ = ["GrantWriter", "GrantProposal", "BudgetItem", "TimelinePhase", "Personnel"]
