"""
TalAI Adversarial Red Team System

Comprehensive adversarial testing framework to identify and exploit vulnerabilities
in TalAI's validation pipeline through sophisticated attack strategies.
"""

from .red_team_engine import RedTeamEngine
from .attack_strategies import (
    HypothesisPoisoning,
    BiasExploitation,
    CitationFabrication,
    LogicalFallacyInjection,
    PromptInjection,
    DataPoisoning,
    ModelInversion,
    AdversarialExamples,
    TrojanAttacks,
    BackdoorAttacks
)
from .vulnerability_scanner import VulnerabilityScanner
from .attack_orchestrator import AttackOrchestrator
from .defense_evaluator import DefenseEvaluator

__version__ = "1.0.0"

__all__ = [
    "RedTeamEngine",
    "HypothesisPoisoning",
    "BiasExploitation",
    "CitationFabrication",
    "LogicalFallacyInjection",
    "PromptInjection",
    "DataPoisoning",
    "ModelInversion",
    "AdversarialExamples",
    "TrojanAttacks",
    "BackdoorAttacks",
    "VulnerabilityScanner",
    "AttackOrchestrator",
    "DefenseEvaluator",
]