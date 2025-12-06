"""
Turingo Agent Package

All 14 agents for the autonomous problem-solving carnival
"""

from .base import TuringoAgent, ExecutiveAgent, SpecialistAgent, ConsultantAgent
from .executive.ringmaster import Ringmaster
from .executive.blueprint_boss import BlueprintBoss
from .executive.deal_maker import DealMaker
from .executive.ethics_enforcer import EthicsEnforcer

from .specialists.puzzle_prodigy import PuzzleProdigy
from .specialists.quantum_quokka import QuantumQuokka
from .specialists.ml_magician import MLMagician
from .specialists.analogy_alchemist import AnalogyAlchemist
from .specialists.proof_pirate import ProofPirate
from .specialists.verification_vigilante import VerificationVigilante
from .specialists.benchmark_bandit import BenchmarkBandit
from .specialists.code_cowboy import CodeCowboy
from .specialists.novelty_ninja import NoveltyNinja
from .specialists.skeptic_sorcerer import SkepticSorcerer

__all__ = [
    'TuringoAgent', 'ExecutiveAgent', 'SpecialistAgent', 'ConsultantAgent',
    'Ringmaster', 'BlueprintBoss', 'DealMaker', 'EthicsEnforcer',
    'PuzzleProdigy', 'QuantumQuokka', 'MLMagician', 'AnalogyAlchemist',
    'ProofPirate', 'VerificationVigilante', 'BenchmarkBandit', 'CodeCowboy',
    'NoveltyNinja', 'SkepticSorcerer'
]
