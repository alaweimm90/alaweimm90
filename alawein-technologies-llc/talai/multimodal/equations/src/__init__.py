"""
TalAI Mathematical Equation Processing Module

Advanced system for parsing, understanding, and working with
mathematical equations from research papers.
"""

from .equation_parser import EquationParser
from .latex_processor import LaTeXProcessor
from .handwriting_ocr import HandwritingOCR
from .symbolic_engine import SymbolicEngine
from .dimensional_analyzer import DimensionalAnalyzer
from .equation_similarity import EquationSimilarity
from .code_generator import EquationToCode

__all__ = [
    'EquationParser',
    'LaTeXProcessor',
    'HandwritingOCR',
    'SymbolicEngine',
    'DimensionalAnalyzer',
    'EquationSimilarity',
    'EquationToCode'
]

__version__ = '1.0.0'