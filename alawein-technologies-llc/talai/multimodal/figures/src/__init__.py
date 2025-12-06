"""
TalAI Scientific Figure Understanding Module

Comprehensive system for extracting and understanding data from scientific figures,
charts, plots, and diagrams with multi-modal AI integration.
"""

from .figure_analyzer import FigureAnalyzer
from .data_extractor import DataExtractor
from .chart_detector import ChartTypeDetector
from .ocr_engine import FigureOCR
from .caption_processor import CaptionProcessor
from .accessibility_generator import AccessibilityDescriptor

__all__ = [
    'FigureAnalyzer',
    'DataExtractor',
    'ChartTypeDetector',
    'FigureOCR',
    'CaptionProcessor',
    'AccessibilityDescriptor'
]

__version__ = '1.0.0'