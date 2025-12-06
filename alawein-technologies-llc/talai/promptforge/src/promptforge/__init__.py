"""Package initialization."""

from .main import main
from .models import PromptPattern, ExtractionResult
from .extractor import PatternExtractor
from .library import PromptLibrary

__version__ = "0.1.0"

__all__ = [
    'main',
    'PromptPattern',
    'ExtractionResult',
    'PatternExtractor',
    'PromptLibrary',
]
