"""
MetaHub Visualization Library

Professional visualization utilities and styles for scientific computing.

Consolidated from:
- organizations/alaweimm90-science/MagLogic/ (Berkeley style)
- Future: Additional visualization code from other projects

Usage:
    from metahub.libs.visualization import BerkeleyStyle, berkeley_colors

    style = BerkeleyStyle()
    style.setup()  # Apply Berkeley styling
"""

from .berkeley_style import (
    BerkeleyStyle,
    BERKELEY_COLORS,
    berkeley_style
)

__all__ = [
    "BerkeleyStyle",
    "BERKELEY_COLORS",
    "berkeley_style"
]

__version__ = "1.0.0"
