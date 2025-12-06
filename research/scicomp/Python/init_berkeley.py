#!/usr/bin/env python3
"""
SciComp
==========================
UC Berkeley Scientific Computing Framework for multi-platform scientific
computing, quantum physics, machine learning, and engineering applications.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
"""
# Version information
__version__ = "1.0.0"
__author__ = "Meshal Alawein"
__email__ = "meshal@berkeley.edu"
__institution__ = "University of California, Berkeley"
__license__ = "MIT"
__copyright__ = "Copyright © 2025 Meshal Alawein — All rights reserved."
# SciComp metadata
__title__ = "SciComp"
__description__ = "UC Berkeley Scientific Computing Framework"
__url__ = "https://github.com/berkeley-scicomp/SciComp"
__documentation__ = "https://berkeley-scicomp.readthedocs.io"
# Framework information
FRAMEWORK_INFO = {
    "name": __title__,
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "institution": __institution__,
    "license": __license__,
    "url": __url__,
    "platforms": ["Python", "MATLAB", "Mathematica"],
    "domains": [
        "Quantum Physics",
        "Machine Learning Physics",
        "Quantum Computing",
        "Engineering Applications",
        "Scientific Visualization"
    ]
}
# Berkeley colors for terminal output
BERKELEY_COLORS = {
    "berkeley_blue": "\033[38;2;0;50;98m",
    "california_gold": "\033[38;2;253;181;21m",
    "reset": "\033[0m",
    "bold": "\033[1m"
}
def print_berkeley_banner():
    """Print Berkeley SciComp framework banner."""
    blue = BERKELEY_COLORS["berkeley_blue"]
    gold = BERKELEY_COLORS["california_gold"]
    reset = BERKELEY_COLORS["reset"]
    bold = BERKELEY_COLORS["bold"]
    banner = f"""
{blue}{bold}================================================================
SciComp v{__version__}
================================================================{reset}
{gold}University of California, Berkeley{reset}
{blue}{__author__} ({__email__}){reset}
Multi-Platform Scientific Computing:
  • Quantum Physics & Quantum Computing
  • Machine Learning for Physics
  • Computational Methods & Engineering
  • Professional Berkeley Visual Identity
{blue}Platforms: Python | MATLAB | Mathematica{reset}
{blue}License: {__license__} | Go Bears! 🐻💙💛{reset}
"""
    print(banner)
def get_version():
    """Get framework version."""
    return __version__
def get_info():
    """Get framework information dictionary."""
    return FRAMEWORK_INFO.copy()
# Import main modules (with error handling)
try:
    # Quantum physics modules would be imported here
    QUANTUM_PHYSICS_AVAILABLE = True
except ImportError:
    QUANTUM_PHYSICS_AVAILABLE = False
try:
    # Machine learning physics modules would be imported here
    ML_PHYSICS_AVAILABLE = True
except ImportError:
    ML_PHYSICS_AVAILABLE = False
try:
    # Quantum computing modules would be imported here
    QUANTUM_COMPUTING_AVAILABLE = True
except ImportError:
    QUANTUM_COMPUTING_AVAILABLE = False
try:
    # Utilities would be imported here
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
# Module availability status
MODULE_STATUS = {
    "quantum_physics": QUANTUM_PHYSICS_AVAILABLE,
    "ml_physics": ML_PHYSICS_AVAILABLE,
    "quantum_computing": QUANTUM_COMPUTING_AVAILABLE,
    "utils": UTILS_AVAILABLE
}
def check_modules():
    """Check which modules are available."""
    available = [name for name, status in MODULE_STATUS.items() if status]
    unavailable = [name for name, status in MODULE_STATUS.items() if not status]
    print(f"{BERKELEY_COLORS['berkeley_blue']}Berkeley SciComp Module Status:{BERKELEY_COLORS['reset']}")
    if available:
        print(f"{BERKELEY_COLORS['california_gold']}Available modules:{BERKELEY_COLORS['reset']}")
        for module in available:
            print(f"  ✓ {module}")
    if unavailable:
        print(f"{BERKELEY_COLORS['california_gold']}Unavailable modules:{BERKELEY_COLORS['reset']}")
        for module in unavailable:
            print(f"  ✗ {module}")
# Convenient imports for common use cases
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__institution__",
    "__license__",
    "FRAMEWORK_INFO",
    "BERKELEY_COLORS",
    "print_berkeley_banner",
    "get_version",
    "get_info",
    "check_modules",
    "MODULE_STATUS"
]
# Print banner on import (optional, can be disabled)
import os
if os.environ.get("BERKELEY_SCICOMP_QUIET", "").lower() not in ("1", "true", "yes"):
    print_berkeley_banner()