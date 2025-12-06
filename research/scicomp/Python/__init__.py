#!/usr/bin/env python3
"""
SciComp - Core Python Package
==========================================
A cross-platform scientific computing suite for research and education,
featuring quantum physics, machine learning, and engineering applications.
This package provides:
- Quantum Physics: Harmonic oscillators, time evolution, tunneling
- Machine Learning Physics: PINNs, Neural Operators, Uncertainty Quantification
- Quantum Computing: Algorithms, Circuits, Gates
- Visualization: Berkeley-styled plotting and analysis tools
- Utilities: Parallel computing, file I/O, constants
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
"""
# Import from init_berkeley for backward compatibility
from .init_berkeley import (
    __version__,
    __author__,
    __email__,
    __institution__,
    __license__,
    __copyright__,
    FRAMEWORK_INFO,
    BERKELEY_COLORS,
    print_berkeley_banner,
    get_version,
    get_info,
    check_modules,
    MODULE_STATUS
)
# Core module imports with error handling
try:
    from . import quantum_physics
    QUANTUM_PHYSICS_AVAILABLE = True
except ImportError:
    QUANTUM_PHYSICS_AVAILABLE = False
    quantum_physics = None
try:
    from . import ml_physics
    ML_PHYSICS_AVAILABLE = True
except ImportError:
    ML_PHYSICS_AVAILABLE = False
    ml_physics = None
try:
    from . import quantum_computing
    QUANTUM_COMPUTING_AVAILABLE = True
except ImportError:
    QUANTUM_COMPUTING_AVAILABLE = False
    quantum_computing = None
try:
    from . import quantum_materials
    QUANTUM_MATERIALS_AVAILABLE = True
except ImportError:
    QUANTUM_MATERIALS_AVAILABLE = False
    quantum_materials = None
try:
    from . import utils
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    utils = None
try:
    from . import visualization
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    visualization = None
# Update module status
MODULE_STATUS.update({
    "quantum_physics": QUANTUM_PHYSICS_AVAILABLE,
    "ml_physics": ML_PHYSICS_AVAILABLE,
    "quantum_computing": QUANTUM_COMPUTING_AVAILABLE,
    "quantum_materials": QUANTUM_MATERIALS_AVAILABLE,
    "utils": UTILS_AVAILABLE,
    "visualization": VISUALIZATION_AVAILABLE
})
# Define what gets imported with "from Python import *"
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    "__institution__",
    "__license__",
    "__copyright__",
    # Framework info
    "FRAMEWORK_INFO",
    "BERKELEY_COLORS",
    "MODULE_STATUS",
    # Utility functions
    "print_berkeley_banner",
    "get_version",
    "get_info",
    "check_modules",
    # Core modules (if available)
    "quantum_physics",
    "ml_physics",
    "quantum_computing",
    "quantum_materials",
    "utils",
    "visualization"
]
# Optional: Print banner on import (can be disabled with environment variable)
import os
if os.environ.get("BERKELEY_SCICOMP_QUIET", "").lower() not in ("1", "true", "yes"):
    print_berkeley_banner()