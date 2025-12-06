"""
Quantum Computing Integration for Librex

This module provides quantum optimization methods and converters for
combinatorial optimization problems. Quantum features are completely optional
and require installing with: pip install Librex[quantum]

Features:
- QUBO and Ising model converters for classical problems
- Quantum annealing support (D-Wave style)
- QAOA (Quantum Approximate Optimization Algorithm)
- VQE (Variational Quantum Eigensolver)
- Problem size validation for NISQ devices
- Graceful fallback when quantum libraries not available

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
import warnings
from typing import Optional

logger = logging.getLogger(__name__)

# Check for quantum library availability
QISKIT_AVAILABLE = False
PENNYLANE_AVAILABLE = False

try:
    import qiskit
    from qiskit import __version__ as qiskit_version
    QISKIT_AVAILABLE = True
    logger.info(f"Qiskit {qiskit_version} is available for quantum optimization")
except ImportError:
    logger.debug("Qiskit not installed. Quantum features will be limited.")

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
    logger.info(f"PennyLane {qml.__version__} is available for quantum optimization")
except ImportError:
    logger.debug("PennyLane not installed. Quantum features will be limited.")


def check_quantum_availability(library: Optional[str] = None) -> bool:
    """
    Check if quantum libraries are available.

    Args:
        library: Specific library to check ('qiskit' or 'pennylane').
                If None, checks if any quantum library is available.

    Returns:
        bool: True if the specified library (or any library) is available
    """
    if library is None:
        return QISKIT_AVAILABLE or PENNYLANE_AVAILABLE
    elif library.lower() == 'qiskit':
        return QISKIT_AVAILABLE
    elif library.lower() == 'pennylane':
        return PENNYLANE_AVAILABLE
    else:
        raise ValueError(f"Unknown quantum library: {library}")


def require_quantum_library(library: Optional[str] = None):
    """
    Decorator to check if quantum libraries are available before executing a function.

    Args:
        library: Specific library required ('qiskit' or 'pennylane').
                If None, requires at least one quantum library.

    Raises:
        ImportError: If the required quantum library is not available
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not check_quantum_availability(library):
                lib_msg = f"{library}" if library else "any quantum library"
                raise ImportError(
                    f"Quantum feature requires {lib_msg}. "
                    f"Please install with: pip install Librex[quantum]"
                )
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# Import quantum components if available
__all__ = [
    'QISKIT_AVAILABLE',
    'PENNYLANE_AVAILABLE',
    'check_quantum_availability',
    'require_quantum_library',
]

# Conditionally import quantum modules
if QISKIT_AVAILABLE or PENNYLANE_AVAILABLE:
    try:
        from Librex.quantum.adapters import (
            QUBOConverter,
            IsingEncoder,
            QuantumProblemAdapter,
        )
        from Librex.quantum.validators import (
            QuantumProblemValidator,
            QubitEstimator,
        )
        from Librex.quantum.utils import (
            QuantumStateDecoder,
            HamiltonianBuilder,
            quantum_to_classical_result,
        )

        __all__.extend([
            'QUBOConverter',
            'IsingEncoder',
            'QuantumProblemAdapter',
            'QuantumProblemValidator',
            'QubitEstimator',
            'QuantumStateDecoder',
            'HamiltonianBuilder',
            'quantum_to_classical_result',
        ])

        # Import quantum methods if available
        from Librex.quantum.methods import (
            quantum_annealing_optimize,
            qaoa_optimize,
            vqe_optimize,
        )

        __all__.extend([
            'quantum_annealing_optimize',
            'qaoa_optimize',
            'vqe_optimize',
        ])

    except ImportError as e:
        logger.debug(f"Some quantum components could not be imported: {e}")
else:
    # Provide informative message when quantum features are accessed without libraries
    def _quantum_not_available(*args, **kwargs):
        raise ImportError(
            "Quantum optimization features require quantum libraries. "
            "Install with: pip install Librex[quantum]"
        )

    # Create placeholder functions that raise helpful errors
    QUBOConverter = _quantum_not_available
    IsingEncoder = _quantum_not_available
    QuantumProblemAdapter = _quantum_not_available
    QuantumProblemValidator = _quantum_not_available
    QubitEstimator = _quantum_not_available
    QuantumStateDecoder = _quantum_not_available
    HamiltonianBuilder = _quantum_not_available
    quantum_to_classical_result = _quantum_not_available
    quantum_annealing_optimize = _quantum_not_available
    qaoa_optimize = _quantum_not_available
    vqe_optimize = _quantum_not_available


# Version info
__version__ = "1.0.0"
__author__ = "Meshal Alawein"