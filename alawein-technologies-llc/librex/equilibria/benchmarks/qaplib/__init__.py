"""
QAPLIB Benchmark Suite for Librex

This module provides access to the complete QAPLIB benchmark collection,
containing 138 standard QAP instances for benchmarking optimization algorithms.

Main functions:
    load_qaplib_instance(name): Load a specific instance
    list_qaplib_instances(): List available instances
    get_qaplib_metadata(name): Get instance metadata
    benchmark_method(): Run benchmarks on multiple instances
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .loader import (
    QAPLIBLoader,
    load_qaplib_instance,
    list_qaplib_instances,
    get_qaplib_metadata,
    get_loader
)
from .registry import (
    QAPLIB_REGISTRY,
    QAPLIBInstance,
    QAPLIBRegistry,
    get_instance_by_size,
    get_instance_by_class,
    get_small_instances,
    get_all_instance_names,
    get_problem_classes,
    get_instance_metadata
)
from .benchmark_runner import QAPLIBBenchmark, QAPLIBBenchmarkRunner, run_qaplib_benchmark
from . import embedded_data as _embedded_data

# Legacy compatibility: certain callers still expect EMBEDDED_QAPLIB_DATA to be
# exported from the embedded_data module even though the canonical constant is
# EMBEDDED_INSTANCES. Patch the module at import time if necessary.
if not hasattr(_embedded_data, "EMBEDDED_QAPLIB_DATA"):
    _embedded_data.EMBEDDED_QAPLIB_DATA = _embedded_data.EMBEDDED_INSTANCES

if hasattr(_embedded_data, "__all__"):
    if "EMBEDDED_QAPLIB_DATA" not in _embedded_data.__all__:
        _embedded_data.__all__ = list(_embedded_data.__all__) + ["EMBEDDED_QAPLIB_DATA"]
else:
    _embedded_data.__all__ = ["EMBEDDED_INSTANCES", "EMBEDDED_QAPLIB_DATA"]

EMBEDDED_QAPLIB_DATA = _embedded_data.EMBEDDED_QAPLIB_DATA

# Version
__version__ = "1.0.0"

# Main exports
__all__ = [
    # Core functions
    "load_qaplib_instance",
    "list_qaplib_instances",
    "get_qaplib_metadata",
    "get_instance",
    "list_instances",

    # Embedded data
    "EMBEDDED_QAPLIB_DATA",

    # Registry functions
    "get_instance_by_size",
    "get_instance_by_class",
    "get_small_instances",
    "get_all_instance_names",
    "get_problem_classes",

    # Benchmark functions
    "QAPLIBBenchmark",
    "QAPLIBBenchmarkRunner",
    "run_qaplib_benchmark",

    # Classes
    "QAPLIBLoader",
    "QAPLIBInstance",
    "QAPLIBRegistry",

    # Constants
    "QAPLIB_REGISTRY",
]


def quick_info():
    """Print quick info about QAPLIB suite"""
    print(f"QAPLIB Benchmark Suite v{__version__}")
    print(f"Total instances: {len(QAPLIB_REGISTRY)}")

    # Count by class
    class_counts = {}
    for inst in QAPLIB_REGISTRY.values():
        class_counts[inst.problem_class] = class_counts.get(inst.problem_class, 0) + 1

    print("\nInstances by problem class:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:25} {count:3} instances")

    # Size distribution
    sizes = [inst.size for inst in QAPLIB_REGISTRY.values()]
    print(f"\nSize range: {min(sizes)} to {max(sizes)}")
    print(f"Small instances (n≤20): {len(get_small_instances())}")
    print(f"Medium instances (20<n≤50): {len(get_instance_by_size(21, 50))}")
    print(f"Large instances (n>50): {len(get_instance_by_size(51, 300))}")


def validate_installation():
    """Validate that QAPLIB suite is properly installed"""
    errors = []

    # Check registry
    if len(QAPLIB_REGISTRY) != 138:
        errors.append(f"Registry has {len(QAPLIB_REGISTRY)} instances, expected 138")

    # Try loading a small embedded instance
    try:
        data = load_qaplib_instance("nug12")
        if "flow_matrix" not in data or "distance_matrix" not in data:
            errors.append("Failed to load embedded instance nug12")
    except Exception as e:
        errors.append(f"Error loading nug12: {e}")

    # Check loader
    try:
        loader = get_loader()
        if not isinstance(loader, QAPLIBLoader):
            errors.append("Invalid loader instance")
    except Exception as e:
        errors.append(f"Error creating loader: {e}")

    if errors:
        print("QAPLIB validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("QAPLIB suite validated successfully!")
        return True


def _with_legacy_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
    if 'flow' in payload and 'distance' in payload:
        return payload

    enriched = dict(payload)
    if 'flow_matrix' in enriched:
        enriched.setdefault('flow', enriched['flow_matrix'])
    if 'distance_matrix' in enriched:
        enriched.setdefault('distance', enriched['distance_matrix'])
    return enriched


def get_instance(name: str) -> Dict[str, Any]:
    """Legacy alias that mirrors the pre-restructure API."""
    return _with_legacy_keys(load_qaplib_instance(name))


def list_instances() -> List[str]:
    """Return the canonical list of registered QAPLIB instances."""
    return list_qaplib_instances()
