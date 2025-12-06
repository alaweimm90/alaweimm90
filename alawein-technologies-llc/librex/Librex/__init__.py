"""Compatibility layer that aliases the historical `Librex` namespace to `equilibria`.

Many legacy test suites and downstream integrations still import modules such as
`Librex.acceleration` or `Librex.ai`. The canonical package has been renamed to
`equilibria`, so this module forwards attribute/submodule lookups and exposes the
same package search path to maintain backwards compatibility.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, List

_CANONICAL_PACKAGE = "equilibria"
# Prepopulate the package search path so nested imports (Librex.core, etc.)
# succeed even while this module is still being initialized.
__path__: Iterable[str] = [
    str(Path(__file__).resolve().parent.parent / _CANONICAL_PACKAGE)
]

_CANONICAL_MODULE = importlib.import_module(_CANONICAL_PACKAGE)

# Re-export top-level metadata so `Librex` looks indistinguishable from
# `equilibria` for attribute introspection.
__doc__ = _CANONICAL_MODULE.__doc__
__all__ = getattr(_CANONICAL_MODULE, "__all__", [])

def _alias_submodule(name: str) -> ModuleType:
    """Load the canonical submodule and register it under the Librex namespace."""
    canonical_name = f"{_CANONICAL_PACKAGE}.{name}"
    module = importlib.import_module(canonical_name)
    alias_name = f"{__name__}.{name}"
    sys.modules[alias_name] = module
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover - dynamic proxy
    if hasattr(_CANONICAL_MODULE, name):
        return getattr(_CANONICAL_MODULE, name)

    try:
        module = _alias_submodule(name)
    except ModuleNotFoundError as exc:  # pragma: no cover - error path
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    return module


def __dir__() -> List[str]:  # pragma: no cover - introspection helper
    canonical_dir = set(dir(_CANONICAL_MODULE))
    return sorted(canonical_dir.union(globals().keys()))
