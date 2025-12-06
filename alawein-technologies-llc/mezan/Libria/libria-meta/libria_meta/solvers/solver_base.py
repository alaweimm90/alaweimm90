from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type


class LibriaSolver(ABC):
    """Base class for all Libria optimization solvers.

    Implementations should be pure-Python friendly and expose deterministic
    behavior when a random seed is provided in parameters.
    """

    name: str = "LibriaSolver"

    @abstractmethod
    def solve(self, problem: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Solve an optimization problem and return a solution dict."""
        raise NotImplementedError

    def benchmark(self, dataset: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
        """Optional: run a benchmark suite; return summary metrics.

        Default is a no-op stub for lightweight adoption.
        """
        return {"dataset": dataset, "status": "not_implemented"}


_REGISTRY: Dict[str, Type[LibriaSolver]] = {}


def register_solver(key: str, cls: Type[LibriaSolver]) -> None:
    key_l = key.strip().lower()
    _REGISTRY[key_l] = cls


def get_solver(key: str) -> Type[LibriaSolver]:
    key_l = key.strip().lower()
    if key_l not in _REGISTRY:
        raise KeyError(f"Solver not registered: {key}")
    return _REGISTRY[key_l]


def list_solvers() -> Dict[str, str]:
    return {k: v.__name__ for k, v in _REGISTRY.items()}

