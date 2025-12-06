"""Client stub for Librex's public optimization APIs.

This module provides small, stable wrapper functions around Librex's
public `optimize` and `optimize_qap` APIs so that other workspaces or
meta-systems (e.g. tal-ai) can depend on a single, documented entry
surface without knowing Librex's internal structure.

It assumes the `Librex` package is installed and importable in the
current Python environment.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


try:
    # Universal optimize entry point
    from Librex import optimize as _optimize

    # QAP convenience API (thin facade over optimize + QAPAdapter)
    from Librex.Librex.QAP import optimize_qap as _optimize_qap
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "The 'Librex' Python package is not installed. "
        "Install it in this environment (e.g. 'pip install -e .') "
        "before using Librex_client."
    ) from exc


def optimize_problem(
    problem: Any,
    adapter: Any,
    *,
    method: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Optimize an arbitrary problem using Librex's universal interface.

    This is a light wrapper around `Librex.optimize` that simply
    ensures `config` defaults to an empty dict if omitted.
    """

    return _optimize(problem, adapter, method=method, config=config or {})


def optimize_qap_problem(
    flow_matrix: Any,
    distance_matrix: Any,
    *,
    method: str = "simulated_annealing",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Optimize a Quadratic Assignment Problem via Librex's QAP API.

    This is a light wrapper around `Librex.Librex.QAP.optimize_qap`.
    """

    return _optimize_qap(flow_matrix, distance_matrix, method=method, config=config or {})


# Optional aliases for callers that want names close to the original API
optimize = optimize_problem
optimize_qap = optimize_qap_problem
