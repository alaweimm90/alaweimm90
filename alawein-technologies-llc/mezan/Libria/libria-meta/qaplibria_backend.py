from __future__ import annotations

"""
QAPFlow external backend adapter (stub).

If you have a real QAP solver package, implement:

def solve(problem: dict, parameters: dict | None = None) -> dict:
    ... -> return dict in QAPFlow solution schema

This stub implements a minimal greedy solution so the backend mode runs without errors.
Replace with calls into your own solver.
"""

from typing import Any, Dict, List, Optional
import math
import os
import sys
import numpy as np
try:
    # Optional validation if libria_meta is importable in this context
    from libria_meta.schemas.validate import validate as _validate
    import json as _json
    import pathlib as _pathlib
    _SCHEMAS_DIR = _pathlib.Path(__file__).resolve().parent / 'libria_meta' / 'schemas'
    _SOL_SCHEMA = None
except Exception:
    _validate = None
    _SCHEMAS_DIR = None
    _SOL_SCHEMA = None


def _as_qap_matrices(problem: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert MEZAN problem dict into QAP (A,B) matrices when possible.

    Accepts the following keys:
      - Preferred: 'A' and 'B' (already in QAPLIB-like form)
      - Fallback: 'fit' (n x n cost), optional 'interaction' (n x n). If only 'fit' is
        provided, use A=fit and B=I, which reduces to linear assignment cost in QAP form.
    """
    if "A" in problem and "B" in problem:
        A = np.array(problem["A"], dtype=float)
        B = np.array(problem["B"], dtype=float)
        if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1] or A.shape != B.shape:
            raise ValueError("Librex.QAP_backend: 'A' and 'B' must be square and same shape")
        return A, B

    fit = problem.get("fit")
    if fit is None:
        raise ValueError("Librex.QAP_backend: expected 'A'/'B' or 'fit'")
    A = np.array(fit, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Librex.QAP_backend: 'fit' must be square (n x n)")
    if "interaction" in problem:
        B = np.array(problem["interaction"], dtype=float)
        if B.shape != A.shape:
            raise ValueError("Librex.QAP_backend: 'interaction' must match 'fit' shape")
    else:
        B = np.eye(n, dtype=float)
    return A, B


_FUNC_CACHE: Dict[str, Any] = {}


def _get_auto_policy() -> Dict[str, int]:
    """Return size thresholds for auto mode selection.

    Environment override: QAPFLOW_AUTO_POLICY="S,M" where S and M are integers.
    Policy:
      - n <= S -> 'hybrid'
      - S < n <= M -> 'nesterov'
      - n > M -> 'enhanced'
    Defaults: S=20, M=50.
    """
    env = os.environ.get("QAPFLOW_AUTO_POLICY")
    small, medium = 20, 50
    if env:
        try:
            parts = [int(x.strip()) for x in env.split(',') if x.strip()]
            if len(parts) >= 1:
                small = parts[0]
            if len(parts) >= 2:
                medium = parts[1]
        except Exception:
            pass
    if medium < small:
        medium = small
    return {"small": small, "medium": medium}


def _import_mode_func(mode: str) -> Optional[Any]:
    if mode in _FUNC_CACHE:
        return _FUNC_CACHE[mode]
    import importlib
    try:
        if mode == 'hybrid':
            mod = importlib.import_module('solvers.hybrid_combined_solver')
            fn = getattr(mod, 'hybrid_combined_solver', None)
        elif mode == 'fft':
            mod = importlib.import_module('solvers.fft_accelerated_solver')
            fn = getattr(mod, 'fft_accelerated_solver', None)
        elif mode == 'enhanced':
            mod = importlib.import_module('solvers.enhanced_solver')
            fn = getattr(mod, 'enhanced_solver', None)
        elif mode == 'nesterov':
            mod = importlib.import_module('solvers.nesterov_accelerated_solver')
            fn = getattr(mod, 'nesterov_accelerated_solver', None)
        elif mode == 'instance_adaptive':
            mod = importlib.import_module('solvers.instance_adaptive_optimizer')
            fn = getattr(mod, 'adaptive_solver_with_params', None)
        elif mode == 'aggressive':
            mod = importlib.import_module('solvers.aggressive_solver')
            fn = getattr(mod, 'aggressive_solver', None)
        else:
            fn = None
    except Exception:
        fn = None
    _FUNC_CACHE[mode] = fn
    return fn


def _try_modular_repo_solver(A: np.ndarray, B: np.ndarray, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attempt to call a modular-repo solver by mode if available.

    Supported modes (if present in modular repo):
      - hybrid (default)
      - fft
      - enhanced
      - nesterov
      - instance_adaptive
      - aggressive
    """
    # Safe mode: refuse external backend if enabled
    if str(parameters.get("safe_mode", os.environ.get("QAPFLOW_SAFE_MODE", "0"))).lower() in ("1","true","yes"):
        return None
    # Allow explicit override via env
    mod_path = os.environ.get("QAP_MODULAR_REPO_PATH")
    # Default guess
    default_path = "/mnt/c/Users/mesha/Documents/Projects/Active/Librex.QAP/projects/QAP_modular_repo-main/src"
    for candidate in [mod_path, default_path]:
        if not candidate:
            continue
        if os.path.isdir(candidate):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            try:
                # Auto mode: size-based selection if requested (env override supported)
                mode = str(parameters.get("mode", "hybrid")).lower()
                auto_policy = None
                if mode == 'auto':
                    n = int(A.shape[0])
                    pol = _get_auto_policy()
                    auto_policy = pol.copy()
                    if n <= pol["small"]:
                        mode = 'hybrid'
                    elif n <= pol["medium"]:
                        mode = 'nesterov'
                    else:
                        mode = 'enhanced'
                max_time = float(parameters.get("time_limit", 60))

                fn = _import_mode_func(mode)
                if mode == "hybrid" and fn is not None:
                    # Optional homotopy hint if supported
                    use_homotopy = bool(parameters.get("use_homotopy", False))
                    try:
                        P, obj, solve_time, iters = fn(A, B, max_time=max_time, verbose=False, use_homotopy=use_homotopy)
                        hom_used = use_homotopy
                    except TypeError:
                        P, obj, solve_time, iters = fn(A, B, max_time=max_time, verbose=False)
                        hom_used = False
                    assignment = [int(x) for x in np.argmax(P, axis=1)]
                    res = {
                        "solver": "QAPFlow-Backend:HybridCombined",
                        "assignment": assignment,
                        "objective": float(obj),
                        "solve_time": float(solve_time),
                        "metadata": {"backend": "external", "iterations": int(iters), "solve_time": float(solve_time), "mode": mode, "use_homotopy": hom_used},
                    }
                    if auto_policy:
                        res["metadata"]["auto_policy"] = auto_policy
                    return res

                if mode == "fft" and fn is not None:
                    out = fn(A, B, max_time=max_time, verbose=False)
                    # Some solvers return (P,obj), others (P,obj,time,...)
                    if isinstance(out, tuple) and len(out) >= 2:
                        P, obj = out[0], out[1]
                        solve_time = out[2] if len(out) >= 3 else 0.0
                    assignment = [int(x) for x in np.argmax(P, axis=1)]
                    res = {
                        "solver": "QAPFlow-Backend:FFT",
                        "assignment": assignment,
                        "objective": float(obj),
                        "solve_time": float(solve_time),
                        "metadata": {"backend": "external", "solve_time": float(solve_time), "mode": mode},
                    }
                    if auto_policy:
                        res["metadata"]["auto_policy"] = auto_policy
                    return res

                if mode == "enhanced" and fn is not None:
                    # Optional homotopy hint if supported; capture any bound info from extra returns
                    use_homotopy = bool(parameters.get("use_homotopy", False))
                    try:
                        out = fn(A, B, max_time=max_time, verbose=False, use_homotopy=use_homotopy)
                        hom_used = use_homotopy
                    except TypeError:
                        out = fn(A, B, max_time=max_time, verbose=False)
                        hom_used = False
                    # Unpack flexible outputs
                    if not isinstance(out, tuple) or len(out) < 3:
                        raise RuntimeError("enhanced solver did not return expected tuple")
                    P, obj, solve_time, *_rest = out
                    # Attempt to extract a lower bound from rest payloads if provided
                    bound = None
                    bound_source = None
                    for item in _rest:
                        if isinstance(item, dict):
                            if 'bound' in item:
                                bound = float(item['bound'])
                                bound_source = item.get('bound_source', 'solver_metadata')
                                break
                            if 'lower_bound' in item:
                                bound = float(item['lower_bound'])
                                bound_source = item.get('bound_source', 'lower_bound')
                                break
                    assignment = [int(x) for x in np.argmax(P, axis=1)]
                    result = {
                        "solver": "QAPFlow-Backend:Enhanced",
                        "assignment": assignment,
                        "objective": float(obj),
                        "solve_time": float(solve_time),
                        "metadata": {"backend": "external", "solve_time": float(solve_time), "mode": mode, "use_homotopy": hom_used},
                    }
                    if auto_policy:
                        result["metadata"]["auto_policy"] = auto_policy
                    if bound is not None:
                        result["bound"] = float(bound)
                        result["metadata"]["bound_source"] = bound_source
                    return result

                if mode == "nesterov" and fn is not None:
                    P, obj, solve_time = fn(A, B, max_time=max_time, verbose=False)
                    assignment = [int(x) for x in np.argmax(P, axis=1)]
                    res = {
                        "solver": "QAPFlow-Backend:Nesterov",
                        "assignment": assignment,
                        "objective": float(obj),
                        "solve_time": float(solve_time),
                        "metadata": {"backend": "external", "solve_time": float(solve_time), "mode": mode},
                    }
                    if auto_policy:
                        res["metadata"]["auto_policy"] = auto_policy
                    return res

                if mode == "instance_adaptive" and fn is not None:
                    # Pull selected params from parameters dict if present
                    params = {
                        "step_size": float(parameters.get("step_size", 0.01)),
                        "momentum": float(parameters.get("momentum", 0.9)),
                        "entropy_max": float(parameters.get("entropy_max", 0.005)),
                        "entropy_schedule": float(parameters.get("entropy_schedule", 0.5)),
                        "perturb_freq": int(parameters.get("perturb_freq", 250)),
                        "perturb_strength": float(parameters.get("perturb_strength", 0.05)),
                    }
                    P, obj, solve_time, *_ = fn(A, B, params, max_time=max_time)
                    assignment = [int(x) for x in np.argmax(P, axis=1)]
                    res = {
                        "solver": "QAPFlow-Backend:InstanceAdaptive",
                        "assignment": assignment,
                        "objective": float(obj),
                        "solve_time": float(solve_time),
                        "metadata": {"backend": "external", "solve_time": float(solve_time), "mode": mode, "params": params},
                    }
                    if auto_policy:
                        res["metadata"]["auto_policy"] = auto_policy
                    return res

                if mode == "aggressive" and fn is not None:
                    P, obj, *rest = fn(A, B, max_time=max_time, verbose=False)
                    solve_time = rest[0] if rest else 0.0
                    assignment = [int(x) for x in np.argmax(P, axis=1)]
                    res = {
                        "solver": "QAPFlow-Backend:Aggressive",
                        "assignment": assignment,
                        "objective": float(obj),
                        "solve_time": float(solve_time),
                        "metadata": {"backend": "external", "solve_time": float(solve_time), "mode": mode},
                    }
                    if auto_policy:
                        res["metadata"]["auto_policy"] = auto_policy
                    return res

                # Unknown mode: fall through to try default hybrid
                from solvers.hybrid_combined_solver import hybrid_combined_solver  # type: ignore
                try:
                    P, obj, solve_time, iters = hybrid_combined_solver(A, B, max_time=max_time, verbose=False, use_homotopy=bool(parameters.get("use_homotopy", False)))
                    hom_used = bool(parameters.get("use_homotopy", False))
                except TypeError:
                    P, obj, solve_time, iters = hybrid_combined_solver(A, B, max_time=max_time, verbose=False)
                    hom_used = False
                assignment = list(np.argmax(P, axis=1).astype(int))
                res = {
                    "solver": "QAPFlow-Backend:HybridCombined",
                    "assignment": assignment,
                    "objective": float(obj),
                    "solve_time": float(solve_time),
                    "metadata": {"backend": "external", "iterations": int(iters), "solve_time": float(solve_time), "mode": "hybrid", "use_homotopy": hom_used},
                }
                if auto_policy:
                    res["metadata"]["auto_policy"] = auto_policy
                return res
            except Exception:
                continue
    return None


def solve(problem: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    parameters = parameters or {}
    A, B = _as_qap_matrices(problem)
    # Optional robustness perturbation to test sensitivity (applies before solve)
    robust_eps = float(parameters.get("robust_eps", 0.0) or 0.0)
    if robust_eps > 0:
        scaleA = max(1.0, float(np.max(np.abs(A))))
        scaleB = max(1.0, float(np.max(np.abs(B))))
        rng = np.random.default_rng(parameters.get("seed", None))
        A = A + rng.uniform(-robust_eps*scaleA, robust_eps*scaleA, size=A.shape)
        B = B + rng.uniform(-robust_eps*scaleB, robust_eps*scaleB, size=B.shape)

    # Try modular repo solver first
    res = _try_modular_repo_solver(A, B, parameters)
    if res is not None:
        return res

    # Fallback: greedy Hungarian-like on 'fit' diagonal costs
    fit = A.tolist()
    n_agents = len(fit)
    n_tasks = len(fit[0]) if n_agents else 0
    used = set()
    assignment: List[int] = [-1] * n_agents
    for i in range(n_agents):
        best_j = None
        best_cost = math.inf
        for j in range(n_tasks):
            if j in used:
                continue
            cost = float(fit[i][j])
            if cost < best_cost:
                best_cost = cost
                best_j = j
        if best_j is None:
            best_j = 0
        assignment[i] = best_j
        used.add(best_j)

    result = {
        "solver": "QAPFlow-Backend:Greedy",
        "assignment": assignment,
        "objective": float(sum(fit[i][assignment[i]] for i in range(n_agents))),
        "metadata": {"backend": "fallback"},
    }
    # Optional solution validation
    if _validate is not None:
        try:
            global _SOL_SCHEMA
            if _SOL_SCHEMA is None and _SCHEMAS_DIR is not None:
                _SOL_SCHEMA = _json.loads((_SCHEMAS_DIR / 'qapflow_solution.json').read_text())
            if _SOL_SCHEMA is not None:
                _validate(result, _SOL_SCHEMA)
        except Exception:
            # Do not hard fail; leave validation to CLI/HTTP layers
            pass
    return result
