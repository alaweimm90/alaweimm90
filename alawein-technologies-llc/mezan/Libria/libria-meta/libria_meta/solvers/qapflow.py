from __future__ import annotations

from typing import Any, Dict, List, Optional

import math

from .solver_base import LibriaSolver, register_solver


class QAPFlow(LibriaSolver):
    name = "QAPFlow"

    def solve(self, problem: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Minimal heuristic QAP solver stub.

        Expects problem dict with keys:
          - fit: matrix [n_agents x n_tasks]
          - interaction: optional matrix [n_agents x n_agents]
          - constraints: optional dict
        Returns assignment list of length n_agents with task indices.
        """
        parameters = parameters or {}
        # Optional external backend integration
        backend = parameters.get("backend") or __import__("os").environ.get("QAPFLOW_BACKEND")
        if backend and backend.lower() == "external":
            try:
                ext = __import__("Librex.QAP_backend")
                return ext.solve(problem, parameters)  # type: ignore[attr-defined]
            except Exception:
                # Fall back to built-in heuristic
                pass
        fit = problem.get("fit")
        if not fit:
            raise ValueError("QAPFlow: 'fit' matrix required")

        n_agents = len(fit)
        n_tasks = len(fit[0]) if n_agents else 0
        used = set()
        assignment: List[int] = [-1] * n_agents

        # Greedy by best fit per agent avoiding already taken tasks
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
                # pad if more agents than tasks
                best_j = 0
            assignment[i] = best_j
            used.add(best_j)

        return {
            "solver": self.name,
            "assignment": assignment,
            "objective": float(sum(fit[i][assignment[i]] for i in range(n_agents))),
            "metadata": {"mode": parameters.get("mode", "heuristic")},
        }


register_solver("qapflow", QAPFlow)
