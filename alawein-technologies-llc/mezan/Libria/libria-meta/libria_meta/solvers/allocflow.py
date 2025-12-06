from __future__ import annotations

from typing import Any, Dict, List, Optional

from .solver_base import LibriaSolver, register_solver


class AllocFlow(LibriaSolver):
    name = "AllocFlow"

    def solve(self, problem: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Minimal constrained allocation stub using greedy value/cost ratio.

        Problem dict:
          - options: list of {id, cost, expected_return}
          - budget: number
          - fairness: optional constraints (unused in stub)
        """
        parameters = parameters or {}
        options: List[Dict[str, Any]] = list(problem.get("options", []))
        budget = float(problem.get("budget", 0))

        options_sorted = sorted(options, key=lambda x: (float(x.get("expected_return", 0)) / max(1e-9, float(x.get("cost", 1)))), reverse=True)
        chosen: List[Dict[str, Any]] = []
        spent = 0.0
        total_return = 0.0
        for opt in options_sorted:
            c = float(opt.get("cost", 0))
            r = float(opt.get("expected_return", 0))
            if spent + c <= budget:
                chosen.append(opt)
                spent += c
                total_return += r

        return {
            "solver": self.name,
            "allocation": [o.get("id") for o in chosen],
            "spent": spent,
            "expected_return": total_return,
            "metadata": {"policy": parameters.get("policy", "greedy_ratio")},
        }


register_solver("allocflow", AllocFlow)

