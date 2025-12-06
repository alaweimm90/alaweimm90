from __future__ import annotations

from typing import Any, Dict, List, Optional

from .solver_base import LibriaSolver, register_solver


class MetaFlow(LibriaSolver):
    name = "MetaFlow"

    def solve(self, problem: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Minimal solver selection stub (UCB-like scoring on synthetic counts).

        Problem:
          - solvers: list of solver names
          - contexts: list of dicts (features), ignored here
          - history: optional {solver: [scores]} dict
        Returns:
          - selections: list of selected solver per context
        """
        solvers: List[str] = problem.get("solvers", [])
        contexts = problem.get("contexts", [])
        history = problem.get("history", {})
        if not solvers:
            return {"solver": self.name, "selections": []}
        # Build simple score = mean(history) + sqrt(1/(1+n))
        import math
        priors = {}
        for s in solvers:
            lst = history.get(s, [])
            n = len(lst)
            mean = sum(lst)/n if n else 0.0
            priors[s] = mean + math.sqrt(1.0/(1+n))
        selections = []
        for _ in contexts:
            best = max(solvers, key=lambda s: priors.get(s, 0.0))
            selections.append(best)
        return {"solver": self.name, "selections": selections, "metadata": {"scores": priors}}


register_solver("metaflow", MetaFlow)

