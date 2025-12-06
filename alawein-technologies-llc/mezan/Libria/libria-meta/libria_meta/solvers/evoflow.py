from __future__ import annotations

from typing import Any, Dict, List, Optional

from .solver_base import LibriaSolver, register_solver


class EvoFlow(LibriaSolver):
    name = "EvoFlow"

    def solve(self, problem: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Minimal evolutionary portfolio stub.

        Problem:
          - population: list of {candidate: any, score: float}
          - mutate_rate: optional
        Returns:
          - new_population: reweighted/mutated list
          - best: best candidate id/index
        """
        pop: List[Dict[str, Any]] = problem.get("population", [])
        if not pop:
            return {"solver": self.name, "new_population": [], "best": None}
        # Simple: keep top half and duplicate with slight score noise
        pop2 = sorted(pop, key=lambda x: float(x.get('score', 0.0)), reverse=True)
        top = pop2[: max(1, len(pop2)//2)]
        new_pop = top + [{"candidate": c.get("candidate"), "score": float(c.get("score", 0.0))*0.99} for c in top]
        best = top[0].get("candidate")
        return {"solver": self.name, "new_population": new_pop, "best": best}


register_solver("evoflow", EvoFlow)

