from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .solver_base import LibriaSolver, register_solver


class WorkFlow(LibriaSolver):
    name = "WorkFlow"

    def solve(self, problem: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Minimal workflow routing stub.

        Problem dict:
          - stages: list of stage names
          - edges: list of (u, v, time_cost) triples or dicts
          - start: stage name
          - end: stage name
          - safety: optional list of mandatory stages that must appear

        Strategy: If a default route exists in problem['route'], use it.
        Otherwise, return a linear pass of stages from start to end.
        """
        parameters = parameters or {}
        stages: List[str] = problem.get("stages", [])
        start = problem.get("start") or (stages[0] if stages else None)
        end = problem.get("end") or (stages[-1] if stages else None)
        if not stages or start is None or end is None:
            raise ValueError("WorkFlow: stages, start, and end are required")

        if "route" in problem and problem["route"]:
            route = list(problem["route"])
        else:
            # naive route: from start to end, include all safety stages if present
            safety = [s for s in problem.get("safety", []) if s in stages]
            route = [start] + [s for s in stages if s not in {start, end} and s in safety] + [end]

        return {
            "solver": self.name,
            "route": route,
            "metadata": {"strategy": parameters.get("strategy", "linear")},
        }


register_solver("workflow", WorkFlow)

