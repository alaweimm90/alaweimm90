from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .solver_base import LibriaSolver, register_solver


class GraphFlow(LibriaSolver):
    name = "GraphFlow"

    def solve(self, problem: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Minimal topology optimization stub.

        Problem:
          - nodes: list of node ids
          - edges: list of {u, v, weight}
          - budget: optional (max edges)
        Returns:
          - pruned_edges: simple selection of lowest weight edges up to budget or spanning set
        """
        nodes: List[str] = problem.get("nodes", [])
        edges_in = problem.get("edges", [])
        budget = int(problem.get("budget", max(0, len(nodes) - 1)))
        # Normalize edges to (u,v,weight)
        edges: List[Tuple[str, str, float]] = []
        for e in edges_in:
            if isinstance(e, dict):
                u, v = e.get('u'), e.get('v')
                w = float(e.get('weight', 1.0))
            else:
                u, v, w = e[0], e[1], float(e[2] if len(e) > 2 else 1.0)
            edges.append((u, v, w))
        # Sort by weight and take up to budget
        pruned = []
        seen = set()
        for u, v, w in sorted(edges, key=lambda x: x[2]):
            pruned.append({"u": u, "v": v, "weight": w})
            if len(pruned) >= budget:
                break
        return {
            "solver": self.name,
            "pruned_edges": pruned,
            "metadata": {"selected": len(pruned), "budget": budget}
        }


register_solver("graphflow", GraphFlow)

