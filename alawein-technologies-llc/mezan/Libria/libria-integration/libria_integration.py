from __future__ import annotations

from typing import Any, Dict, List

# Import solvers directly from libria-meta package
import os
import sys


class LibriaRouter:
    """Adapter between ORCHEX engine and Libria solvers.

    Provides simple methods to solve assignment, plan workflows, and allocate resources.
    """

    def __init__(self):
        # Ensure libria-meta package is importable when running from ORCHEX
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libria-meta'))
        if root not in sys.path:
            sys.path.insert(0, root)

        from libria_meta.solvers.qapflow import QAPFlow  # type: ignore
        from libria_meta.solvers.workflow import WorkFlow  # type: ignore
        from libria_meta.solvers.allocflow import AllocFlow  # type: ignore

        self._qap = QAPFlow()
        self._wf = WorkFlow()
        self._alloc = AllocFlow()

    def solve_assignment(self, agents: List[Any], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a simple fit matrix and call QAPFlow.

        agents: list of objects having attributes or dict keys usable to score tasks
        tasks: list of dicts; minimal interface with 'task_id'
        """
        if not agents or not tasks:
            return {"assignment": None}

        # Minimal heuristic fit: cost = index distance (placeholder)
        n_agents = len(agents)
        n_tasks = len(tasks)
        fit = [[abs(i - j) for j in range(n_tasks)] for i in range(n_agents)]
        problem = {"fit": fit, "constraints": {}}
        sol = self._qap.solve(problem, parameters={"mode": "heuristic"})

        # Map to IDs if present
        mapping = []
        for i, j in enumerate(sol.get("assignment", [])):
            agent_id = getattr(agents[i], 'config', getattr(agents[i], 'id', None))
            if isinstance(agent_id, dict) and 'agent_id' in agent_id:
                agent_id = agent_id['agent_id']
            elif hasattr(agents[i], 'config') and hasattr(agents[i].config, 'agent_id'):
                agent_id = agents[i].config.agent_id  # type: ignore
            task_id = tasks[j].get('task_id', str(j)) if j is not None else None
            mapping.append({"agent_id": agent_id, "task_id": task_id})

        return {"assignment": mapping, "raw": sol}

    def plan_workflow(self, stages: List[str], start: str, end: str, safety: List[str] | None = None) -> Dict[str, Any]:
        problem = {"stages": stages, "start": start, "end": end, "safety": safety or []}
        return self._wf.solve(problem)

    def allocate_resources(self, options: List[Dict[str, Any]], budget: float) -> Dict[str, Any]:
        problem = {"options": options, "budget": budget}
        return self._alloc.solve(problem)

