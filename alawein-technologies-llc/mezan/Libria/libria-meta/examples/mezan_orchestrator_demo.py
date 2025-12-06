from __future__ import annotations

"""
MEZAN Orchestrator Demo

Runs assignment (QAPFlow), workflow routing (WorkFlow), and allocation (AllocFlow)
using the LibriaRouter adapter.
"""

import os
import sys
from typing import Any, Dict


def run_demo() -> Dict[str, Any]:
    # Ensure imports work regardless of CWD
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(root, 'libria-integration'))
    sys.path.insert(0, root)

    from libria_integration import LibriaRouter  # type: ignore
    import types

    # Mock agents and a single task
    Agent = types.SimpleNamespace
    agents = [Agent(config=Agent(agent_id=f"A{i}")) for i in range(4)]
    tasks = [{"task_id": f"T{j}"} for j in range(4)]

    router = LibriaRouter()
    assign = router.solve_assignment(agents, tasks)

    # Simple workflow definition
    stages = ["designer", "critic", "refactorer", "validator"]
    wf = router.plan_workflow(stages=stages, start="designer", end="validator", safety=["validator"])

    # Simple allocation options
    options = [
        {"id": "compute_small", "cost": 2, "expected_return": 3},
        {"id": "compute_med", "cost": 4, "expected_return": 5},
        {"id": "compute_large", "cost": 6, "expected_return": 6}
    ]
    alloc = router.allocate_resources(options=options, budget=6)

    return {"assignment": assign, "workflow": wf, "allocation": alloc}


if __name__ == "__main__":
    import json
    print(json.dumps(run_demo(), indent=2))

