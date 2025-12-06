import sys
import os

import types


def test_libria_router_assignment():
    # Ensure libria-integration is importable
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.path.join(root, 'libria-integration'))
    sys.path.insert(0, os.path.join(root, 'libria-meta'))

    from libria_integration import LibriaRouter  # type: ignore

    # Create mock agents with ids
    Agent = types.SimpleNamespace
    agents = [Agent(config=Agent(agent_id=f"A{i}")) for i in range(3)]
    tasks = [{"task_id": f"T{j}"} for j in range(3)]

    router = LibriaRouter()
    res = router.solve_assignment(agents, tasks)
    assert "assignment" in res
    assert len(res["assignment"]) == 3

