import json
import os
import sys
from pathlib import Path

import pytest

from libria_meta.schemas.validate import validate


ROOT = Path(__file__).resolve().parents[1]


def load(schema_name: str):
    schema_path = ROOT / "libria_meta/schemas" / schema_name
    return json.loads(schema_path.read_text())


def test_qapflow_schema_valid():
    schema = load("qapflow_problem.json")
    problem = json.loads((ROOT / "examples/problems/qapflow_minimal.json").read_text())
    validate(problem, schema)


def test_workflow_schema_valid():
    schema = load("workflow_problem.json")
    problem = json.loads((ROOT / "examples/problems/workflow_minimal.json").read_text())
    validate(problem, schema)


def test_allocflow_schema_valid():
    schema = load("allocflow_problem.json")
    problem = json.loads((ROOT / "examples/problems/allocflow_minimal.json").read_text())
    validate(problem, schema)


def test_orchestrator_demo_runs():
    sys.path.insert(0, str(ROOT))
    # Ensure integration path is importable
    sys.path.insert(0, str(ROOT.parent / "libria-integration"))
    from examples.mezan_orchestrator_demo import run_demo

    res = run_demo()
    assert "assignment" in res and "workflow" in res and "allocation" in res


def test_orchestrator_http_server():
    # Start server on ephemeral port
    sys.path.insert(0, str(ROOT))
    from services.mezan_orchestrator import run_server
    server, th = run_server(port=0)
    host, port = server.server_address

    import urllib.request
    import time
    time.sleep(0.1)

    def post(path: str, payload: dict):
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url=f"http://{host}:{port}{path}", data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode('utf-8'))

    qap = json.loads((ROOT / "examples/problems/qapflow_minimal.json").read_text())
    out = post('/qapflow/solve', qap)
    assert "assignment" in out

    wf = json.loads((ROOT / "examples/problems/workflow_minimal.json").read_text())
    out2 = post('/workflow/solve', wf)
    assert "route" in out2

    alloc = json.loads((ROOT / "examples/problems/allocflow_minimal.json").read_text())
    out3 = post('/allocflow/solve', alloc)
    assert "allocation" in out3

    # Test A/B only payload
    ab_payload = {"A": [[1,0],[0,1]], "B": [[1,2],[2,1]]}
    out_ab = post('/qapflow/solve_ab', ab_payload)
    assert "assignment" in out_ab

    # health endpoint
    import urllib.request
    with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5) as resp:
        health = json.loads(resp.read().decode('utf-8'))
        assert health.get('status') == 'ok'
        assert 'env_path' in health
    server.shutdown()
