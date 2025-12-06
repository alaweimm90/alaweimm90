import json
import sys
import os
from pathlib import Path

import subprocess


ROOT = Path(__file__).resolve().parents[1]


def run_cli(cmd):
    """Run a CLI command and parse JSON output if printed to stdout."""
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip()
    return json.loads(out) if out.startswith('{') else out


def test_qapflow_cli_minimal():
    prob = ROOT / "examples/problems/qapflow_minimal.json"
    cmd = [sys.executable, "-m", "libria_meta.cli.qapflow_cli", "--problem", str(prob)]
    res = run_cli(cmd)
    assert res["solver"] == "QAPFlow"
    assert len(res["assignment"]) == 3


def test_workflow_cli_minimal():
    prob = ROOT / "examples/problems/workflow_minimal.json"
    cmd = [sys.executable, "-m", "libria_meta.cli.workflow_cli", "--problem", str(prob)]
    res = run_cli(cmd)
    assert res["solver"] == "WorkFlow"
    assert res["route"][0] == "designer"
    assert res["route"][-1] == "validator"


def test_allocflow_cli_minimal():
    prob = ROOT / "examples/problems/allocflow_minimal.json"
    cmd = [sys.executable, "-m", "libria_meta.cli.allocflow_cli", "--problem", str(prob)]
    res = run_cli(cmd)
    assert res["solver"] == "AllocFlow"
    assert res["spent"] <= 7

