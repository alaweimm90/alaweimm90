from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from libria_meta.solvers.workflow import WorkFlow
from libria_meta.schemas.validate import validate
from pathlib import Path as _Path
import json as _json


def main():
    p = argparse.ArgumentParser(description="WorkFlow CLI")
    p.add_argument("--problem", required=True, help="Path to problem.libria.json")
    p.add_argument("--out", help="Path to write solution JSON (default: stdout)")
    p.add_argument("--strategy", default="linear")
    args = p.parse_args()

    problem_path = Path(args.problem)
    problem: Dict[str, Any] = json.loads(problem_path.read_text())
    schema_path = _Path(__file__).resolve().parents[1] / "schemas" / "workflow_problem.json"
    schema = _json.loads(schema_path.read_text())
    validate(problem, schema)
    solver = WorkFlow()
    sol = solver.solve(problem, parameters={"strategy": args.strategy})
    sol_schema_path = _Path(__file__).resolve().parents[1] / "schemas" / "workflow_solution.json"
    sol_schema = _json.loads(sol_schema_path.read_text())
    validate(sol, sol_schema)
    txt = json.dumps(sol, indent=2)
    if args.out:
        Path(args.out).write_text(txt)
    else:
        print(txt)


if __name__ == "__main__":
    main()
