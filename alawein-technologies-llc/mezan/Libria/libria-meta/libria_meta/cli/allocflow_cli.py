from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from libria_meta.solvers.allocflow import AllocFlow
from libria_meta.schemas.validate import validate
from pathlib import Path as _Path
import json as _json


def main():
    p = argparse.ArgumentParser(description="AllocFlow CLI")
    p.add_argument("--problem", required=True, help="Path to problem.libria.json")
    p.add_argument("--out", help="Path to write solution JSON (default: stdout)")
    p.add_argument("--policy", default="greedy_ratio")
    args = p.parse_args()

    problem_path = Path(args.problem)
    problem: Dict[str, Any] = json.loads(problem_path.read_text())
    schema_path = _Path(__file__).resolve().parents[1] / "schemas" / "allocflow_problem.json"
    schema = _json.loads(schema_path.read_text())
    validate(problem, schema)
    solver = AllocFlow()
    sol = solver.solve(problem, parameters={"policy": args.policy})
    sol_schema_path = _Path(__file__).resolve().parents[1] / "schemas" / "allocflow_solution.json"
    sol_schema = _json.loads(sol_schema_path.read_text())
    validate(sol, sol_schema)
    txt = json.dumps(sol, indent=2)
    if args.out:
        Path(args.out).write_text(txt)
    else:
        print(txt)


if __name__ == "__main__":
    main()
