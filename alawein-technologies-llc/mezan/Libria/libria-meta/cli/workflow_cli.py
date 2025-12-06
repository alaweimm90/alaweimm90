from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from libria_meta.solvers.workflow import WorkFlow


def main():
    p = argparse.ArgumentParser(description="WorkFlow CLI")
    p.add_argument("--problem", required=True, help="Path to problem.libria.json")
    p.add_argument("--out", help="Path to write solution JSON (default: stdout)")
    p.add_argument("--strategy", default="linear")
    args = p.parse_args()

    problem: Dict[str, Any] = json.loads(Path(args.problem).read_text())
    solver = WorkFlow()
    sol = solver.solve(problem, parameters={"strategy": args.strategy})
    txt = json.dumps(sol, indent=2)
    if args.out:
        Path(args.out).write_text(txt)
    else:
        print(txt)


if __name__ == "__main__":
    main()

