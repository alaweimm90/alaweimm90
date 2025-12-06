from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from libria_meta.solvers.qapflow import QAPFlow


def main():
    p = argparse.ArgumentParser(description="QAPFlow CLI")
    p.add_argument("--problem", required=True, help="Path to problem.libria.json")
    p.add_argument("--out", help="Path to write solution JSON (default: stdout)")
    p.add_argument("--mode", default="heuristic")
    p.add_argument("--time-limit", type=int, default=0)
    args = p.parse_args()

    problem: Dict[str, Any] = json.loads(Path(args.problem).read_text())
    solver = QAPFlow()
    sol = solver.solve(problem, parameters={"mode": args.mode, "time_limit": args.time_limit})
    txt = json.dumps(sol, indent=2)
    if args.out:
        Path(args.out).write_text(txt)
    else:
        print(txt)


if __name__ == "__main__":
    main()

