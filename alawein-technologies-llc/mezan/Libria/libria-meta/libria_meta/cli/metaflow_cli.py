from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any, Dict

from libria_meta.solvers.metaflow import MetaFlow


def main():
    p = argparse.ArgumentParser(description="MetaFlow CLI")
    p.add_argument("--problem", required=True)
    p.add_argument("--out")
    args = p.parse_args()
    data: Dict[str, Any] = json.loads(Path(args.problem).read_text())
    solver = MetaFlow()
    res = solver.solve(data)
    txt = json.dumps(res, indent=2)
    if args.out:
        Path(args.out).write_text(txt)
    else:
        print(txt)


if __name__ == '__main__':
    main()

