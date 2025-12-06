from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import yaml


def run_qaplib(entry: dict) -> dict:
    from libria_meta.cli.qapflow_cli import _bench_load_qaplib_dat as load_dat
    from libria_meta.solvers.qapflow import QAPFlow
    pattern = entry.get('pattern')
    regex = re.compile(pattern) if pattern else None
    qap_dir = os.environ.get('QAPLIB_DATA_DIR')
    if not qap_dir:
        raise SystemExit("Set QAPLIB_DATA_DIR to directory with .dat files")
    files = sorted([p for p in Path(qap_dir).glob('*.dat')])
    if regex:
        files = [f for f in files if regex.search(f.stem)]
    modes = entry.get('modes') or ['hybrid']
    time_limit = int(entry.get('time_limit', 30))
    solver = QAPFlow()
    res = []
    for f in files:
        A, B = load_dat(f)
        problem = {"A": A, "B": B}
        for m in modes:
            r = solver.solve(problem, parameters={"mode": m, "time_limit": time_limit})
            res.append({
                "instance": f.stem,
                "mode": m,
                "objective": r.get("objective"),
                "solve_time": r.get("solve_time") or r.get("metadata", {}).get("solve_time"),
                "bound": r.get("bound"),
            })
    # Summaries
    try:
        from libria_meta.common.bench_utils import summarize_by_mode
        summary = summarize_by_mode(res)
    except Exception:
        summary = {}
    return {"count": len(res), "results": res, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry', required=True, help='Name of registry entry to run')
    ap.add_argument('--out', required=True, help='Output JSON path')
    args = ap.parse_args()

    reg = yaml.safe_load(Path(__file__).with_name('registry.yaml').read_text())
    entry = next((b for b in reg.get('benchmarks', []) if b.get('name') == args.entry), None)
    if not entry:
        raise SystemExit(f"Unknown entry: {args.entry}")

    if entry.get('type') == 'qaplib':
        out = run_qaplib(entry)
    else:
        raise SystemExit(f"Unsupported type: {entry.get('type')}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
