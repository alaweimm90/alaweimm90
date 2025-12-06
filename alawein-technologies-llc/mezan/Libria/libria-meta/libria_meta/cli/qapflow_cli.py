from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from libria_meta.solvers.qapflow import QAPFlow
from libria_meta.schemas.validate import validate
from libria_meta.common.schema_utils import load_schema, validate_with_schema
from libria_meta.common.bench_utils import (
    load_qaplib_dat as _bench_load_qaplib_dat,
    filter_instances as _bench_filter_instances,
    summarize_by_mode as _bench_summarize_by_mode,
    summarize_by_bucket as _bench_summarize_by_bucket,
    write_results_and_summaries as _bench_write_outputs,
    print_summary_table as _bench_print_summary,
)
from pathlib import Path as _Path
import json as _json


def main():
    p = argparse.ArgumentParser(description="QAPFlow CLI")
    p.add_argument("--problem", required=True, help="Path to problem.libria.json")
    p.add_argument("--out", help="Path to write solution JSON (default: stdout)")
    p.add_argument("--mode", default="heuristic", help="heuristic|hybrid|fft|enhanced|nesterov|instance_adaptive|aggressive")
    p.add_argument("--time-limit", type=int, default=0)
    p.add_argument("--use-homotopy", action='store_true', help="Pass hint to backend if supported")
    p.add_argument("--backend", default=None, help="external to use modular repo backend")
    p.add_argument("--bench", default=None, help="Run benchmark suite: qaplib")
    p.add_argument("--modes", default=None, help="Comma-separated modes to sweep in bench")
    p.add_argument("--instances", default=None, help="Comma-separated substrings to filter QAPLIB instances (bench only)")
    p.add_argument("--dry-run", action='store_true', help="Print planned instances/modes and exit (bench only)")
    p.add_argument("--html", default=None, help="Write an HTML report alongside CSV/JSON (bench only)")
    p.add_argument("--robust-eps", type=float, default=0.0, help="Optional robustness noise epsilon for bench (0=off)")
    args = p.parse_args()

    problem_path = Path(args.problem)
    problem: Dict[str, Any] = json.loads(problem_path.read_text())
    # Validate against schema
    # Precheck: require either 'fit' or ('A' and 'B')
    if not (('fit' in problem) or ('A' in problem and 'B' in problem)):
        raise SystemExit("QAP problem must include 'fit' or both 'A' and 'B'")
    validate_with_schema(problem, 'qapflow_problem.json')
    # Benchmark mode
    if args.bench:
        if args.bench.lower() == 'qaplib':
            return run_qaplib_bench(args)
        else:
            raise SystemExit(f"Unknown bench suite: {args.bench}")

    solver = QAPFlow()
    params = {"mode": args.mode, "time_limit": args.time_limit}
    if args.backend:
        params["backend"] = args.backend
    if args.robust_eps and args.robust_eps > 0:
        params["robust_eps"] = args.robust_eps
    if args.use_homotopy:
        params["use_homotopy"] = True
    sol = solver.solve(problem, parameters=params)
    # Validate solution
    sol_schema_path = _Path(__file__).resolve().parents[1] / "schemas" / "qapflow_solution.json"
    sol_schema = _json.loads(sol_schema_path.read_text())
    validate(sol, sol_schema)
    txt = json.dumps(sol, indent=2)
    if args.out:
        Path(args.out).write_text(txt)
    else:
        print(txt)


def _load_qaplib_dat(path: Path):
    return _bench_load_qaplib_dat(path)


def run_qaplib_bench(args) -> None:
    import os, time, json as _json
    qap_dir = os.environ.get('QAPLIB_DATA_DIR')
    if not qap_dir:
        raise SystemExit("Set QAPLIB_DATA_DIR to directory with .dat files")
    qap_dir = Path(qap_dir)
    files = sorted([p for p in qap_dir.glob('*.dat')])
    if not files:
        raise SystemExit(f"No .dat files in {qap_dir}")

    results = []
    solver = QAPFlow()
    start_all = time.time()
    modes = []
    if args.modes:
        modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    if not modes:
        modes = [args.mode]
    files = _bench_filter_instances(files, args.instances.split(',') if args.instances else None)
    if args.dry_run:
        print("Planned instances:")
        for f in files:
            print(" -", f.stem)
        print("Modes:")
        for m in modes:
            print(" -", m)
        raise SystemExit(0)

    for f in files:
        try:
            A, B = _load_qaplib_dat(f)
            problem = {"A": A, "B": B, "constraints": {}}
            n = len(A)
            for mode in modes:
                params = {"mode": mode, "time_limit": args.time_limit or 60}
                if args.backend:
                    params["backend"] = args.backend
                if args.robust_eps and args.robust_eps > 0:
                    params["robust_eps"] = args.robust_eps
                sol = solver.solve(problem, parameters=params)
                rec = {
                    "instance": f.stem,
                    "n": n,
                    "mode": mode,
                    "objective": sol.get("objective"),
                    "metadata": sol.get("metadata", {}),
                    "robust_eps": args.robust_eps or 0.0,
                }
                results.append(rec)
        except Exception as e:
            results.append({"instance": f.stem, "error": str(e)})

    # Build summaries
    summary_modes = _bench_summarize_by_mode(results)
    summary_buckets = _bench_summarize_by_bucket(results)
    summary = {
        "count": len(results),
        "elapsed": time.time() - start_all,
        "results": results,
        "summary": summary_modes,
        "buckets": summary_buckets,
    }
    txt = _json.dumps(summary, indent=2)
    if args.out:
        outp = Path(args.out)
        _bench_write_outputs(results, summary, outp)
        if args.html:
            from libria_meta.common.bench_html import generate_html
            html = generate_html(summary, results)
            Path(args.html).write_text(html)
    else:
        # Print compact table summary then JSON
        _bench_print_summary(summary)
        print(txt)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
