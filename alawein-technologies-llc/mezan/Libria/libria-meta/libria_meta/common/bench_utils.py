from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_qaplib_dat(path: Path) -> Tuple[List[List[float]], List[List[float]]]:
    data = path.read_text().split()
    if not data:
        raise ValueError(f"Empty file: {path}")
    n = int(data[0])
    vals = list(map(float, data[1:1 + 2 * n * n]))
    A = [vals[i * n:(i + 1) * n] for i in range(n)]
    B = [vals[n * n + i * n:n * n + (i + 1) * n] for i in range(n)]
    return A, B


def filter_instances(files: List[Path], patterns: Iterable[str] | None) -> List[Path]:
    if not patterns:
        return files
    pats = [s.strip().lower() for s in patterns if s and s.strip()]
    if not pats:
        return files
    return [f for f in files if any(p in f.stem.lower() for p in pats)]


def summarize_by_mode(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_mode: Dict[str, Dict[str, Any]] = {}
    for r in results:
        mode = r.get('mode') or 'unknown'
        rec = by_mode.setdefault(mode, {'count': 0, 'sum_obj': 0.0, 'sum_time': 0.0, 'has_time': 0})
        rec['count'] += 1
        if r.get('objective') is not None:
            rec['sum_obj'] += float(r['objective'])
        st = r.get('metadata', {}).get('solve_time')
        if st is not None:
            rec['sum_time'] += float(st)
            rec['has_time'] += 1
    return {
        m: {
            'count': d['count'],
            'avg_objective': (d['sum_obj'] / d['count']) if d['count'] else None,
            'avg_solve_time': (d['sum_time'] / d['has_time']) if d['has_time'] else None,
        }
        for m, d in by_mode.items()
    }


def summarize_by_bucket(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    def bucket(n: int) -> str:
        return 'small' if n <= 20 else ('medium' if n <= 50 else 'large')

    bucket_mode: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in results:
        n = int(r.get('n') or 0)
        b = bucket(n)
        m = r.get('mode') or 'unknown'
        key = (b, m)
        rec = bucket_mode.setdefault(key, {'count': 0, 'sum_obj': 0.0, 'sum_time': 0.0, 'has_time': 0})
        rec['count'] += 1
        if r.get('objective') is not None:
            rec['sum_obj'] += float(r['objective'])
        st = r.get('metadata', {}).get('solve_time')
        if st is not None:
            rec['sum_time'] += float(st)
            rec['has_time'] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for (b, m), d in bucket_mode.items():
        out.setdefault(b, {})[m] = {
            'count': d['count'],
            'avg_objective': (d['sum_obj'] / d['count']) if d['count'] else None,
            'avg_solve_time': (d['sum_time'] / d['has_time']) if d['has_time'] else None,
        }
    return out


def write_results_and_summaries(results: List[Dict[str, Any]], summary: Dict[str, Any], outp: Path) -> None:
    if outp.suffix.lower() == '.csv':
        with outp.open('w', newline='') as cf:
            w = csv.writer(cf)
            w.writerow(["instance", "n", "mode", "objective", "solve_time", "seed", "robust_eps"]) 
            for r in results:
                w.writerow([
                    r.get("instance"),
                    r.get('n'),
                    r.get("mode"),
                    r.get("objective"),
                    r.get("metadata", {}).get("solve_time"),
                    r.get('seed'),
                    r.get('robust_eps')
                ])
        sum_path = outp.parent / (outp.stem + "_summary.csv")
        with sum_path.open('w', newline='') as sf:
            w = csv.writer(sf)
            w.writerow(["mode", "count", "avg_objective", "avg_solve_time"]) 
            for m, d in (summary.get('summary') or {}).items():
                w.writerow([m, d.get('count'), d.get('avg_objective'), d.get('avg_solve_time')])
        if summary.get('buckets'):
            sum_b_path = outp.parent / (outp.stem + "_buckets.csv")
            with sum_b_path.open('w', newline='') as sb:
                w = csv.writer(sb)
                w.writerow(["bucket", "mode", "count", "avg_objective", "avg_solve_time"]) 
                for b, modes_map in summary['buckets'].items():
                    for m, d in modes_map.items():
                        w.writerow([b, m, d.get('count'), d.get('avg_objective'), d.get('avg_solve_time')])
        md_path = outp.parent / (outp.stem + "_summary.md")
        with md_path.open('w') as mf:
            mf.write("| mode | count | avg_objective | avg_solve_time |\n")
            mf.write("|---|---:|---:|---:|\n")
            for m, d in (summary.get('summary') or {}).items():
                mf.write(f"| {m} | {d.get('count')} | {d.get('avg_objective')} | {d.get('avg_solve_time')} |\n")
            if summary.get('buckets'):
                mf.write("\n\n### Per-size buckets\n\n")
                for b, modes_map in summary['buckets'].items():
                    mf.write(f"#### {b}\n\n")
                    mf.write("| mode | count | avg_objective | avg_solve_time |\n")
                    mf.write("|---|---:|---:|---:|\n")
                    for m, d in modes_map.items():
                        mf.write(f"| {m} | {d.get('count')} | {d.get('avg_objective')} | {d.get('avg_solve_time')} |\n")
    elif outp.suffix.lower() == '.jsonl':
        with outp.open('w') as jl:
            for r in results:
                jl.write(json.dumps(r) + "\n")
        (outp.parent / (outp.stem + "_summary.json")).write_text(json.dumps(summary, indent=2))
    else:
        outp.write_text(json.dumps(summary, indent=2))


def print_summary_table(summary: Dict[str, Any]) -> None:
    modes = summary.get('summary') or {}
    if modes:
        print("mode\tcount\tavg_objective\tavg_solve_time")
        for m, d in modes.items():
            print(f"{m}\t{d.get('count')}\t{d.get('avg_objective')}\t{d.get('avg_solve_time')}")

