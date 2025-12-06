from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description="Promote OptiBench result to baseline thresholds")
    ap.add_argument('--result', required=True, help='Path to replay result JSON')
    ap.add_argument('--baseline', required=True, help='Path to baseline JSON to update')
    ap.add_argument('--out', required=False, help='Output path (defaults to in-place update)')
    ap.add_argument('--golden', required=False, help='Comma list of instances or auto:N to select N instances from results')
    ap.add_argument('--golden-tol-abs', type=float, default=None, help='Absolute tolerance for golden checks')
    ap.add_argument('--golden-tol-pct', type=float, default=None, help='Percent tolerance (0.05 = 5%) for golden checks')
    args = ap.parse_args()

    result = load_json(Path(args.result))
    base = load_json(Path(args.baseline))

    # Ensure base structure
    base.setdefault('entry', Path(args.baseline).stem)
    base.setdefault('min_count', 1)
    base.setdefault('modes', {})

    cnt = int(result.get('count', 0))
    if cnt > 0:
        base['min_count'] = max(int(base.get('min_count', 1)), cnt)

    # Update per-mode thresholds: set max_avg_time to 1.25x current avg (guardrail)
    summary = result.get('summary') or {}
    for mode, d in (summary.items() if isinstance(summary, dict) else []):
        avg_time = d.get('avg_solve_time')
        if avg_time is None:
            continue
        try:
            thresh = float(avg_time) * 1.25
        except Exception:
            continue
        base['modes'].setdefault(mode, {})
        base['modes'][mode]['max_avg_time'] = max(float(base['modes'][mode].get('max_avg_time', 0) or 0), float(thresh))

    # Optional golden updates via environment
    gold_list = args.golden or os.environ.get('OPTIBENCH_GOLDEN_INSTANCES')
    tol_abs = args.golden_tol_abs if args.golden_tol_abs is not None else float(os.environ.get('OPTIBENCH_GOLDEN_TOL_ABS', '0') or 0)
    tol_pct = args.golden_tol_pct if args.golden_tol_pct is not None else float(os.environ.get('OPTIBENCH_GOLDEN_TOL_PCT', '0') or 0)
    if gold_list:
        base.setdefault('golden', {})
        # index result objectives by (instance, mode)
        idx = {}
        for r in result.get('results', []):
            inst = r.get('instance'); mode = r.get('mode'); obj = r.get('objective')
            if inst and mode and obj is not None:
                try:
                    idx[(inst, mode)] = float(obj)
                except Exception:
                    pass
        # auto:N select N instances from results if requested
        selected: list[str]
        if gold_list.startswith('auto:'):
            try:
                k = int(gold_list.split(':',1)[1])
            except Exception:
                k = 1
            insts = sorted({i for (i, _) in idx.keys()})
            selected = insts[:k]
        else:
            selected = [s.strip() for s in gold_list.split(',') if s.strip()]
        for inst in selected:
            # pick all modes present for this instance
            modes = {m for (i, m) in idx.keys() if i == inst}
            if not modes:
                continue
            base['golden'].setdefault(inst, {})
            for m in modes:
                val = idx.get((inst, m))
                if val is None:
                    continue
                base['golden'][inst].setdefault(m, {})
                base['golden'][inst][m]['target'] = val
                if tol_abs:
                    base['golden'][inst][m]['absolute_tolerance'] = tol_abs
                if tol_pct:
                    base['golden'][inst][m]['percent_tolerance'] = tol_pct

    outp = Path(args.out) if args.out else Path(args.baseline)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(base, indent=2))
    print(f"Wrote {outp}")


if __name__ == '__main__':
    main()
