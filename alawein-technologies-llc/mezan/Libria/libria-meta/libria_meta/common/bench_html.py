from __future__ import annotations

import html
from typing import Any, Dict, List


def generate_html(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    modes = list((summary.get('summary') or {}).keys())
    times = [(summary.get('summary') or {}).get(m, {}).get('avg_solve_time') or 0 for m in modes]
    labels_js = '[' + ','.join([f"'"+m+"'" for m in modes]) + ']'
    times_js = '[' + ','.join([str(x if x is not None else 0) for x in times]) + ']'

    # Tables
    rows_mode = ''.join([
        f"<tr><td>{html.escape(m)}</td><td>{d.get('count')}</td><td>{d.get('avg_objective')}</td><td>{d.get('avg_solve_time')}</td></tr>"
        for m, d in (summary.get('summary') or {}).items()
    ])
    rows_buckets = ''.join([
        ''.join([f"<tr><td>{html.escape(b)}</td><td>{html.escape(m)}</td><td>{dd.get('count')}</td><td>{dd.get('avg_objective')}</td><td>{dd.get('avg_solve_time')}</td></tr>" for m, dd in modes_map.items()])
        for b, modes_map in (summary.get('buckets') or {}).items()
    ])
    rows_results = ''.join([
        f"<tr><td>{html.escape(str(r.get('instance')))}</td><td>{r.get('n')}</td><td>{html.escape(str(r.get('mode')))}</td><td>{r.get('objective')}</td><td>{r.get('solve_time')}</td><td>{r.get('bound','')}</td><td>{r.get('bound_gap','')}</td><td>{r.get('bound_gap_pct','')}</td><td>{html.escape(str(r.get('error','')))}</td></tr>"
        for r in results
    ])

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>QAPFlow Bench Report</title>"
        "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 10px} .chart{margin:12px 0}</style>"
        "</head><body>"
        "<h1>QAPFlow Bench Report</h1>"
        "<h2>Avg Solve Time by Mode</h2>"
        "<canvas id='c' width='720' height='260' class='chart'></canvas>"
        "<script>const labels=" + labels_js + "; const data=" + times_js + ";"
        "const canvas=document.getElementById('c'); const ctx=canvas.getContext('2d');"
        "const W=canvas.width, H=canvas.height; const pad=40; const max=Math.max(1, ...data);"
        "ctx.clearRect(0,0,W,H); ctx.strokeStyle='#ccc'; ctx.beginPath(); ctx.moveTo(pad,10); ctx.lineTo(pad,H-30); ctx.lineTo(W-10,H-30); ctx.stroke();"
        "ctx.fillStyle='#000'; ctx.fillText('Avg Solve Time', 10, 10); ctx.fillText('Modes', W-70, H-10);"
        "const n=data.length; const bw=(W-pad-20)/Math.max(1,n); for(let i=0;i<n;i++){const v=data[i]; const x=pad+ i*bw + 8; const h=(H-50)*(v/max); const y=(H-30)-h; ctx.fillStyle='#4a90e2'; ctx.fillRect(x,y,bw-16,h); ctx.fillStyle='#000'; ctx.fillText(labels[i], x, H-12); ctx.fillText(String(v.toFixed? v.toFixed(2):v), x, y-4);}"
        "</script>"
        "<h2>Summary (per mode)</h2>"
        "<table><tr><th>Mode</th><th>Count</th><th>Avg Objective</th><th>Avg Solve Time</th></tr>" + rows_mode + "</table>"
        "<h2>Summary (by size bucket)</h2>"
        "<table><tr><th>Bucket</th><th>Mode</th><th>Count</th><th>Avg Objective</th><th>Avg Solve Time</th></tr>" + rows_buckets + "</table>"
        "<h2>Results</h2>"
        "<table><tr><th>Instance</th><th>n</th><th>Mode</th><th>Objective</th><th>Solve Time</th><th>Bound</th><th>Bound Gap</th><th>Gap %</th><th>Error</th></tr>" + rows_results + "</table>"
        "</body></html>"
    )

