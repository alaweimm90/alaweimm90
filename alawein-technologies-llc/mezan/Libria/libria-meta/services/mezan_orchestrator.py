from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Tuple, Optional, List
import os
import sys


def _load_env_paths():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(root, 'libria-integration'))
    sys.path.insert(0, root)


_load_env_paths()
from libria_integration import LibriaRouter  # type: ignore
import Librex.QAP_backend  # type: ignore
from libria_meta.mezan_engine.orchestrator import MezanOrchestrator
from libria_meta.schemas.validate import validate
from libria_meta.common.bench_utils import load_qaplib_dat as _bench_load_qaplib_dat
from libria_meta.solvers.qapflow import QAPFlow
from libria_meta.common.bench_html import generate_html as _bench_generate_html
import pathlib
import html as htmllib


class OrchestratorHandler(BaseHTTPRequestHandler):
    router = LibriaRouter()
    schemas_dir = pathlib.Path(__file__).resolve().parents[1] / 'libria_meta' / 'schemas'
    engine = MezanOrchestrator()
    jobs_lock = threading.Lock()
    jobs: Dict[str, Dict[str, Any]] = {}
    job_seq: int = 0

    def _send_json(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        max_body = int(os.environ.get('ORCH_MAX_BODY', str(2 * 1024 * 1024)))
        length = int(self.headers.get('Content-Length', '0'))
        if length > max_body:
            self._send_json(413, {"error": "payload_too_large", "max_body": max_body})
            return
        data = self.rfile.read(length)
        payload = None
        ctype = self.headers.get('Content-Type','').lower()
        if 'application/json' in ctype:
            try:
                payload = json.loads(data.decode('utf-8'))
            except Exception:
                self._send_json(400, {"error": "invalid_json"}); return
        elif 'application/x-www-form-urlencoded' in ctype or 'multipart/form-data' in ctype:
            try:
                from urllib.parse import parse_qs
                parsed = parse_qs(data.decode('utf-8'))
                payload = {k: (v[0] if isinstance(v, list) and v else v) for k, v in parsed.items()}
                # normalize known fields
                if 'modes' in payload and isinstance(payload['modes'], str):
                    payload['modes'] = [m.strip() for m in payload['modes'].split(',') if m.strip()]
                for k in ('time_limit',):
                    if k in payload:
                        try:
                            payload[k] = int(payload[k])
                        except Exception:
                            pass
                for k in ('robust_eps',):
                    if k in payload:
                        try:
                            payload[k] = float(payload[k])
                        except Exception:
                            pass
            except Exception:
                self._send_json(400, {"error": "invalid_form"}); return
        else:
            try:
                payload = json.loads(data.decode('utf-8'))
            except Exception:
                payload = {}

        try:
            if self.path in ('/qapflow/solve','/qapflow/solve_ab'):
                schema = json.loads((self.schemas_dir / 'qapflow_problem.json').read_text())
                # Require either 'fit' or ('A' and 'B') manually before schema validation
                if not ((isinstance(payload, dict)) and (('fit' in payload) or ('A' in payload and 'B' in payload))):
                    self._send_json(400, {"error": "require 'fit' or both 'A' and 'B'"})
                    return
                validate(payload, schema)
                params = {}
                for k in ("mode", "time_limit", "backend", "robust_eps", "seed"):
                    if k in payload:
                        params[k] = payload[k]
                if os.environ.get('QAPFLOW_SAFE_MODE'):
                    params['safe_mode'] = True
                # Optional request timeout
                req_timeout = float(os.environ.get('ORCH_REQUEST_TIMEOUT', '0'))
                res_holder: Dict[str, Any] = {}
                err_holder: Dict[str, Any] = {}

                def _run():
                    try:
                        res_holder['res'] = Librex.QAP_backend.solve(payload, parameters=params or None)
                    except Exception as e:  # pragma: no cover - error propagation
                        err_holder['err'] = str(e)

                if req_timeout > 0:
                    t = threading.Thread(target=_run, daemon=True)
                    t.start()
                    t.join(timeout=req_timeout)
                    if t.is_alive():
                        self._send_json(408, {"error": "timeout", "timeout_seconds": req_timeout})
                        return
                    if 'err' in err_holder:
                        self._send_json(400, {"error": err_holder['err']})
                        return
                    res = res_holder.get('res')
                else:
                    # Use backend adapter to support both A/B and fit/interaction
                    res = Librex.QAP_backend.solve(payload, parameters=params or None)
                # Optional bound-gap enrichment (minimization expected)
                try:
                    obj = float(res.get('objective')) if res.get('objective') is not None else None
                    bd = res.get('bound')
                    if obj is not None and bd is not None:
                        bdv = float(bd)
                        gap = obj - bdv
                        denom = max(1.0, abs(obj))
                        pct = gap / denom
                        tol = 1e-6 * denom
                        consistent = (obj + tol) >= bdv
                        res.setdefault('metadata', {})
                        res['metadata'].update({
                            'bound_gap': gap,
                            'bound_gap_pct': pct,
                            'bound_consistent': bool(consistent),
                        })
                except Exception:
                    pass
                sol_schema = json.loads((self.schemas_dir / 'qapflow_solution.json').read_text())
                validate(res, sol_schema)
                self._send_json(200, res)
                return
            if self.path == '/bench':
                # Start an async bench job. Payload example:
                # {"type": "qaplib", "modes": ["hybrid","nesterov"], "instances": "tai20a,nug20", "time_limit": 30, "backend": "external"}
                job_id = self._start_bench_job(payload)
                if not job_id:
                    self._send_json(400, {"error": "invalid_bench_request"})
                    return
                self._send_json(202, {"job_id": job_id, "status": "running"})
                return
            if self.path == '/bench/ui/new':
                job_id = self._start_bench_job(payload or {})
                if not job_id:
                    self._send_json(400, {"error": "invalid_bench_request"}); return
                # Redirect to job detail
                self.send_response(303)
                self.send_header('Location', f"/bench/ui/{job_id}")
                self.end_headers()
                return
            if self.path == '/workflow/solve':
                schema = json.loads((self.schemas_dir / 'workflow_problem.json').read_text())
                validate(payload, schema)
                res = self.router._wf.solve(payload)
                sol_schema = json.loads((self.schemas_dir / 'workflow_solution.json').read_text())
                validate(res, sol_schema)
                self._send_json(200, res)
                return
            if self.path == '/allocflow/solve':
                schema = json.loads((self.schemas_dir / 'allocflow_problem.json').read_text())
                validate(payload, schema)
                res = self.router._alloc.solve(payload)
                sol_schema = json.loads((self.schemas_dir / 'allocflow_solution.json').read_text())
                validate(res, sol_schema)
                self._send_json(200, res)
                return
            if self.path == '/graphflow/solve':
                res = self.engine.adapter.solve_graphflow(payload)
                self._send_json(200, res)
                return
            if self.path == '/metaflow/solve':
                res = self.engine.adapter.solve_metaflow(payload)
                self._send_json(200, res)
                return
            if self.path == '/dualflow/solve':
                res = self.engine.adapter.solve_dualflow(payload)
                self._send_json(200, res)
                return
            if self.path == '/evoflow/solve':
                res = self.engine.adapter.solve_evoflow(payload)
                self._send_json(200, res)
                return
            if self.path == '/mezan/solve':
                schema = json.loads((self.schemas_dir / 'mezan_composite.json').read_text())
                validate(payload, schema)
                res = self.engine.solve_composite(payload)
                self._send_json(200, res)
                return
            self._send_json(404, {"error": "not_found"})
        except Exception as e:
            self._send_json(400, {"error": str(e)})

    def do_GET(self):  # noqa: N802
        if self.path == '/health':
            modes = []
            details = {}
            try:
                # Detect modular repo presence and available modes by import checks
                import importlib
                available = {}
                for mod, attr in [
                    ('solvers.hybrid_combined_solver','hybrid_combined_solver'),
                    ('solvers.fft_accelerated_solver','fft_accelerated_solver'),
                    ('solvers.enhanced_solver','enhanced_solver'),
                    ('solvers.nesterov_accelerated_solver','nesterov_accelerated_solver'),
                    ('solvers.instance_adaptive_optimizer','adaptive_solver_with_params'),
                    ('solvers.aggressive_solver','aggressive_solver'),
                ]:
                    try:
                        importlib.import_module(mod)
                        available[attr] = True
                    except Exception:
                        available[attr] = False
                modes = [k for k,v in available.items() if v]
                details = available
            except Exception:
                modes = []
            env_path = os.environ.get('QAP_MODULAR_REPO_PATH','')
            self._send_json(200, {"status": "ok", "modes": modes, "env_path": env_path, "details": details})
        elif self.path == '/ui':
            # Main dashboard: shows health and links to bench UI
            # Determine latest OptiBench nightly HTML if available
            latest_optibench = ''
            try:
                base_dir = pathlib.Path(__file__).resolve().parents[2]
                opti_dir = base_dir / 'results' / 'optibench'
                files = sorted(opti_dir.glob('*.html'), key=lambda p: p.stat().st_mtime, reverse=True)
                if files:
                    latest_name = files[0].name
                    latest_label = files[0].stem
                    latest_optibench = f" · Latest Nightly: <a href='/bench/optibench/{latest_name}'>{htmllib.escape(latest_label)}</a>"
            except Exception:
                latest_optibench = ''
            html = (
                "<!doctype html><html><head><meta charset='utf-8'><title>MEZAN Dashboard</title>"
                "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 10px} .section{margin-bottom:24px} .card{border:1px solid #ddd;padding:12px;border-radius:8px}</style>"
                "</head><body><h1>MEZAN Dashboard</h1>"
                "<div class='section'><h2>Health</h2><div id='health'>Loading...</div></div>"
                ("<div class='section'><h2>Bench</h2><p><a href='/bench/ui'>Open Bench UI</a> · <a href='/bench/ui/new'>Start New Job</a> · <a href='/bench/reports/index'>Reports Index</a> · <a href='/bench/optibench/index'>OptiBench Index</a>" + latest_optibench + "</p><div id='jobs'>Loading...</div>" +
                 "<div class='card' style='margin-top:12px'><h3 style='margin-top:0'>Nightly Summary</h3>"+
                 "<div style='margin-bottom:8px'>Select: <select id='nightlySel'></select> · <label><input type='checkbox' id='gapToggle'> Show gap chart</label></div>"+
                 "<div id='latest'>Loading...</div><canvas id='latestChart' width='480' height='200' style='display:block;margin-top:10px'></canvas></div></div>")
                "<script>async function loadHealth(){try{let r=await fetch('/health');let j=await r.json();"
                "let h='<table><tr><th>Status</th><th>Repo Path</th><th>Modes</th></tr>';"
                "h+=`<tr><td>${j.status}</td><td>${j.env_path||''}</td><td>${(j.modes||[]).join(', ')}</td></tr>`;"
                "h+='</table>';document.getElementById('health').innerHTML=h;}catch(e){document.getElementById('health').textContent='Error: '+e}}"
                "async function loadJobs(){try{let r=await fetch('/bench/summary');let j=await r.json();"
                "let h='<table><tr><th>Job ID</th><th>Status</th><th>Created</th></tr>';"
                "for(const [id,info] of Object.entries(j.jobs||{})){h+=`<tr><td><a href='/bench/ui/${id}'>${id}</a></td><td>${info.status}</td><td>${info.created}</td></tr>`;}"
                "h+='</table>';document.getElementById('jobs').innerHTML=h;}catch(e){document.getElementById('jobs').textContent='Error: '+e}}"
                "async function drawChart(labels, data, ylabel){const c=document.getElementById('latestChart'); const ctx=c.getContext('2d'); const W=c.width, H=c.height; const pad=40; const max=Math.max(1,...data); ctx.clearRect(0,0,W,H); ctx.strokeStyle='#ccc'; ctx.beginPath(); ctx.moveTo(pad,10); ctx.lineTo(pad,H-30); ctx.lineTo(W-10,H-30); ctx.stroke(); ctx.fillStyle='#000'; ctx.fillText(ylabel, 10, 10); ctx.fillText('Modes', W-70, H-10); const n=data.length; const bw=(W-pad-20)/Math.max(1,n); for(let i=0;i<n;i++){const v=data[i]; const x=pad+ i*bw + 8; const h=(H-50)*(v/max); const y=(H-30)-h; ctx.fillStyle='#7fb3d5'; ctx.fillRect(x,y,bw-16,h); ctx.fillStyle='#000'; ctx.fillText(labels[i], x, H-12); ctx.fillText(String(v.toFixed? v.toFixed(2):v), x, y-4);} }
                async function loadNightlyList(){try{let sel=document.getElementById('nightlySel'); let r=await fetch('/bench/optibench/list.json'); if(!r.ok){sel.innerHTML='<option>None</option>'; return;} let j=await r.json(); sel.innerHTML=''; (j.files||[]).forEach((f,i)=>{let opt=document.createElement('option'); opt.value=f.name; opt.textContent=f.name; if(i==0) opt.selected=true; sel.appendChild(opt);});}catch(e){console.log(e);}}
                async function loadLatest(){try{let sel=document.getElementById('nightlySel'); let name= sel && sel.value; let endpoint = name? ('/bench/optibench/summary.json?name='+encodeURIComponent(name)) : '/bench/optibench/latest.json'; let r=await fetch(endpoint); if(!r.ok){document.getElementById('latest').textContent='No nightly summary found'; return;} let j=await r.json(); let modes=Object.keys(j.summary||{}); if(modes.length==0){document.getElementById('latest').textContent='No data'; return;} let h='<table><tr><th>Mode</th><th>Avg Time</th><th>Count</th><th>Avg Gap</th></tr>'; for(const m of modes){let d=j.summary[m]||{}; let g=(j.gaps&&j.gaps[m]&&j.gaps[m].avg_gap)||''; h+=`<tr><td>${m}</td><td>${(d.avg_solve_time??'')}</td><td>${(d.count??'')}</td><td>${g}</td></tr>`;} h+='</table>'; document.getElementById('latest').innerHTML=h; let gapOn=document.getElementById('gapToggle').checked; const labels=modes; const data = gapOn ? modes.map(m => (j.gaps&&j.gaps[m]&&j.gaps[m].avg_gap)||0) : modes.map(m => (j.summary[m]||{}).avg_solve_time||0); await drawChart(labels, data, gapOn? 'Avg Gap' : 'Avg Solve Time'); }catch(e){document.getElementById('latest').textContent='Error: '+e}}
                "loadHealth();loadJobs();setInterval(loadHealth,5000);setInterval(loadJobs,5000);</script>"
                "<script>loadNightlyList().then(loadLatest); document.getElementById('gapToggle').addEventListener('change', loadLatest); document.getElementById('nightlySel').addEventListener('change', loadLatest);</script>"
                "</body></html>"
            ).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == '/bench/ui/history':
            # Render simple history viewer if file exists
            try:
                import json as _json
                from pathlib import Path as _Path
                base_dir = _Path(__file__).resolve().parents[2]
                hist = base_dir / 'results' / 'bench_history.jsonl'
                entries = []
                if hist.exists():
                    with hist.open() as hf:
                        for line in hf:
                            try:
                                entries.append(_json.loads(line))
                            except Exception:
                                continue
                rows = ''.join([f"<tr><td><a href='/bench/ui/{htmllib.escape(e.get('job_id',''))}'>{htmllib.escape(e.get('job_id',''))}</a></td><td>{htmllib.escape(e.get('completed_at',''))}</td><td>{htmllib.escape(str(e.get('params',{})))}</td></tr>" for e in entries])
                html_doc = ("<!doctype html><html><head><meta charset='utf-8'><title>Bench History</title>"
                            "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 10px}</style>"
                            "</head><body><h1>Bench History</h1>"
                            "<p><a href='/ui'>Back to dashboard</a> | <a href='/bench/ui'>Bench list</a></p>"
                            "<table><tr><th>Job ID</th><th>Completed</th><th>Params</th></tr>" + rows + "</table>"
                            "</body></html>")
                body = html_doc.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        elif self.path.startswith('/bench/reports/index'):
            # Serve dynamic reports index
            try:
                from pathlib import Path as _Path
                base_dir = _Path(__file__).resolve().parents[2]
                rep_dir = base_dir / 'results' / 'bench_reports'
                files = sorted(rep_dir.glob('*.html'), key=lambda p: p.stat().st_mtime, reverse=True)
                items = ''.join([f"<li><a href='/bench/report/{p.stem}.html'>{htmllib.escape(p.stem)}</a></li>" for p in files])
                latest = files[0].stem if files else 'N/A'
                html_doc = ("<!doctype html><html><head><meta charset='utf-8'><title>Bench Reports</title>"
                            "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;}</style></head><body>"
                            "<h1>Bench Reports</h1>"
                            f"<p>Latest: <strong>{htmllib.escape(latest)}</strong></p>"
                            f"<ul>{items}</ul>"
                            "<p><a href='/ui'>Back to dashboard</a></p>"
                            "</body></html>")
                body = html_doc.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        elif self.path.startswith('/bench/summary'):
            with self.jobs_lock:
                listing = {jid: {"status": j.get('status'), "created": j.get('created')} for jid, j in self.jobs.items()}
            self._send_json(200, {"jobs": listing})
        elif self.path == '/bench/optibench/latest.json':
            # Return latest optibench summary JSON (trimmed to summary only)
            try:
                opti_dir = pathlib.Path(__file__).resolve().parents[2] / 'results' / 'optibench'
                files = sorted(opti_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
                if not files:
                    self._send_json(404, {"error": "not_found"}); return
                import json as _json
                data = _json.loads(files[0].read_text())
                # compute gaps by mode if results contain bounds
                gaps = {}
                try:
                    for r in data.get('results') or []:
                        b = r.get('bound'); o = r.get('objective'); m = r.get('mode')
                        if b is None or o is None or m is None: continue
                        try:
                            g = float(o) - float(b)
                        except Exception:
                            continue
                        rec = gaps.setdefault(m, {'sum':0.0,'n':0})
                        rec['sum'] += g; rec['n'] += 1
                    gaps = {k: {'avg_gap': (v['sum']/v['n'] if v['n'] else 0.0), 'count': v['n']} for k,v in gaps.items()}
                except Exception:
                    gaps = {}
                out = {"summary": data.get('summary') or {}, "count": data.get('count'), "gaps": gaps}
                self._send_json(200, out)
            except Exception as e:
                self._send_json(404, {"error": str(e)})
        elif self.path.startswith('/bench/optibench/summary.json'):
            # Return summary for specified name query
            try:
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(self.path).query)
                name = (q.get('name') or [None])[0]
                if not name:
                    self._send_json(400, {"error": "missing_name"}); return
                opti_dir = pathlib.Path(__file__).resolve().parents[2] / 'results' / 'optibench'
                path = (opti_dir / name).resolve()
                if not str(path).startswith(str(opti_dir.resolve())):
                    self._send_json(400, {"error": "invalid_path"}); return
                if not path.exists():
                    self._send_json(404, {"error": "not_found"}); return
                import json as _json
                data = _json.loads(path.read_text())
                gaps = {}
                try:
                    for r in data.get('results') or []:
                        b = r.get('bound'); o = r.get('objective'); m = r.get('mode')
                        if b is None or o is None or m is None: continue
                        try:
                            g = float(o) - float(b)
                        except Exception:
                            continue
                        rec = gaps.setdefault(m, {'sum':0.0,'n':0})
                        rec['sum'] += g; rec['n'] += 1
                    gaps = {k: {'avg_gap': (v['sum']/v['n'] if v['n'] else 0.0), 'count': v['n']} for k,v in gaps.items()}
                except Exception:
                    gaps = {}
                out = {"summary": data.get('summary') or {}, "count": data.get('count'), "gaps": gaps}
                self._send_json(200, out)
            except Exception as e:
                self._send_json(404, {"error": str(e)})
        elif self.path == '/bench/optibench/list.json':
            try:
                opti_dir = pathlib.Path(__file__).resolve().parents[2] / 'results' / 'optibench'
                files = sorted(opti_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
                out = {"files": [{"name": f.name, "mtime": int(f.stat().st_mtime)} for f in files]}
                self._send_json(200, out)
            except Exception as e:
                self._send_json(200, {"files": []})
        elif self.path.startswith('/bench/optibench/index'):
            # List OptiBench HTML reports saved locally by nightly
            try:
                from pathlib import Path as _Path
                base_dir = _Path(__file__).resolve().parents[2]
                opti_dir = base_dir / 'results' / 'optibench'
                files = sorted(opti_dir.glob('*.html'), key=lambda p: p.stat().st_mtime, reverse=True)
                items = ''.join([f"<li><a href='/bench/optibench/{p.name}'>{htmllib.escape(p.stem)}</a></li>" for p in files])
                latest = files[0].name if files else ''
                html_doc = ("<!doctype html><html><head><meta charset='utf-8'><title>OptiBench Reports</title>"
                            "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;}</style></head><body>"
                            "<h1>OptiBench Reports</h1>"
                            f"<p>Latest: {('<a href=\'/bench/optibench/'+latest+"\'>"+htmllib.escape(latest)+"</a>") if latest else 'N/A'}</p>"
                            f"<ul>{items}</ul>"
                            "<p><a href='/ui'>Back to dashboard</a></p>"
                            "</body></html>")
                body = html_doc.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        elif self.path.startswith('/bench/optibench/') and self.path.endswith('.html'):
            # Serve OptiBench HTML produced by nightly
            try:
                from pathlib import Path as _Path
                base_dir = _Path(__file__).resolve().parents[2]
                opti_dir = base_dir / 'results' / 'optibench'
                path = (opti_dir / self.path.split('/bench/optibench/',1)[1]).resolve()
                if not str(path).startswith(str(opti_dir.resolve())):
                    self._send_json(400, {"error": "invalid_path"}); return
                if not path.exists():
                    self._send_json(404, {"error": "not_found"}); return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        elif self.path.startswith('/bench/ui'):
            html = (
                "<!doctype html><html><head><meta charset='utf-8'><title>MEZAN Bench</title>"
                "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 10px}</style>"
                "</head><body><h1>Bench Jobs</h1>"
                "<div id='jobs'>Loading...</div>"
                "<script>async function load(){try{let r=await fetch('/bench/summary');let j=await r.json();"
                "let h='<table><tr><th>Job ID</th><th>Status</th><th>Created</th></tr>';"
                "for(const [id,info] of Object.entries(j.jobs||{})){h+=`<tr><td><a href='/bench/${id}'>${id}</a></td><td>${info.status}</td><td>${info.created}</td></tr>`;}"
                "h+='</table>';document.getElementById('jobs').innerHTML=h;}catch(e){document.getElementById('jobs').textContent='Error: '+e}}"
                "load();setInterval(load,3000);</script></body></html>"
            ).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == '/bench/ui/new':  # creation form
            form = (
                "<!doctype html><html><head><meta charset='utf-8'><title>New Bench Job</title>"
                "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;} label{display:block;margin:8px 0}</style>"
                "</head><body><h1>Start Bench Job</h1>"
                "<form method='post' action='/bench/ui/new'>"
                "<label>Type <input name='type' value='qaplib'></label>"
                "<label>Modes (comma) <input name='modes' value='hybrid'></label>"
                "<label>Instances (comma) <input name='instances' placeholder='tai20a,nug20'></label>"
                "<label>Time limit (sec) <input name='time_limit' value='30'></label>"
                "<label>Backend <input name='backend' placeholder='external'></label>"
                "<label>Robust eps <input name='robust_eps' value='0.0'></label>"
                "<label>Data dir (optional) <input name='data_dir' placeholder='/path/to/qaplib'></label>"
                "<button type='submit'>Start</button>"
                "</form>"
                "<p><a href='/bench/ui'>Back to bench</a> | <a href='/ui'>Dashboard</a></p>"
                "</body></html>"
            ).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(form)))
            self.end_headers()
            self.wfile.write(form)
        elif self.path.startswith('/bench/ui/'):  # detailed job HTML
            jid = self.path.split('/bench/ui/', 1)[1].strip()
            with self.jobs_lock:
                job = self.jobs.get(jid)
            if not job:
                self._send_json(404, {"error": "job_not_found"})
                return
            # Build a simple HTML view with results table
            rows = []
            for r in job.get('results', []):
                rows.append(f"<tr><td>{r.get('instance')}</td><td>{r.get('n')}</td><td>{r.get('mode')}</td><td>{r.get('objective')}</td><td>{r.get('solve_time')}</td><td>{r.get('bound','')}</td><td>{r.get('bound_gap','')}</td><td>{r.get('bound_gap_pct','')}</td><td>{r.get('error','')}</td></tr>")
            # Prepare per-mode data for simple bar chart (avg solve time)
            per_mode = job.get('summary',{}).get('summary') or {}
            modes = list(per_mode.keys())
            times = [ (per_mode[m].get('avg_solve_time') or 0) for m in modes ]
            labels_js = '[' + ','.join(["'" + m + "'" for m in modes]) + ']'
            times_js = '[' + ','.join([str(x if x is not None else 0) for x in times]) + ']'
            html = ("<!doctype html><html><head><meta charset='utf-8'><title>Bench Job " + jid + "</title>"
                    "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 10px} .ok{color:#090}.err{color:#900} .chart{margin:12px 0}</style>"
                    "</head><body>"
                    f"<h1>Bench Job {jid}</h1>"
                    f"<p>Status: <strong>{job.get('status')}</strong> | Created: {job.get('created')}</p>"
                    "<h2>Avg Solve Time by Mode</h2>"
                    "<canvas id='c' width='640' height='240' class='chart'></canvas>"
                    "<script>const labels=" + labels_js + "; const data=" + times_js + ";"
                    "const canvas=document.getElementById('c'); const ctx=canvas.getContext('2d');"
                    "const W=canvas.width, H=canvas.height; const pad=40; const max=Math.max(1, ...data);"
                    "ctx.clearRect(0,0,W,H); ctx.strokeStyle='#ccc'; ctx.beginPath(); ctx.moveTo(pad,10); ctx.lineTo(pad,H-30); ctx.lineTo(W-10,H-30); ctx.stroke();"
                    "ctx.fillStyle='#000'; ctx.fillText('Avg Solve Time', 10, 10); ctx.fillText('Modes', W-70, H-10);"
                    "const n=data.length; const bw=(W-pad-20)/Math.max(1,n); for(let i=0;i<n;i++){const v=data[i]; const x=pad+ i*bw + 8; const h=(H-50)*(v/max); const y=(H-30)-h; ctx.fillStyle='#4a90e2'; ctx.fillRect(x,y,bw-16,h); ctx.fillStyle='#000'; ctx.fillText(labels[i], x, H-12); ctx.fillText(String(v.toFixed? v.toFixed(2):v), x, y-4);}"
                    "</script>"
                    "<h2>Summary (per mode)</h2>"
                    "<table><tr><th>Mode</th><th>Count</th><th>Avg Objective</th><th>Avg Solve Time</th></tr>" +
                    ''.join([f"<tr><td>{m}</td><td>{d.get('count')}</td><td>{d.get('avg_objective')}</td><td>{d.get('avg_solve_time')}</td></tr>" for m,d in (job.get('summary',{}).get('summary') or {}).items()]) +
                    "</table>"
                    "<h2>Summary (by size bucket)</h2>"
                    "<table><tr><th>Bucket</th><th>Mode</th><th>Count</th><th>Avg Objective</th><th>Avg Solve Time</th></tr>" +
                    ''.join([ ''.join([f"<tr><td>{b}</td><td>{m}</td><td>{dd.get('count')}</td><td>{dd.get('avg_objective')}</td><td>{dd.get('avg_solve_time')}</td></tr>" for m,dd in modes.items()]) for b,modes in (job.get('summary',{}).get('buckets') or {}).items()]) +
                    "</table>"
                    "<h2>Results</h2>"
                    "<table><tr><th>Instance</th><th>n</th><th>Mode</th><th>Objective</th><th>Solve Time</th><th>Bound</th><th>Bound Gap</th><th>Gap %</th><th>Error</th></tr>" + ''.join(rows) + "</table>"
                    f"<p><a href='/bench/ui'>Back to jobs</a> | <a href='/ui'>Dashboard</a> | <a href='/bench/{jid}.csv'>Download CSV</a></p>"
                    "</body></html>")
            body = html.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith('/bench/') and self.path.endswith('.csv'):
            # CSV export for a job: /bench/{job_id}.csv
            jid = self.path.rsplit('/', 1)[-1].removesuffix('.csv')
            with self.jobs_lock:
                job = self.jobs.get(jid)
            if not job:
                self._send_json(404, {"error": "job_not_found"})
                return
            rows = job.get('results') or []
            import csv, io
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["instance","n","mode","objective","solve_time","bound","bound_gap","bound_gap_pct","error"])
            for r in rows:
                w.writerow([r.get('instance'), r.get('n'), r.get('mode'), r.get('objective'), r.get('solve_time'), r.get('bound'), r.get('bound_gap'), r.get('bound_gap_pct'), r.get('error')])
            data = buf.getvalue().encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/csv; charset=utf-8')
            self.send_header('Content-Disposition', f"attachment; filename={jid}.csv")
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith('/bench/report/') and self.path.endswith('.html'):
            # Serve saved report
            job_html = self.path.split('/bench/report/', 1)[1]
            try:
                from pathlib import Path as _Path
                base_dir = _Path(__file__).resolve().parents[2]
                rep_dir = base_dir / 'results' / 'bench_reports'
                path = (rep_dir / job_html).resolve()
                if not str(path).startswith(str(rep_dir.resolve())):
                    self._send_json(400, {"error": "invalid_path"}); return
                if not path.exists():
                    self._send_json(404, {"error": "not_found"}); return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        elif self.path.startswith('/bench/'):
            jid = self.path.split('/bench/', 1)[1].strip()
            with self.jobs_lock:
                job = self.jobs.get(jid)
            if not job:
                self._send_json(404, {"error": "job_not_found"})
                return
            self._send_json(200, job)
        else:
            self._send_json(404, {"error": "not_found"})

    # Internal helpers
    def _start_bench_job(self, payload: Dict[str, Any]) -> Optional[str]:
        btype = str(payload.get('type', 'qaplib')).lower()
        if btype != 'qaplib':
            return None
        modes = payload.get('modes') or ([] if payload.get('mode') is None else [payload.get('mode')])
        if isinstance(modes, str):
            modes = [m.strip() for m in modes.split(',') if m.strip()]
        if not modes:
            modes = ['hybrid']
        instances = payload.get('instances')
        time_limit = int(payload.get('time_limit') or 30)
        backend = payload.get('backend')
        robust_eps = float(payload.get('robust_eps') or 0.0)
        qap_dir = payload.get('data_dir') or os.environ.get('QAPLIB_DATA_DIR')
        if not qap_dir:
            return None
        import time
        with self.jobs_lock:
            OrchestratorHandler.job_seq += 1
            jid = f"b-{OrchestratorHandler.job_seq}"
            OrchestratorHandler.jobs[jid] = {
                "status": "running",
                "created": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "params": {"type": btype, "modes": modes, "instances": instances, "time_limit": time_limit, "backend": backend, "robust_eps": robust_eps},
            }

        def _runner():
            try:
                from pathlib import Path as _Path
                files = sorted([p for p in _Path(qap_dir).glob('*.dat')])
                if instances:
                    pats = [s.strip().lower() for s in str(instances).split(',') if s.strip()]
                    files = [f for f in files if any(p in f.stem.lower() for p in pats)]
                results: List[Dict[str, Any]] = []
                solver = QAPFlow()
                for f in files:
                    try:
                        A, B = _bench_load_qaplib_dat(f)
                        problem = {"A": A, "B": B}
                        n = len(A)
                        for m in modes:  # type: ignore
                            params = {"mode": m, "time_limit": time_limit}
                            if backend:
                                params["backend"] = backend
                            if robust_eps and robust_eps > 0:
                                params["robust_eps"] = robust_eps
                            sol = solver.solve(problem, parameters=params)
                            rec = {
                                "instance": f.stem,
                                "n": n,
                                "mode": m,
                                "objective": sol.get("objective"),
                                "solve_time": sol.get("solve_time") or sol.get("metadata", {}).get("solve_time"),
                            }
                            # Optional bound + gap
                            bd = sol.get("bound")
                            if bd is not None and sol.get("objective") is not None:
                                try:
                                    obj = float(sol["objective"]) ; bdf = float(bd)
                                    gap = obj - bdf
                                    denom = max(1.0, abs(obj))
                                    rec["bound"] = bdf
                                    rec["bound_gap"] = gap
                                    rec["bound_gap_pct"] = gap / denom
                                except Exception:
                                    pass
                            results.append(rec)
                    except Exception as e:  # capture per-instance errors
                        results.append({"instance": f.stem, "error": str(e)})
                # Summaries
                from libria_meta.common.bench_utils import summarize_by_mode, summarize_by_bucket
                summary = {
                    "summary": summarize_by_mode(results),
                    "buckets": summarize_by_bucket(results),
                    "count": len(results),
                }
                with self.jobs_lock:
                    OrchestratorHandler.jobs[jid].update({"status": "done", "results": results, "summary": summary})
                # Optional bench history logging
                try:
                    if str(os.environ.get('ORCH_BENCH_HISTORY','0')).lower() in ('1','true','yes'):
                        base_dir = _Path(__file__).resolve().parents[2]  # Libria/libria-meta
                        hist = base_dir / 'results' / 'bench_history.jsonl'
                        hist.parent.mkdir(parents=True, exist_ok=True)
                        import time, json as _json
                        entry = {
                            'job_id': jid,
                            'completed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                            'params': {"type": btype, "modes": modes, "instances": instances, "time_limit": time_limit, "backend": backend, "robust_eps": robust_eps},
                            'summary': summary,
                        }
                        with hist.open('a') as hf:
                            hf.write(_json.dumps(entry) + "\n")
                    # Optional bench report HTML generation + index
                    if str(os.environ.get('ORCH_BENCH_GENERATE_HTML','0')).lower() in ('1','true','yes'):
                        base_dir = _Path(__file__).resolve().parents[2]
                        rep_dir = base_dir / 'results' / 'bench_reports'
                        rep_dir.mkdir(parents=True, exist_ok=True)
                        html_path = rep_dir / f"{jid}.html"
                        try:
                            html_text = _bench_generate_html(summary, results)
                            html_path.write_text(html_text, encoding='utf-8')
                        except Exception:
                            pass
                        # Update simple index page
                        try:
                            files = sorted(rep_dir.glob('*.html'), key=lambda p: p.stat().st_mtime, reverse=True)
                            items = ''.join([f"<li><a href='/bench/report/{p.stem}.html'>{htmllib.escape(p.stem)}</a></li>" for p in files])
                            idx = ("<!doctype html><html><head><meta charset='utf-8'><title>Bench Reports</title>"
                                   "<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;}</style></head><body>"
                                   "<h1>Bench Reports</h1>"
                                   f"<p>Latest: <strong>{htmllib.escape(files[0].stem) if files else 'N/A'}</strong></p>"
                                   f"<ul>{items}</ul>"
                                   "</body></html>")
                            (rep_dir / 'index.html').write_text(idx, encoding='utf-8')
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception as e:  # unexpected
                with self.jobs_lock:
                    OrchestratorHandler.jobs[jid].update({"status": "error", "error": str(e)})

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return jid


def run_server(host: str = '127.0.0.1', port: int = 8081) -> Tuple[HTTPServer, threading.Thread]:
    server = HTTPServer((host, port), OrchestratorHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server, th


if __name__ == '__main__':
    server, _ = run_server()
    print(f"MEZAN Orchestrator listening on http://{server.server_address[0]}:{server.server_address[1]}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
