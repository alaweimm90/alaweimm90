from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE = Path(__file__).resolve().parents[2]
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TOOLS_DIR / "build_matrix_config.json"
REPORT_ROOT = WORKSPACE / "automation" / "reports" / "build_matrix"
PACKAGE_SCRIPT_CACHE: Dict[Path, Dict[str, str]] = {}


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Build matrix config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "projects" in data and isinstance(data["projects"], dict):
        return data["projects"]
    return data


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(WORKSPACE).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_report_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = REPORT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_package_scripts(pkg_path: Path) -> Dict[str, str]:
    if pkg_path in PACKAGE_SCRIPT_CACHE:
        return PACKAGE_SCRIPT_CACHE[pkg_path]
    scripts: Dict[str, str] = {}
    if pkg_path.exists():
        try:
            pkg_data = json.loads(pkg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pkg_data = {}
        scripts = pkg_data.get("scripts", {}) if isinstance(pkg_data, dict) else {}
    PACKAGE_SCRIPT_CACHE[pkg_path] = scripts
    return scripts


def step_should_run(project_root: Path, step: Dict[str, Any]) -> Optional[str]:
    required_paths = step.get("if_files", []) or []
    missing = [rel for rel in required_paths if not (project_root / rel).exists()]
    if missing:
        return f"missing required path(s): {', '.join(missing)}"

    script_name = step.get("requires_npm_script")
    if script_name:
        package_rel = step.get("package_json", "package.json")
        pkg_path = project_root / package_rel
        scripts = load_package_scripts(pkg_path)
        if script_name not in scripts:
            return f"npm script '{script_name}' not defined in {relative_path(pkg_path)}"

    cwd_rel = step.get("cwd")
    if cwd_rel:
        target_dir = project_root / cwd_rel
        if not target_dir.exists():
            return f"cwd '{cwd_rel}' not found"

    return None


def run_command(cmd: List[str], cwd: Path, env: Dict[str, str], log_file: Path) -> Dict[str, Any]:
    start = time.time()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"# cwd: {cwd.as_posix()}\n")
        log.write(f"# cmd: {' '.join(cmd)}\n\n")
        try:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        except FileNotFoundError as exc:
            log.write(f"Command not found: {exc}\n")
            duration = time.time() - start
            return {
                "return_code": 127,
                "duration_seconds": duration,
                "error": str(exc),
            }

        output_lines: List[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            output_lines.append(line)
        process.wait()
        duration = time.time() - start

    return {
        "return_code": process.returncode,
        "duration_seconds": duration,
    }


def execute_project(
    name: str,
    definition: Dict[str, Any],
    run_dir: Path,
    continue_on_error: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    project_root = WORKSPACE / definition.get("root", "")
    project_entry: Dict[str, Any] = {
        "project": name,
        "root": relative_path(project_root),
        "status": "pending",
        "steps": [],
    }

    if not project_root.exists():
        project_entry["status"] = "missing"
        project_entry["error"] = f"Project root not found: {project_root}"
        return project_entry

    project_env = {k: str(v) for k, v in (definition.get("env") or {}).items()}
    steps: List[Dict[str, Any]] = definition.get("steps") or []
    if not steps:
        project_entry["status"] = "skipped"
        project_entry["error"] = "No steps configured"
        return project_entry

    for step in steps:
        step_name = step.get("name", "unnamed-step")
        cmd = step.get("cmd")
        if not cmd:
            project_entry["steps"].append(
                {
                    "name": step_name,
                    "status": "skipped",
                    "reason": "No cmd configured",
                }
            )
            continue

        skip_reason = step_should_run(project_root, step)
        if skip_reason:
            project_entry["steps"].append(
                {
                    "name": step_name,
                    "status": "skipped",
                    "reason": skip_reason,
                }
            )
            continue

        cmd_list = [str(part) for part in cmd]
        cwd = project_root / step.get("cwd", ".")
        env = os.environ.copy()
        env.update(project_env)
        env.update({k: str(v) for k, v in (step.get("env") or {}).items()})

        if dry_run:
            project_entry["steps"].append(
                {
                    "name": step_name,
                    "status": "planned",
                    "command": cmd_list,
                    "log": None,
                }
            )
            continue

        log_file = run_dir / f"{name}__{step_name}.log"
        result = run_command(cmd_list, cwd, env, log_file)
        step_status = "passed" if result["return_code"] == 0 else "failed"
        project_entry["steps"].append(
            {
                "name": step_name,
                "status": step_status,
                "command": cmd_list,
                "return_code": result["return_code"],
                "duration_seconds": round(result["duration_seconds"], 2),
                "log": relative_path(log_file),
            }
        )

        if step_status == "failed" and not continue_on_error:
            project_entry["status"] = "failed"
            project_entry["error"] = f"Step '{step_name}' failed"
            break
    else:
        project_entry["status"] = "planned" if dry_run else "passed"

    if project_entry["status"] == "pending":
        failed = any(step_info["status"] == "failed" for step_info in project_entry["steps"])
        project_entry["status"] = "failed" if failed else ("planned" if dry_run else "passed")

    return project_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-project build matrix")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to build matrix config JSON")
    parser.add_argument("--project", action="append", dest="projects", help="Limit execution to specific project keys")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue remaining steps even if a step fails")
    parser.add_argument("--dry-run", action="store_true", help="Plan steps without executing commands")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    selected_projects = args.projects or list(config.keys())
    run_dir = ensure_report_dir()
    started_at = datetime.now(timezone.utc)

    summary: Dict[str, Any] = {
        "run_id": run_dir.name,
        "started_at": started_at.isoformat(),
        "config": relative_path(args.config.resolve()),
        "projects": [],
    }

    for project_name in selected_projects:
        definition = config.get(project_name)
        if not definition:
            summary["projects"].append(
                {
                    "project": project_name,
                    "status": "missing",
                    "error": "Project not defined in config",
                }
            )
            continue

        report = execute_project(project_name, definition, run_dir, args.continue_on_error, args.dry_run)
        summary["projects"].append(report)

    ended_at = datetime.now(timezone.utc)
    summary["ended_at"] = ended_at.isoformat()
    summary["duration_seconds"] = round((ended_at - started_at).total_seconds(), 2)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest_path = REPORT_ROOT / "latest_summary.json"
    latest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # any projects that are neither recovered nor skipped are failures
    failed_projects = [
        p
        for p in summary["projects"]
        if p.get("status") not in {"passed", "skipped", "planned"}
    ]

    print(f"Build matrix run stored in {relative_path(summary_path)}")
    print(f"Projects processed: {len(summary['projects'])}")
    if failed_projects:
        print(f"Failures: {[p['project'] for p in failed_projects]}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
