import argparse
import json
import sys
from pathlib import Path

def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_text(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    if not path.exists():
        path.write_text(content, encoding="utf-8")

def append_if_missing(path: Path, marker: str, block: str) -> bool:
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    if marker in content:
        return True
    path.write_text(content.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
    return True

def ensure_governance_config(repo_root: Path) -> bool:
    target = repo_root / "governance" / "master-config.yaml"
    content = (
        "organization_framework:\n"
        "  organization_types:\n"
        "    research:\n"
        "      prefix: 'Librex'\n"
        "      governance_level: 'research'\n"
        "      required_files:\n"
        "        - 'README.md'\n"
        "        - 'LICENSE'\n"
        "        - 'CODE_OF_CONDUCT.md'\n"
        "        - 'SECURITY.md'\n"
        "        - 'CONTRIBUTING.md'\n"
        "        - '.gitignore'\n"
        "        - 'pyproject.toml'\n"
        "compliance_rules:\n"
        "  research:\n"
        "    - 'reproducibility_standards'\n"
        "    - 'benchmark_validation'\n"
        "    - 'method_documentation'\n"
        "    - 'test_coverage_95plus'\n"
    )
    write_text(target, content)
    return True

def ensure_compliance_script(repo_root: Path) -> bool:
    target = repo_root / "scripts" / "compliance_checker.py"
    content = (
        "import json\n"
        "from pathlib import Path\n"
        "def check_exists(root: Path, names: list[str]) -> dict[str, bool]:\n"
        "    return {n: (root / n).exists() for n in names}\n"
        "def summarize(root: Path) -> dict:\n"
        "    required = ['README.md','LICENSE','CODE_OF_CONDUCT.md','SECURITY.md','CONTRIBUTING.md','.gitignore','pyproject.toml']\n"
        "    files = check_exists(root, required)\n"
        "    passed = all(files.values())\n"
        "    return {'passed': passed, 'files': files}\n"
        "def main() -> None:\n"
        "    root = Path(__file__).resolve().parents[2]\n"
        "    report_dir = root / 'reports'\n"
        "    report_dir.mkdir(exist_ok=True)\n"
        "    report = summarize(root)\n"
        "    (report_dir / 'compliance_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')\n"
        "    print(json.dumps(report, indent=2))\n"
        "    raise SystemExit(0 if report['passed'] else 1)\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    write_text(target, content)
    return True

def deploy_agents_and_workflows(repo_root: Path) -> bool:
    try:
        ai_dir = repo_root / "ai"
        ensure_directory(ai_dir)
        src_agents = Path('.meta') / 'agents' / 'AGENT_REGISTRY.yaml'
        src_flows = Path('.meta') / 'workflows' / 'WORKFLOWS.yaml'
        if src_agents.exists():
            (ai_dir / 'AGENT_REGISTRY.yaml').write_text(src_agents.read_text(encoding='utf-8'), encoding='utf-8')
        if src_flows.exists():
            (ai_dir / 'WORKFLOWS.yaml').write_text(src_flows.read_text(encoding='utf-8'), encoding='utf-8')
        return True
    except Exception:
        return False

def ensure_ssot_references(repo_root: Path) -> bool:
    target = repo_root / "CLAUDE.md"
    marker = "## SSOT References"
    block = (
        "## SSOT References\n\n"
        "- `ORGANIZATIONS/alaweimm90-tools/agis/docs/standards/INDEX.md` — master SSOT index\n"
        "- `ORGANIZATIONS/alaweimm90-tools/agis/docs/standards/5-TOOLS/ide-integration.md` — IDE integration guide\n"
        "- `ORGANIZATIONS/alaweimm90-tools/agis/docs/standards/2-PROMPTS/PROMPT_REGISTRY.md` — prompt registry\n"
    )
    return append_if_missing(target, marker, block)

def generate_style_report(repo_root: Path) -> bool:
    try:
        sys.path.insert(0, str(Path('.meta').resolve() / 'styles'))
        from style_enforcer import StyleEnforcer  # type: ignore
        enforcer = StyleEnforcer()
        report = enforcer.generate_style_report(str(repo_root))
        report_dir = repo_root / "reports"
        ensure_directory(report_dir)
        (report_dir / "style_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False

def apply(target_path: str) -> dict:
    root = Path(target_path).resolve()
    results = {
        "governance": ensure_governance_config(root),
        "compliance": ensure_compliance_script(root),
        "agents_workflows": deploy_agents_and_workflows(root),
        "ssot": ensure_ssot_references(root),
        "style_report": generate_style_report(root)
    }
    try:
        workflows_root = Path('.github') / 'workflows'
        ensure_directory(workflows_root)
        wf = workflows_root / 'Librex-meta-compliance.yml'
        wf_content = (
            "name: Librex Meta Compliance\n"
            "on:\n"
            "  push:\n"
            "  pull_request:\n"
            "jobs:\n"
            "  compliance:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-python@v5\n"
            "        with:\n"
            "          python-version: '3.11'\n"
            "      - name: Run compliance checker\n"
            "        run: |
python ORGANIZATIONS/AlaweinOS/Librex/scripts/compliance_checker.py\n"
            "      - name: Upload compliance report\n"
            "        uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: Librex-compliance\n"
            "          path: ORGANIZATIONS/AlaweinOS/Librex/reports/compliance_report.json\n"
            "  style:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-python@v5\n"
            "        with:\n"
            "          python-version: '3.11'\n"
            "      - name: Generate style report\n"
            "        run: |
python -c \"import sys,json; sys.path.append('.meta/styles'); from style_enforcer import StyleEnforcer; r=StyleEnforcer().generate_style_report('ORGANIZATIONS/AlaweinOS/Librex'); print(json.dumps(r,indent=2)); open('ORGANIZATIONS/AlaweinOS/Librex/reports/style_report_ci.json','w').write(json.dumps(r))\"\n"
            "      - name: Fail if compliance below threshold\n"
            "        run: |
python - <<'PY'\nimport json, sys\nwith open('ORGANIZATIONS/AlaweinOS/Librex/reports/style_report_ci.json') as f:\n    r=json.load(f)\nrate=r.get('compliance_rate',0)\nprint(f'Compliance rate: {rate}')\nsys.exit(0 if rate>=90 else 1)\nPY\n"
            "      - name: Upload style report\n"
            "        uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: Librex-style\n"
            "          path: ORGANIZATIONS/AlaweinOS/Librex/reports/style_report_ci.json\n"
        )
        write_text(wf, wf_content)
        results["workflow"] = True
    except Exception:
        results["workflow"] = False
    summary_dir = root / "reports"
    ensure_directory(summary_dir)
    (summary_dir / "solo_coder_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["apply"], nargs="?", default="apply")
    parser.add_argument("--target", dest="target", default=str(Path("ORGANIZATIONS") / "AlaweinOS" / "Librex"))
    args = parser.parse_args()
    if args.command == "apply":
        res = apply(args.target)
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
