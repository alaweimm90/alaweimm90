import json
from pathlib import Path
def check_exists(root: Path, names: list[str]) -> dict[str, bool]:
    return {n: (root / n).exists() for n in names}
def summarize(root: Path) -> dict:
    required = ['README.md','LICENSE','CODE_OF_CONDUCT.md','SECURITY.md','CONTRIBUTING.md','.gitignore','pyproject.toml']
    files = check_exists(root, required)
    passed = all(files.values())
    rd = (root / 'README.md').read_text(encoding='utf-8') if (root / 'README.md').exists() else ''
    sections = ['Quick Start','Architecture','Testing']
    sections_present = {s: (s in rd) for s in sections}
    claude = (root / 'CLAUDE.md').read_text(encoding='utf-8') if (root / 'CLAUDE.md').exists() else ''
    ssot_ok = '## SSOT References' in claude
    return {'passed': passed and all(sections_present.values()) and ssot_ok, 'files': files, 'readme_sections': sections_present, 'ssot_references': ssot_ok}
def main() -> None:
    root = Path(__file__).resolve().parent.parent
    report_dir = root / 'reports'
    report_dir.mkdir(exist_ok=True)
    report = summarize(root)
    (report_dir / 'compliance_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
if __name__ == '__main__':
    main()
