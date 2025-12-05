#!/usr/bin/env python3
"""Generate CI workflows for all projects using reusable workflow."""
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
ORGS = ROOT / "organizations"

TEMPLATE = """name: CI

on:
  push:
    branches: [main, master]
  pull_request:

jobs:
  ci:
    uses: ./.github/workflows/reusable-universal-ci.yml
    with:
      language: {language}
"""

def detect_language(proj: Path) -> str:
    """Detect project language."""
    if (proj / "package.json").exists():
        return "typescript"
    if (proj / "pyproject.toml").exists() or (proj / "requirements.txt").exists():
        return "python"
    return None

def generate_ci(proj: Path):
    """Generate CI workflow for project."""
    lang = detect_language(proj)
    if not lang:
        return False
    
    gh_dir = proj / ".github" / "workflows"
    gh_dir.mkdir(parents=True, exist_ok=True)
    
    ci_file = gh_dir / "ci.yml"
    if ci_file.exists():
        return False  # Don't overwrite existing
    
    ci_file.write_text(TEMPLATE.format(language=lang))
    return True

if __name__ == "__main__":
    count = 0
    for org in ORGS.iterdir():
        if not org.is_dir() or org.name.startswith("."):
            continue
        
        for proj in org.iterdir():
            if not proj.is_dir() or proj.name.startswith("."):
                continue
            
            if generate_ci(proj):
                print(f"CREATE {proj.relative_to(ORGS)}/.github/workflows/ci.yml")
                count += 1
    
    print(f"\nGenerated {count} CI workflows")
