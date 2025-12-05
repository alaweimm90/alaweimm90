#!/usr/bin/env python3
"""Analyze which projects should merge based on similarity."""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
ORGS = ROOT / "organizations"

def analyze_project(proj: Path) -> dict:
    """Extract project metadata."""
    meta = {
        "name": proj.name,
        "lang": None,
        "type": None,
        "size": sum(1 for _ in proj.rglob("*.py")) + sum(1 for _ in proj.rglob("*.ts")),
        "has_tests": (proj / "tests").exists() or (proj / "test").exists(),
        "has_docs": (proj / "docs").exists(),
    }
    
    if (proj / "package.json").exists():
        meta["lang"] = "typescript"
        pkg = json.loads((proj / "package.json").read_text())
        deps = list(pkg.get("dependencies", {}).keys())
        if "react" in deps:
            meta["type"] = "react"
        elif "next" in deps:
            meta["type"] = "next"
        else:
            meta["type"] = "node"
    elif (proj / "pyproject.toml").exists() or (proj / "requirements.txt").exists():
        meta["lang"] = "python"
        if (proj / "src").exists():
            meta["type"] = "library"
        else:
            meta["type"] = "app"
    
    return meta

def suggest_merges(org: Path) -> list:
    """Suggest project merges based on similarity."""
    projects = [p for p in org.iterdir() if p.is_dir() and not p.name.startswith(".")]
    metadata = {p.name: analyze_project(p) for p in projects}
    
    # Group by language and type
    groups = defaultdict(list)
    for name, meta in metadata.items():
        if meta["lang"] and meta["type"]:
            key = f"{meta['lang']}-{meta['type']}"
            groups[key].append(name)
    
    suggestions = []
    for key, projs in groups.items():
        if len(projs) > 1:
            suggestions.append({
                "category": key,
                "projects": projs,
                "count": len(projs),
                "action": f"Merge {len(projs)} {key} projects"
            })
    
    return suggestions

if __name__ == "__main__":
    print("=== MERGE ANALYSIS ===\n")
    
    all_suggestions = {}
    for org in sorted(ORGS.iterdir()):
        if not org.is_dir() or org.name.startswith("."):
            continue
        
        suggestions = suggest_merges(org)
        if suggestions:
            all_suggestions[org.name] = suggestions
            print(f"[{org.name}]")
            for s in suggestions:
                print(f"  {s['action']}: {', '.join(s['projects'][:3])}{'...' if len(s['projects']) > 3 else ''}")
            print()
    
    # Save report
    report = ROOT / ".metaHub" / "reports" / "merge-analysis.json"
    report.write_text(json.dumps(all_suggestions, indent=2))
    print(f"Report: {report}")
