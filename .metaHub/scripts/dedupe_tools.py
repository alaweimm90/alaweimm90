#!/usr/bin/env python3
"""Find duplicate tools across projects and suggest consolidation."""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

ROOT = Path(__file__).parent.parent.parent
GLOBAL_TOOLS = ROOT / "tools"
ORGS = ROOT / "organizations"

# Tool patterns to detect
PATTERNS = {
    "lint": ["eslint", "ruff", "pylint", "flake8", ".eslintrc"],
    "format": ["prettier", "black", ".prettierrc"],
    "test": ["pytest", "vitest", "jest", "playwright"],
    "build": ["vite", "webpack", "rollup", "esbuild"],
    "docker": ["Dockerfile", "docker-compose"],
    "ci": [".github/workflows"],
    "security": ["trivy", "bandit", "safety"],
    "docs": ["mkdocs", "sphinx", "typedoc"],
}

def scan_project(proj: Path) -> Dict[str, List[str]]:
    """Scan project for tool usage."""
    found = defaultdict(list)
    
    for category, patterns in PATTERNS.items():
        for pattern in patterns:
            matches = list(proj.rglob(f"*{pattern}*"))
            if matches:
                found[category].extend([str(m.relative_to(proj)) for m in matches[:3]])
    
    return dict(found)

def find_global_tools() -> Dict[str, Path]:
    """Find available global tools."""
    tools = {}
    if GLOBAL_TOOLS.exists():
        for tool in GLOBAL_TOOLS.rglob("*"):
            if tool.is_file() and not tool.name.startswith("."):
                tools[tool.stem] = tool
    return tools

def analyze_org(org: Path) -> Dict:
    """Analyze organization for tool duplication."""
    projects = [p for p in org.iterdir() if p.is_dir() and not p.name.startswith(".")]
    
    tool_usage = {}
    for proj in projects:
        tools = scan_project(proj)
        if tools:
            tool_usage[proj.name] = tools
    
    # Find duplicates
    category_counts = defaultdict(int)
    for tools in tool_usage.values():
        for cat in tools:
            category_counts[cat] += 1
    
    duplicates = {k: v for k, v in category_counts.items() if v > 1}
    
    return {
        "projects": len(projects),
        "tool_usage": tool_usage,
        "duplicates": duplicates,
        "consolidation_potential": sum(duplicates.values()) - len(duplicates)
    }

def generate_recommendations(org_name: str, analysis: Dict, global_tools: Dict) -> List[str]:
    """Generate consolidation recommendations."""
    recs = []
    
    for category, count in analysis["duplicates"].items():
        if count > 2:
            recs.append(f"CONSOLIDATE {category}: {count} projects -> 1 global config")
    
    # Check if global tools exist
    for proj, tools in analysis["tool_usage"].items():
        for cat in tools:
            if cat in ["security", "ci", "docs"]:
                recs.append(f"USE GLOBAL: {proj} can use metaHub/{cat}")
    
    return recs

if __name__ == "__main__":
    global_tools = find_global_tools()
    print(f"=== GLOBAL TOOLS ({len(global_tools)}) ===")
    for name in sorted(global_tools.keys())[:10]:
        print(f"  {name}")
    
    print("\n=== ORGANIZATION ANALYSIS ===\n")
    
    all_recs = []
    for org in sorted(ORGS.iterdir()):
        if not org.is_dir() or org.name.startswith("."):
            continue
        
        print(f"[{org.name}]")
        analysis = analyze_org(org)
        
        print(f"  Projects: {analysis['projects']}")
        print(f"  Duplicates: {len(analysis['duplicates'])}")
        print(f"  Savings: {analysis['consolidation_potential']} configs")
        
        recs = generate_recommendations(org.name, analysis, global_tools)
        all_recs.extend([(org.name, r) for r in recs])
        
        if analysis["duplicates"]:
            print("  Top duplicates:")
            for cat, count in sorted(analysis["duplicates"].items(), key=lambda x: -x[1])[:3]:
                print(f"    {cat}: {count}x")
        print()
    
    print("\n=== RECOMMENDATIONS ===")
    for org, rec in all_recs[:15]:
        print(f"{org}: {rec}")
    
    # Save detailed report
    report = ROOT / ".metaHub" / "reports" / "tool-deduplication.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps({
        "global_tools": list(global_tools.keys()),
        "recommendations": [{"org": o, "action": r} for o, r in all_recs]
    }, indent=2))
    print(f"\nReport: {report}")
