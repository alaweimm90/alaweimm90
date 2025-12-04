#!/usr/bin/env python3
"""Refactor organization projects: consolidate duplicates, standardize configs, clean archives."""
from pathlib import Path
from typing import Dict, List

ORG_ROOT = Path(__file__).parent.parent.parent / "organizations"
TEMPLATES = Path(__file__).parent.parent / "templates"

# Standard files to consolidate (symlink to root)
CONSOLIDATE = {
    "LICENSE": "../../LICENSE",
    "CODE_OF_CONDUCT.md": "../../.github/CODE_OF_CONDUCT.md",
    ".gitignore": "../.gitignore",
    ".pre-commit-config.yaml": "../.pre-commit-config.yaml",
    ".yamllint.yaml": "../.yamllint.yaml",
}

# Files to remove from archives
ARCHIVE_CLEANUP = ["*CLEANUP*.md", "*REPORT*.md", "*SUMMARY*.md", "*.txt"]

def consolidate_files(org: Path):
    """Replace duplicate files with symlinks."""
    for file, target in CONSOLIDATE.items():
        for proj in org.rglob(file):
            if proj.is_file() and not proj.is_symlink():
                rel_target = Path("../" * (len(proj.relative_to(org).parts) - 1)) / target
                proj.unlink()
                proj.symlink_to(rel_target)
                print(f"LINK {proj.relative_to(org)} -> {target}")

def clean_archives(org: Path):
    """Remove redundant archive files."""
    for archive in org.rglob(".archive"):
        if archive.is_dir():
            for pattern in ARCHIVE_CLEANUP:
                for f in archive.glob(pattern):
                    if f.is_file():
                        f.unlink()
                        print(f"DEL {f.relative_to(org)}")

def standardize_configs(org: Path):
    """Ensure all projects have .meta/repo.yaml."""
    for proj in [p for p in org.iterdir() if p.is_dir() and not p.name.startswith(".")]:
        meta = proj / ".meta"
        meta.mkdir(exist_ok=True)
        repo_yaml = meta / "repo.yaml"
        if not repo_yaml.exists():
            repo_yaml.write_text(f"name: {proj.name}\ntype: project\nstatus: active\n")
            print(f"+ Created {repo_yaml.relative_to(org)}")

def analyze_merge_candidates(org: Path) -> Dict[str, List[str]]:
    """Identify projects that could be merged."""
    candidates = {}
    for proj in [p for p in org.iterdir() if p.is_dir() and not p.name.startswith(".")]:
        # Group by tech stack
        if (proj / "package.json").exists():
            candidates.setdefault("typescript", []).append(proj.name)
        elif (proj / "pyproject.toml").exists():
            candidates.setdefault("python", []).append(proj.name)
    return {k: v for k, v in candidates.items() if len(v) > 1}

if __name__ == "__main__":
    orgs = [d for d in ORG_ROOT.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    print("=== REFACTORING ORGANIZATIONS ===\n")
    for org in orgs:
        print(f"\n[{org.name}]")
        consolidate_files(org)
        clean_archives(org)
        standardize_configs(org)
    
    print("\n\n=== MERGE CANDIDATES ===")
    for org in orgs:
        candidates = analyze_merge_candidates(org)
        if candidates:
            print(f"\n{org.name}:")
            for tech, projs in candidates.items():
                print(f"  {tech}: {', '.join(projs)}")
