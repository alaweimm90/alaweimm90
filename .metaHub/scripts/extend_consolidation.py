#!/usr/bin/env python3
"""Extend consolidation to all remaining duplicate configs."""
from pathlib import Path
import shutil

ROOT = Path(__file__).parent.parent.parent
GLOBAL = ROOT / "tools" / "config"
ORGS = ROOT / "organizations"

# Extended config mappings
CONFIGS = {
    "eslint.config.js": GLOBAL / "eslint.config.js",
    ".prettierrc": ROOT / ".prettierrc",
    "vitest.config.ts": GLOBAL / "vitest.config.ts",
    "playwright.config.ts": GLOBAL / "playwright.config.ts",
    "ruff.toml": GLOBAL / "ruff.toml",
    "pyproject.toml": None,  # Skip - too project-specific
}

def consolidate_all():
    """Consolidate all eligible configs across all orgs."""
    stats = {"linked": 0, "skipped": 0, "backed_up": 0}
    
    for org in ORGS.iterdir():
        if not org.is_dir() or org.name.startswith("."):
            continue
        
        for proj in org.iterdir():
            if not proj.is_dir() or proj.name.startswith("."):
                continue
            
            for config_name, global_path in CONFIGS.items():
                if global_path is None or not global_path.exists():
                    continue
                
                proj_config = proj / config_name
                if proj_config.exists() and not proj_config.is_symlink():
                    # Backup
                    backup = proj / f".backup_{config_name}"
                    if not backup.exists():
                        shutil.copy2(proj_config, backup)
                        stats["backed_up"] += 1
                    
                    # Symlink
                    proj_config.unlink()
                    depth = len(proj.relative_to(ROOT).parts)
                    rel_path = Path("../" * depth) / global_path.relative_to(ROOT)
                    proj_config.symlink_to(rel_path)
                    stats["linked"] += 1
                    print(f"LINK {proj.relative_to(ORGS)}/{config_name}")
    
    return stats

if __name__ == "__main__":
    print("=== EXTENDING CONSOLIDATION ===\n")
    stats = consolidate_all()
    print("\n=== SUMMARY ===")
    print(f"Linked: {stats['linked']}")
    print(f"Backed up: {stats['backed_up']}")
    print(f"Skipped: {stats['skipped']}")
