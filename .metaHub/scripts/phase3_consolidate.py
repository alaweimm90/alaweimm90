#!/usr/bin/env python3
"""Phase 3: Consolidate CI, Docker, and Build configs."""
from pathlib import Path
import shutil

ROOT = Path(__file__).parent.parent.parent
GLOBAL = ROOT / "tools"
ORGS = ROOT / "organizations"

CONFIGS = {
    "Dockerfile": None,  # Skip - too project-specific
    "docker-compose.yaml": None,  # Skip
    "vite.config.ts": GLOBAL / "config" / "vite.config.ts",
}

def consolidate():
    stats = {"linked": 0, "backed_up": 0}
    
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
                    backup = proj / f".backup_{config_name}"
                    if not backup.exists():
                        shutil.copy2(proj_config, backup)
                        stats["backed_up"] += 1
                    
                    proj_config.unlink()
                    depth = len(proj.relative_to(ROOT).parts)
                    rel_path = Path("../" * depth) / global_path.relative_to(ROOT)
                    proj_config.symlink_to(rel_path)
                    stats["linked"] += 1
                    print(f"LINK {proj.relative_to(ORGS)}/{config_name}")
    
    return stats

if __name__ == "__main__":
    print("=== PHASE 3: CI/DOCKER/BUILD ===\n")
    stats = consolidate()
    print(f"\nLinked: {stats['linked']}")
    print(f"Backed up: {stats['backed_up']}")
