#!/usr/bin/env python3
"""Consolidate duplicate tools to global metaHub configs."""
from pathlib import Path
import shutil

ROOT = Path(__file__).parent.parent.parent
GLOBAL = ROOT / "tools"
ORGS = ROOT / "organizations"

# Global config templates
CONFIGS = {
    "eslint.config.js": GLOBAL / "config" / "eslint.config.js",
    ".prettierrc": ROOT / ".prettierrc",
    "vitest.config.ts": GLOBAL / "config" / "vitest.config.ts",
    "playwright.config.ts": GLOBAL / "config" / "playwright.config.ts",
}

def consolidate_org(org: Path):
    """Replace project configs with symlinks to global."""
    projects = [p for p in org.iterdir() if p.is_dir() and not p.name.startswith(".")]
    
    for proj in projects:
        for config_name, global_path in CONFIGS.items():
            if not global_path.exists():
                continue
            
            proj_config = proj / config_name
            if proj_config.exists() and not proj_config.is_symlink():
                # Backup original
                backup = proj / f".backup_{config_name}"
                if not backup.exists():
                    shutil.copy2(proj_config, backup)
                
                # Replace with symlink
                proj_config.unlink()
                rel_path = Path("../" * (len(proj.relative_to(org).parts) + 1)) / global_path.relative_to(ROOT)
                proj_config.symlink_to(rel_path)
                print(f"LINK {proj.name}/{config_name} -> global")

def create_global_configs():
    """Create missing global configs."""
    config_dir = GLOBAL / "config"
    config_dir.mkdir(exist_ok=True)
    
    # ESLint
    eslint = config_dir / "eslint.config.js"
    if not eslint.exists():
        eslint.write_text("""export default [
  { ignores: ['dist', 'node_modules', '.git'] },
  { files: ['**/*.{js,ts,tsx}'], rules: { 'no-console': 'warn' } }
];
""")
        print(f"CREATE {eslint}")
    
    # Vitest
    vitest = config_dir / "vitest.config.ts"
    if not vitest.exists():
        vitest.write_text("""import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: { globals: true, environment: 'node' }
});
""")
        print(f"CREATE {vitest}")

if __name__ == "__main__":
    print("=== CREATING GLOBAL CONFIGS ===")
    create_global_configs()
    
    print("\n=== CONSOLIDATING TOOLS ===")
    for org in sorted(ORGS.iterdir()):
        if org.is_dir() and not org.name.startswith("."):
            print(f"\n[{org.name}]")
            consolidate_org(org)
