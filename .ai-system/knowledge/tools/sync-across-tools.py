#!/usr/bin/env python3
"""Sync AI knowledge across all tools."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI_KNOWLEDGE = ROOT / ".ai-knowledge"

# IDE-specific directories (add your IDE here)
IDE_DIRS = [
    Path.home() / ".aws" / "amazonq" / "prompts",  # Amazon Q
    ROOT / ".config" / "claude" / "prompts",        # Claude
    ROOT / ".windsurf" / "prompts",                 # Windsurf
    ROOT / ".cline" / "prompts",                    # Cline
    ROOT / ".cursor" / "prompts",                   # Cursor (if needed)
    # Add more IDEs as needed
]

def sync():
    """Sync prompts to all IDE directories."""
    synced_count = 0
    
    for ide_dir in IDE_DIRS:
        if not ide_dir.parent.exists():
            continue  # Skip if IDE not installed
        
        ide_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all prompts (not just superprompts)
        for category in ["superprompts", "code-review", "refactoring"]:
            src = AI_KNOWLEDGE / "prompts" / category
            if src.exists():
                for prompt in src.glob("*.md"):
                    shutil.copy2(prompt, ide_dir / prompt.name)
                    synced_count += 1
    
    print(f"[OK] Synced {synced_count} prompts to {len([d for d in IDE_DIRS if d.parent.exists()])} IDEs")

if __name__ == "__main__":
    sync()
