#!/usr/bin/env python3
"""Sync AI knowledge across all tools."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI_KNOWLEDGE = ROOT / ".ai-knowledge"

# Tool directories
AMAZON_Q = Path.home() / ".aws" / "amazonq" / "prompts"
CLAUDE = ROOT / ".config" / "claude" / "prompts"
WINDSURF = ROOT / ".windsurf" / "prompts"
CLINE = ROOT / ".cline" / "prompts"

def sync():
    """Sync prompts to all tools."""
    for tool_dir in [AMAZON_Q, CLAUDE, WINDSURF, CLINE]:
        tool_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy superprompts
        src = AI_KNOWLEDGE / "prompts" / "superprompts"
        if src.exists():
            for prompt in src.glob("*.md"):
                shutil.copy2(prompt, tool_dir / prompt.name)
    
    print("✅ Synced across all tools")

if __name__ == "__main__":
    sync()
