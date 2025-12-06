#!/usr/bin/env python3
"""Consolidate ALL prompts, workflows, and rules from across the repository."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI_KNOWLEDGE = ROOT / ".ai-knowledge"

# Source locations
SOURCES = {
    "prompts": [
        ROOT / "automation" / "prompts" / "project",
        ROOT / "automation" / "prompts" / "system",
        ROOT / "automation" / "prompts" / "tasks",
        ROOT / ".config" / "ai" / "prompts",
        ROOT / ".config" / "ai" / "superprompts",
        ROOT / ".archive" / "organizations" / "AlaweinOS" / ".archive" / "prompts",
    ],
    "workflows": [
        ROOT / ".config" / "ai" / "workflows" / "templates",
        ROOT / "automation" / "workflows",
    ],
    "docs": [
        ROOT / "docs" / "AI-AUTO-APPROVE-GUIDE.md",
        ROOT / "docs" / "AI-TOOL-PROFILES.md",
        ROOT / "docs" / "AI-TOOLS-ORCHESTRATION.md",
        ROOT / "docs" / "DEVOPS-AGENTS.md",
    ]
}

def copy_prompts():
    """Copy all prompts to .ai-knowledge/prompts/"""
    count = 0
    
    for source_dir in SOURCES["prompts"]:
        if not source_dir.exists():
            continue
        
        for md_file in source_dir.rglob("*.md"):
            if md_file.name.startswith('_'):
                continue
            
            # Determine category
            if "SUPERPROMPT" in md_file.name.upper():
                dest_dir = AI_KNOWLEDGE / "prompts" / "superprompts"
            elif "review" in md_file.name.lower():
                dest_dir = AI_KNOWLEDGE / "prompts" / "code-review"
            elif "refactor" in md_file.name.lower():
                dest_dir = AI_KNOWLEDGE / "prompts" / "refactoring"
            elif any(x in md_file.name.lower() for x in ["architect", "design", "system"]):
                dest_dir = AI_KNOWLEDGE / "prompts" / "architecture"
            elif any(x in md_file.name.lower() for x in ["debug", "fix", "error"]):
                dest_dir = AI_KNOWLEDGE / "prompts" / "debugging"
            else:
                dest_dir = AI_KNOWLEDGE / "prompts" / "superprompts"
            
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / md_file.name
            
            if not dest_file.exists():
                shutil.copy2(md_file, dest_file)
                count += 1
                print(f"  + {md_file.name}")
    
    # Copy YAML superprompts
    yaml_dir = ROOT / ".config" / "ai" / "superprompts"
    if yaml_dir.exists():
        for yaml_file in yaml_dir.glob("*.yaml"):
            dest_file = AI_KNOWLEDGE / "prompts" / "superprompts" / yaml_file.name
            if not dest_file.exists():
                shutil.copy2(yaml_file, dest_file)
                count += 1
                print(f"  + {yaml_file.name}")
    
    print(f"\n[OK] Copied {count} prompts")

def copy_workflows():
    """Copy workflow templates."""
    count = 0
    
    for source_dir in SOURCES["workflows"]:
        if not source_dir.exists():
            continue
        
        for yaml_file in source_dir.rglob("*.yaml"):
            dest_dir = AI_KNOWLEDGE / "workflows" / "development"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / yaml_file.name
            
            if not dest_file.exists():
                shutil.copy2(yaml_file, dest_file)
                count += 1
                print(f"  + {yaml_file.name}")
    
    print(f"\n[OK] Copied {count} workflows")

def copy_docs():
    """Copy important AI documentation."""
    count = 0
    
    for doc_file in SOURCES["docs"]:
        if not doc_file.exists():
            continue
        
        dest_file = AI_KNOWLEDGE / "docs" / doc_file.name
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not dest_file.exists():
            shutil.copy2(doc_file, dest_file)
            count += 1
            print(f"  + {doc_file.name}")
    
    print(f"\n[OK] Copied {count} docs")

if __name__ == "__main__":
    print("Consolidating all AI knowledge...\n")
    print("=" * 60)
    print("\nPrompts:")
    copy_prompts()
    print("\nWorkflows:")
    copy_workflows()
    print("\nDocumentation:")
    copy_docs()
    print("\n" + "=" * 60)
    print("\n[OK] Consolidation complete!")
    print("\nNext: Run update-catalog.py to index everything")
