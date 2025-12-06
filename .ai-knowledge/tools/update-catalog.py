#!/usr/bin/env python3
"""Auto-update catalog from filesystem."""

import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
AI_KNOWLEDGE = ROOT / ".ai-knowledge"

def scan_prompts():
    """Scan prompts directory and update catalog."""
    prompts = []
    prompts_dir = AI_KNOWLEDGE / "prompts"
    
    for md_file in prompts_dir.rglob("*.md"):
        if md_file.name.startswith('_'):
            continue
        
        content = md_file.read_text()
        
        # Extract metadata from YAML frontmatter
        name = md_file.stem.replace('-', ' ').title()
        category = md_file.parent.name
        
        prompts.append({
            "id": md_file.stem,
            "name": name,
            "category": category,
            "path": str(md_file.relative_to(AI_KNOWLEDGE)),
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        })
    
    catalog_path = AI_KNOWLEDGE / "catalog" / "prompts.json"
    catalog_path.write_text(json.dumps({"prompts": prompts}, indent=2))
    print(f"✅ Updated prompts catalog: {len(prompts)} prompts")

def scan_workflows():
    """Scan workflows directory and update catalog."""
    workflows = []
    workflows_dir = AI_KNOWLEDGE / "workflows"
    
    for py_file in workflows_dir.rglob("*.py"):
        if py_file.name.startswith('_'):
            continue
        
        name = py_file.stem.replace('-', ' ').title()
        category = py_file.parent.name
        
        workflows.append({
            "id": py_file.stem,
            "name": name,
            "category": category,
            "path": str(py_file.relative_to(AI_KNOWLEDGE)),
            "automated": True
        })
    
    catalog_path = AI_KNOWLEDGE / "catalog" / "workflows.json"
    catalog_path.write_text(json.dumps({"workflows": workflows}, indent=2))
    print(f"✅ Updated workflows catalog: {len(workflows)} workflows")

if __name__ == "__main__":
    scan_prompts()
    scan_workflows()
    print("\n✅ Catalog updated successfully")
