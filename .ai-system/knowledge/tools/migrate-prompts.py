#!/usr/bin/env python3
"""Extract and migrate prompts from chat exports and markdown files."""

import json
import re
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[2]
AI_KNOWLEDGE = ROOT / ".ai-knowledge"
ARCHIVE = ROOT / ".archive" / "chat-exports"

def extract_from_markdown(file_path: Path) -> List[Dict]:
    """Extract prompt-like content from markdown files."""
    content = file_path.read_text(encoding='utf-8', errors='ignore')
    prompts = []
    
    # Look for code blocks or structured sections
    sections = re.split(r'\n#{1,3}\s+', content)
    for section in sections:
        if any(keyword in section.lower() for keyword in ['prompt', 'instruction', 'workflow', 'rule']):
            prompts.append({
                'source': str(file_path.relative_to(ROOT)),
                'content': section[:500],
                'type': 'extracted'
            })
    
    return prompts

def scan_existing_docs() -> List[Dict]:
    """Scan existing documentation for reusable prompts."""
    found = []
    
    # Scan docs directory
    docs_dir = ROOT / "docs"
    if docs_dir.exists():
        for md_file in docs_dir.rglob("*.md"):
            if any(keyword in md_file.name.lower() for keyword in ['prompt', 'guide', 'workflow']):
                found.extend(extract_from_markdown(md_file))
    
    # Scan archive
    if ARCHIVE.exists():
        for md_file in ARCHIVE.rglob("*.md"):
            found.extend(extract_from_markdown(md_file))
    
    return found

def generate_migration_report():
    """Generate report of found prompts."""
    prompts = scan_existing_docs()
    
    report = f"""# Prompt Migration Report

Found {len(prompts)} potential prompts to migrate.

## Sources
"""
    
    sources = {}
    for p in prompts:
        src = p['source']
        sources[src] = sources.get(src, 0) + 1
    
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        report += f"\n- `{src}`: {count} prompts"
    
    report += "\n\n## Next Steps\n\n"
    report += "1. Review each source file\n"
    report += "2. Extract valuable prompts\n"
    report += "3. Use template: `.ai-knowledge/templates/new-superprompt.md`\n"
    report += "4. Save to appropriate category\n"
    
    output_path = AI_KNOWLEDGE / "migration-report.md"
    output_path.write_text(report)
    print(f"[OK] Report saved to: {output_path}")
    
    return prompts

if __name__ == "__main__":
    generate_migration_report()
