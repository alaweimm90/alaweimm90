#!/usr/bin/env python3
"""
Convert Jupyter notebooks to markdown documentation for import.
Extracts markdown cells and code cells with comments for documentation.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def extract_notebook_content(notebook_path: Path) -> str:
    """Extract content from Jupyter notebook for documentation."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    content_parts = []
    
    # Add header with metadata
    content_parts.append(f"# {nb.get('metadata', {}).get('title', 'Notebook Documentation')}")
    content_parts.append(f"\n*Converted from: {notebook_path.name}*")
    content_parts.append(f"*Date: {datetime.now().strftime('%Y-%m-%d')}*\n")
    
    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type', '')
        
        if cell_type == 'markdown':
            # Extract markdown content
            source = cell.get('source', [])
            if isinstance(source, list):
                content = ''.join(source)
            else:
                content = source
            content_parts.append(content)
            
        elif cell_type == 'code':
            # Extract code with comments for documentation
            source = cell.get('source', [])
            if isinstance(source, list):
                code_lines = source
            else:
                code_lines = source.split('\n')
            
            # Only include code that has documentation comments
            doc_lines = []
            in_doc_block = False
            
            for line in code_lines:
                line_stripped = line.strip()
                
                # Include comment lines
                if line_stripped.startswith('#'):
                    # Remove # and leading spaces, but keep structure
                    comment_content = line.lstrip('#').strip()
                    if comment_content:
                        doc_lines.append(f"**{comment_content}**")
                    in_doc_block = True
                elif in_doc_block and not line_stripped:
                    # Empty line after comments
                    doc_lines.append("")
                    in_doc_block = False
                elif in_doc_block and line_stripped:
                    # Code line after comments - include if it's short and descriptive
                    if len(line_stripped) < 60 and ('def ' in line or 'class ' in line or '=' in line):
                        doc_lines.append(f"`{line_stripped}`")
                    in_doc_block = False
            
            if doc_lines:
                content_parts.append("\n".join(doc_lines))
        
        # Add spacing between cells
        content_parts.append("\n")
    
    return "\n".join(content_parts)

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python convert_notebook.py <notebook1.ipynb> <notebook2.ipynb> ...")
        return 1
    
    for notebook_path in sys.argv[1:]:
        nb_path = Path(notebook_path)
        if not nb_path.exists() or not nb_path.suffix == '.ipynb':
            print(f"Skip: {notebook_path} (not a valid notebook)")
            continue
        
        try:
            content = extract_notebook_content(nb_path)
            output_name = nb_path.stem + '_documentation.md'
            output_path = nb_path.parent / output_name
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Converted: {nb_path} -> {output_path}")
        except Exception as e:
            print(f"Error converting {notebook_path}: {e}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())