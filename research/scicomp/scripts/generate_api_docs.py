#!/usr/bin/env python3
"""
SciComp - API Documentation Generator
========================================================
Automatically generates comprehensive API documentation for all modules
in the SciComp, including cross-references, examples,
and Berkeley-styled formatting.
Author: UC Berkeley SciComp Team
Date: 2025
License: MIT
"""
import os
import sys
import ast
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import textwrap
# Add Python package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Python"))
# Berkeley colors for console output
class Colors:
    BERKELEY_BLUE = '\033[94m'
    CALIFORNIA_GOLD = '\033[93m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
class APIDocumentationGenerator:
    """Generates comprehensive API documentation."""
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.python_path = root_path / "Python"
        self.docs_path = root_path / "docs"
        self.api_docs_path = self.docs_path / "api"
        # Create docs directory if needed
        self.api_docs_path.mkdir(parents=True, exist_ok=True)
        # Berkeley styling
        self.berkeley_header = """
# 🐻 SciComp - API Reference
![Berkeley SciComp](https://img.shields.io/badge/SciComp-003262?style=flat-square&logo=university)
**University of California, Berkeley**
**Scientific Computing Excellence Since 1868**
---
"""
    def print_status(self, message: str):
        """Print status message with Berkeley styling."""
        print(f"{Colors.BERKELEY_BLUE}[API Docs]{Colors.RESET} {message}")
    def get_all_modules(self) -> List[Path]:
        """Get all Python modules in the framework."""
        modules = []
        for module_dir in self.python_path.iterdir():
            if module_dir.is_dir() and not module_dir.name.startswith('.'):
                # Look for core modules
                core_dir = module_dir / "core"
                if core_dir.exists():
                    for py_file in core_dir.glob("*.py"):
                        if py_file.name != "__init__.py":
                            modules.append(py_file)
                # Look for main module files
                for py_file in module_dir.glob("*.py"):
                    if py_file.name != "__init__.py":
                        modules.append(py_file)
        return modules
    def extract_module_info(self, module_path: Path) -> Dict[str, Any]:
        """Extract information from a Python module."""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source)
            module_info = {
                'name': module_path.stem,
                'path': module_path,
                'docstring': ast.get_docstring(tree),
                'classes': [],
                'functions': [],
                'constants': []
            }
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'methods': [],
                        'line_number': node.lineno
                    }
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            method_info = {
                                'name': method.name,
                                'docstring': ast.get_docstring(method),
                                'args': [arg.arg for arg in method.args.args],
                                'line_number': method.lineno
                            }
                            class_info['methods'].append(method_info)
                    module_info['classes'].append(class_info)
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    # Only top-level functions
                    func_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'args': [arg.arg for arg in node.args.args],
                        'line_number': node.lineno
                    }
                    module_info['functions'].append(func_info)
                elif isinstance(node, ast.Assign) and node.col_offset == 0:
                    # Constants (uppercase variables)
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            module_info['constants'].append({
                                'name': target.id,
                                'line_number': node.lineno
                            })
            return module_info
        except Exception as e:
            self.print_status(f"Error parsing {module_path}: {e}")
            return None
    def format_docstring(self, docstring: Optional[str]) -> str:
        """Format docstring for markdown."""
        if not docstring:
            return "*No documentation available.*"
        # Clean up the docstring
        lines = docstring.strip().split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
            elif cleaned_lines and cleaned_lines[-1]:  # Preserve paragraph breaks
                cleaned_lines.append('')
        return '\n'.join(cleaned_lines)
    def generate_module_doc(self, module_info: Dict[str, Any]) -> str:
        """Generate markdown documentation for a module."""
        if not module_info:
            return ""
        doc = []
        module_name = module_info['name']
        # Module header
        doc.append(f"# {module_name}")
        doc.append("")
        # Module path
        relative_path = module_info['path'].relative_to(self.root_path)
        doc.append(f"**Module:** `{relative_path}`")
        doc.append("")
        # Module docstring
        if module_info['docstring']:
            doc.append("## Overview")
            doc.append("")
            doc.append(self.format_docstring(module_info['docstring']))
            doc.append("")
        # Constants
        if module_info['constants']:
            doc.append("## Constants")
            doc.append("")
            for const in module_info['constants']:
                doc.append(f"- **`{const['name']}`**")
            doc.append("")
        # Functions
        if module_info['functions']:
            doc.append("## Functions")
            doc.append("")
            for func in module_info['functions']:
                doc.append(f"### `{func['name']}({', '.join(func['args'])})`")
                doc.append("")
                if func['docstring']:
                    doc.append(self.format_docstring(func['docstring']))
                else:
                    doc.append("*No documentation available.*")
                doc.append("")
                # Add source link
                line_num = func['line_number']
                doc.append(f"**Source:** [Line {line_num}]({relative_path}#L{line_num})")
                doc.append("")
        # Classes
        if module_info['classes']:
            doc.append("## Classes")
            doc.append("")
            for cls in module_info['classes']:
                doc.append(f"### `{cls['name']}`")
                doc.append("")
                if cls['docstring']:
                    doc.append(self.format_docstring(cls['docstring']))
                else:
                    doc.append("*No documentation available.*")
                doc.append("")
                # Methods
                if cls['methods']:
                    doc.append("#### Methods")
                    doc.append("")
                    for method in cls['methods']:
                        method_name = method['name']
                        args = ', '.join(method['args'])
                        doc.append(f"##### `{method_name}({args})`")
                        doc.append("")
                        if method['docstring']:
                            doc.append(self.format_docstring(method['docstring']))
                        else:
                            doc.append("*No documentation available.*")
                        doc.append("")
                        # Add source link
                        line_num = method['line_number']
                        doc.append(f"**Source:** [Line {line_num}]({relative_path}#L{line_num})")
                        doc.append("")
                # Add class source link
                line_num = cls['line_number']
                doc.append(f"**Class Source:** [Line {line_num}]({relative_path}#L{line_num})")
                doc.append("")
        return '\n'.join(doc)
    def generate_module_index(self, modules: List[Dict[str, Any]]) -> str:
        """Generate index of all modules."""
        doc = []
        doc.append(self.berkeley_header)
        doc.append("## Module Index")
        doc.append("")
        doc.append("Complete API reference for all modules in the SciComp.")
        doc.append("")
        # Group modules by category
        categories = {
            'Quantum Physics': [],
            'Thermal Transport': [],
            'Signal Processing': [],
            'Optimization': [],
            'Control Systems': [],
            'Machine Learning': [],
            'GPU Acceleration': [],
            'Utilities': [],
            'Other': []
        }
        for module in modules:
            if not module:
                continue
            module_name = module['name'].lower()
            path_str = str(module['path']).lower()
            if 'quantum' in path_str:
                categories['Quantum Physics'].append(module)
            elif 'thermal' in path_str or 'heat' in path_str:
                categories['Thermal Transport'].append(module)
            elif 'signal' in path_str or 'fourier' in path_str:
                categories['Signal Processing'].append(module)
            elif 'optimization' in path_str or 'optim' in path_str:
                categories['Optimization'].append(module)
            elif 'control' in path_str:
                categories['Control Systems'].append(module)
            elif 'ml' in path_str or 'neural' in path_str or 'machine' in path_str:
                categories['Machine Learning'].append(module)
            elif 'gpu' in path_str or 'cuda' in path_str:
                categories['GPU Acceleration'].append(module)
            elif 'utils' in path_str or 'util' in path_str:
                categories['Utilities'].append(module)
            else:
                categories['Other'].append(module)
        # Generate categorized index
        for category, modules_in_category in categories.items():
            if modules_in_category:
                doc.append(f"### {category}")
                doc.append("")
                for module in sorted(modules_in_category, key=lambda x: x['name']):
                    module_name = module['name']
                    relative_path = module['path'].relative_to(self.root_path)
                    # Create link to module documentation
                    doc_link = f"{module_name.lower()}.md"
                    doc.append(f"- **[{module_name}]({doc_link})** - `{relative_path}`")
                    if module['docstring']:
                        # Extract first line of docstring as description
                        first_line = module['docstring'].split('\n')[0].strip()
                        if first_line and not first_line.endswith('.'):
                            first_line += '.'
                        doc.append(f"  {first_line}")
                doc.append("")
        # Add quick links
        doc.append("## Quick Links")
        doc.append("")
        doc.append("- [Installation Guide](../docs/INSTALLATION_GUIDE.md)")
        doc.append("- [Contributing Guide](../CONTRIBUTING.md)")
        doc.append("- [Examples](../examples/)")
        doc.append("- [GitHub Repository](https://github.com/berkeley/scicomp)")
        doc.append("")
        # Berkeley footer
        doc.append("---")
        doc.append("")
        doc.append("**🐻💙💛 University of California, Berkeley 💙💛🐻**")
        doc.append("")
        doc.append("*Scientific Computing Excellence Since 1868*")
        doc.append("")
        return '\n'.join(doc)
    def generate_all_docs(self):
        """Generate documentation for all modules."""
        self.print_status("Starting API documentation generation...")
        # Get all modules
        module_paths = self.get_all_modules()
        self.print_status(f"Found {len(module_paths)} modules to document")
        # Extract module information
        modules = []
        for module_path in module_paths:
            self.print_status(f"Processing {module_path.name}...")
            module_info = self.extract_module_info(module_path)
            if module_info:
                modules.append(module_info)
        # Generate individual module docs
        documented_modules = 0
        for module_info in modules:
            if module_info:
                doc_content = self.generate_module_doc(module_info)
                if doc_content:
                    # Write module documentation
                    doc_filename = f"{module_info['name'].lower()}.md"
                    doc_path = self.api_docs_path / doc_filename
                    with open(doc_path, 'w', encoding='utf-8') as f:
                        f.write(doc_content)
                    documented_modules += 1
                    self.print_status(f"Generated documentation for {module_info['name']}")
        # Generate module index
        index_content = self.generate_module_index(modules)
        index_path = self.api_docs_path / "README.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        # Also create in main docs directory
        main_api_path = self.docs_path / "API_REFERENCE.md"
        with open(main_api_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        self.print_status(f"Generated API index with {documented_modules} modules")
        # Generate summary
        print(f"\n{Colors.BERKELEY_BLUE}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}API Documentation Generation Complete{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{'='*60}{Colors.RESET}")
        print(f"Documented modules: {documented_modules}")
        print(f"Documentation location: {self.api_docs_path}")
        print(f"Main API reference: {main_api_path}")
        print(f"{Colors.CALIFORNIA_GOLD}🐻 Berkeley SciComp API docs ready! 🐻{Colors.RESET}")
def main():
    """Main function."""
    print(f"{Colors.BERKELEY_BLUE}🐻 SciComp - API Documentation Generator 🐻{Colors.RESET}")
    print(f"{Colors.CALIFORNIA_GOLD}University of California, Berkeley{Colors.RESET}")
    print()
    # Get root path
    root_path = Path(__file__).parent.parent
    # Create generator and run
    generator = APIDocumentationGenerator(root_path)
    generator.generate_all_docs()
if __name__ == "__main__":
    main()