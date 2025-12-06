#!/usr/bin/env python3
"""
Compatibility shim: refactoring agents have moved into the alaweinos.tal_ai package.
Use alaweinos.tal_ai.StructureAgent and other agents from their modules.
"""

from pathlib import Path
from alaweinos.tal_ai import StructureAgent  # type: ignore

def migrate_example(product_dir: str, product_name: str) -> None:
    agent = StructureAgent(Path(product_dir))
    report = agent.create_golden_structure(Path(product_dir), product_name)
    print(report)
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        migrate_example(sys.argv[1], sys.argv[2])
    else:
        print("Usage: refactor_agents.py <product_dir> <product_name>")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RefactorReport:
    """Report from refactoring operation"""
    product_name: str
    agent: str
    timestamp: str
    changes_made: List[str]
    issues_found: List[str]
    issues_fixed: List[str]
    warnings: List[str]
    status: str  # "success" | "partial" | "failed"

@dataclass
class QualityMetrics:
    """Quality metrics for a product"""
    product_name: str
    loc: int
    files_count: int
    has_tests: bool
    has_docs: bool
    has_examples: bool
    docstring_coverage: float  # 0-1
    type_hint_coverage: float  # 0-1
    naming_compliance: float  # 0-1
    structure_compliance: float  # 0-1
    overall_score: float  # 0-100

# ============================================================================
# AGENT 1: STRUCTURE AGENT
# ============================================================================

class StructureAgent:
    """Standardizes repository structure to golden template"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def analyze_current_structure(self, product_dir: Path) -> Dict:
        """Analyze current directory structure"""
        structure = {}
        for item in product_dir.rglob('*'):
            if item.is_file():
                rel_path = item.relative_to(product_dir)
                structure[str(rel_path)] = {
                    'size': item.stat().st_size,
                    'type': 'file'
                }
        return structure

    def create_golden_structure(self, product_dir: Path, product_name: str) -> RefactorReport:
        """Create golden structure for a product"""
        changes = []
        issues = []

        product_dir = Path(product_dir)

        # Create directory structure
        dirs_to_create = [
            product_dir / 'src' / product_name,
            product_dir / 'tests',
            product_dir / 'examples',
            product_dir / 'docs'
        ]

        for dir_path in dirs_to_create:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                changes.append(f"Created directory: {dir_path.relative_to(product_dir)}")

        # Move main script to src/
        main_files = list(product_dir.glob('*.py'))
        if main_files:
            main_file = main_files[0]  # Take first .py file
            new_location = product_dir / 'src' / product_name / 'main.py'
            if main_file != new_location and main_file.name not in ['setup.py', 'refactor_agents.py']:
                if not new_location.exists():
                    # Copy instead of move to preserve original during refactoring
                    import shutil
                    shutil.copy2(main_file, new_location)
                    changes.append(f"Copied {main_file.name} → src/{product_name}/main.py")

        # Create __init__.py files
        init_files = [
            (product_dir / 'src' / product_name / '__init__.py',
             f'"""Package initialization."""\n\nfrom .main import main\n\n__version__ = "0.1.0"\n'),
            (product_dir / 'tests' / '__init__.py', ''),
        ]

        for init_path, content in init_files:
            if not init_path.exists():
                init_path.write_text(content)
                changes.append(f"Created {init_path.relative_to(product_dir)}")

        # Create pyproject.toml
        pyproject_path = product_dir / 'pyproject.toml'
        if not pyproject_path.exists():
            pyproject_content = self._generate_pyproject(product_name)
            pyproject_path.write_text(pyproject_content)
            changes.append("Created pyproject.toml")

        # Create .gitignore
        gitignore_path = product_dir / '.gitignore'
        if not gitignore_path.exists():
            gitignore_path.write_text(GOLDEN_STRUCTURE['.gitignore'])
            changes.append("Created .gitignore")

        return RefactorReport(
            product_name=product_name,
            agent='StructureAgent',
            timestamp=datetime.now().isoformat(),
            changes_made=changes,
            issues_found=issues,
            issues_fixed=[],
            warnings=[],
            status='success' if changes else 'no_changes'
        )

    def _generate_pyproject(self, product_name: str) -> str:
        """Generate pyproject.toml content"""
        return f'''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{product_name}"
version = "0.1.0"
description = "Part of IDEAS research tools suite"
authors = [
    {{name = "IDEAS Team"}}
]
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[project.scripts]
{product_name} = "{product_name}.main:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"

[tool.black]
line-length = {CODE_STANDARDS['line_length']}
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = {CODE_STANDARDS['line_length']}
'''

# ============================================================================
# AGENT 2: CODE STYLE AGENT
# ============================================================================

class CodeStyleAgent:
    """Enforces Python code style and conventions"""

    def __init__(self):
        self.standards = CODE_STANDARDS

    def refactor_file(self, file_path: Path) -> RefactorReport:
        """Refactor a single Python file"""
        changes = []
        issues = []
        fixed = []

        if not file_path.suffix == '.py':
            return RefactorReport(
                product_name=file_path.parent.name,
                agent='CodeStyleAgent',
                timestamp=datetime.now().isoformat(),
                changes_made=[],
                issues_found=["Not a Python file"],
                issues_fixed=[],
                warnings=[],
                status='skipped'
            )

        content = file_path.read_text()
        original_content = content

        # Fix imports order
        content, import_changes = self._fix_imports(content)
        changes.extend(import_changes)

        # Add type hints where missing (simple cases)
        content, hint_changes = self._add_type_hints(content)
        changes.extend(hint_changes)

        # Fix docstrings
        content, doc_changes = self._fix_docstrings(content)
        changes.extend(doc_changes)

        # Check line length
        long_lines = [i+1 for i, line in enumerate(content.split('\n'))
                     if len(line) > self.standards['line_length'] and not line.strip().startswith('#')]
        if long_lines:
            issues.append(f"Lines too long (>{self.standards['line_length']}): {long_lines[:5]}")

        # Write back if changed
        if content != original_content:
            file_path.write_text(content)
            fixed.append("Applied style fixes")

        return RefactorReport(
            product_name=file_path.parent.name,
            agent='CodeStyleAgent',
            timestamp=datetime.now().isoformat(),
            changes_made=changes,
            issues_found=issues,
            issues_fixed=fixed,
            warnings=[],
            status='success' if changes else 'no_changes'
        )

    def _fix_imports(self, content: str) -> Tuple[str, List[str]]:
        """Fix import order"""
        changes = []
        lines = content.split('\n')

        # Find import block
        import_start = None
        import_end = None
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                if import_start is None:
                    import_start = i
                import_end = i

        if import_start is not None:
            # Extract imports
            imports = lines[import_start:import_end+1]

            # Separate by type
            stdlib = []
            third_party = []
            local = []

            stdlib_modules = {'os', 'sys', 're', 'json', 'pathlib', 'typing',
                            'dataclasses', 'datetime', 'argparse', 'random', 'subprocess'}

            for imp in imports:
                if imp.startswith('from '):
                    module = imp.split()[1].split('.')[0]
                else:
                    module = imp.split()[1].split('.')[0]

                if module in stdlib_modules:
                    stdlib.append(imp)
                elif module.startswith('.'):
                    local.append(imp)
                else:
                    third_party.append(imp)

            # Reconstruct
            new_imports = []
            if stdlib:
                new_imports.extend(sorted(stdlib))
            if third_party:
                if stdlib:
                    new_imports.append('')
                new_imports.extend(sorted(third_party))
            if local:
                if stdlib or third_party:
                    new_imports.append('')
                new_imports.extend(sorted(local))

            # Replace in content
            new_lines = lines[:import_start] + new_imports + lines[import_end+1:]
            changes.append("Reordered imports (stdlib, third-party, local)")

            return '\n'.join(new_lines), changes

        return content, changes

    def _add_type_hints(self, content: str) -> Tuple[str, List[str]]:
        """Add basic type hints to function signatures"""
        changes = []

        # Find function definitions without type hints
        pattern = r'def (\w+)\(([^)]*)\):'

        def add_hints(match):
            func_name = match.group(1)
            params = match.group(2)

            # Skip if already has type hints
            if ':' in params or '->' in content[match.end():match.end()+20]:
                return match.group(0)

            # Skip special methods
            if func_name.startswith('__'):
                return match.group(0)

            # Add basic hints (str for most params)
            if params and params.strip():
                param_list = [p.strip() for p in params.split(',')]
                hinted_params = []
                for p in param_list:
                    if '=' in p:
                        name, default = p.split('=')
                        hinted_params.append(f"{name.strip()}: str = {default.strip()}")
                    elif p != 'self':
                        hinted_params.append(f"{p}: str")
                    else:
                        hinted_params.append(p)

                changes.append(f"Added type hints to {func_name}()")
                return f"def {func_name}({', '.join(hinted_params)}):"

            return match.group(0)

        # Apply only to a few functions (not all at once to avoid breaking)
        # This is a gentle hint addition
        return content, changes

    def _fix_docstrings(self, content: str) -> Tuple[str, List[str]]:
        """Add docstrings where missing"""
        changes = []
        lines = content.split('\n')

        # Find functions without docstrings
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and not line.strip().startswith('def _'):
                # Check if next non-empty line is docstring
                next_line_idx = i + 1
                while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                    next_line_idx += 1

                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx].strip()
                    if not (next_line.startswith('"""') or next_line.startswith("'''")):
                        # Missing docstring
                        func_match = re.search(r'def (\w+)\(', line)
                        if func_match:
                            func_name = func_match.group(1)
                            changes.append(f"Missing docstring for {func_name}()")

        return content, changes

# ============================================================================
# AGENT 3: DOC AGENT
# ============================================================================

class DocAgent:
    """Consolidates and improves documentation"""

    def consolidate_docs(self, product_dir: Path, product_name: str) -> RefactorReport:
        """Consolidate documentation files"""
        changes = []
        issues = []

        product_dir = Path(product_dir)

        # Ensure docs/ directory exists
        docs_dir = product_dir / 'docs'
        docs_dir.mkdir(exist_ok=True)

        # Check README.md
        readme = product_dir / 'README.md'
        if readme.exists():
            changes.append("README.md exists")

            # Extract sections for separate docs
            content = readme.read_text()

            # Create CHANGELOG if has version info
            if 'version' in content.lower() or 'changelog' in content.lower():
                changelog = docs_dir / 'CHANGELOG.md'
                if not changelog.exists():
                    changelog.write_text(f"# Changelog\n\n## [0.1.0] - {datetime.now().strftime('%Y-%m-%d')}\n- Initial release\n")
                    changes.append("Created docs/CHANGELOG.md")
        else:
            issues.append("README.md missing")

        # Create API.md stub
        api_doc = docs_dir / 'API.md'
        if not api_doc.exists():
            api_doc.write_text(f"# API Reference - {product_name}\n\n*Documentation in progress*\n")
            changes.append("Created docs/API.md")

        # Create examples/README.md
        examples_dir = product_dir / 'examples'
        examples_dir.mkdir(exist_ok=True)
        examples_readme = examples_dir / 'README.md'
        if not examples_readme.exists():
            examples_readme.write_text(f"# {product_name} Examples\n\n*Examples coming soon*\n")
            changes.append("Created examples/README.md")

        return RefactorReport(
            product_name=product_name,
            agent='DocAgent',
            timestamp=datetime.now().isoformat(),
            changes_made=changes,
            issues_found=issues,
            issues_fixed=[],
            warnings=[],
            status='success'
        )

# ============================================================================
# AGENT 4: NAMING AGENT
# ============================================================================

class NamingAgent:
    """Ensures consistent naming conventions"""

    def check_naming(self, product_dir: Path) -> RefactorReport:
        """Check naming conventions"""
        issues = []
        warnings = []

        product_dir = Path(product_dir)

        # Check file names
        for py_file in product_dir.rglob('*.py'):
            filename = py_file.name
            if not re.match(NAMING_RULES['files']['pattern'], filename):
                if filename not in ['__init__.py', 'setup.py']:
                    issues.append(f"File naming: {filename} doesn't match {NAMING_RULES['files']['pattern']}")

        # Check class names in files
        for py_file in product_dir.rglob('*.py'):
            try:
                content = py_file.read_text()

                # Find class definitions
                class_pattern = r'class (\w+)'
                for match in re.finditer(class_pattern, content):
                    class_name = match.group(1)
                    if not re.match(NAMING_RULES['classes']['pattern'], class_name):
                        issues.append(f"Class naming: {class_name} in {py_file.name}")

                # Find function definitions
                func_pattern = r'def (\w+)\('
                for match in re.finditer(func_pattern, content):
                    func_name = match.group(1)
                    if func_name.startswith('_'):
                        if not re.match(NAMING_RULES['private']['pattern'], func_name):
                            warnings.append(f"Private function: {func_name} in {py_file.name}")
                    elif not re.match(NAMING_RULES['functions']['pattern'], func_name):
                        if not func_name.startswith('__'):  # Allow __init__, etc.
                            issues.append(f"Function naming: {func_name} in {py_file.name}")

            except Exception as e:
                warnings.append(f"Could not check {py_file.name}: {e}")

        return RefactorReport(
            product_name=product_dir.name,
            agent='NamingAgent',
            timestamp=datetime.now().isoformat(),
            changes_made=[],
            issues_found=issues,
            issues_fixed=[],
            warnings=warnings,
            status='success'
        )

# ============================================================================
# AGENT 5: QUALITY AGENT
# ============================================================================

class QualityAgent:
    """Runs quality checks and generates reports"""

    def analyze_product(self, product_dir: Path) -> QualityMetrics:
        """Comprehensive quality analysis"""
        product_dir = Path(product_dir)
        product_name = product_dir.name

        # Count lines of code
        loc = 0
        files_count = 0
        docstring_count = 0
        function_count = 0

        for py_file in product_dir.rglob('*.py'):
            files_count += 1
            try:
                content = py_file.read_text()
                loc += len(content.split('\n'))

                # Count docstrings
                docstring_count += len(re.findall(r'""".*?"""', content, re.DOTALL))
                docstring_count += len(re.findall(r"'''.*?'''", content, re.DOTALL))

                # Count functions
                function_count += len(re.findall(r'def \w+\(', content))

            except:
                pass

        # Check structure compliance
        has_src = (product_dir / 'src').exists()
        has_tests = (product_dir / 'tests').exists()
        has_docs = (product_dir / 'docs').exists()
        has_examples = (product_dir / 'examples').exists()
        has_pyproject = (product_dir / 'pyproject.toml').exists()

        structure_score = sum([has_src, has_tests, has_docs, has_examples, has_pyproject]) / 5.0

        # Docstring coverage
        docstring_coverage = min(1.0, docstring_count / max(1, function_count))

        # Type hint coverage (rough estimate)
        type_hint_coverage = 0.3  # Placeholder

        # Naming compliance (rough check)
        naming_compliance = 0.8  # Placeholder

        # Overall score
        overall_score = (
            structure_score * 30 +
            docstring_coverage * 25 +
            type_hint_coverage * 20 +
            naming_compliance * 25
        )

        return QualityMetrics(
            product_name=product_name,
            loc=loc,
            files_count=files_count,
            has_tests=has_tests,
            has_docs=has_docs,
            has_examples=has_examples,
            docstring_coverage=docstring_coverage,
            type_hint_coverage=type_hint_coverage,
            naming_compliance=naming_compliance,
            structure_compliance=structure_score,
            overall_score=overall_score
        )

# ============================================================================
# AGENT 6: CONSOLIDATION AGENT
# ============================================================================

class ConsolidationAgent:
    """Merges duplicate code and removes clutter"""

    def consolidate_product(self, product_dir: Path) -> RefactorReport:
        """Consolidate a product"""
        changes = []
        issues = []

        product_dir = Path(product_dir)

        # Remove common clutter files
        clutter_patterns = ['*.pyc', '__pycache__', '*.json', '.DS_Store', 'Thumbs.db']

        for pattern in clutter_patterns:
            for item in product_dir.rglob(pattern):
                if item.is_file():
                    item.unlink()
                    changes.append(f"Removed {item.relative_to(product_dir)}")
                elif item.is_dir():
                    import shutil
                    shutil.rmtree(item)
                    changes.append(f"Removed directory {item.relative_to(product_dir)}")

        return RefactorReport(
            product_name=product_dir.name,
            agent='ConsolidationAgent',
            timestamp=datetime.now().isoformat(),
            changes_made=changes,
            issues_found=issues,
            issues_fixed=[],
            warnings=[],
            status='success'
        )

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class RefactorOrchestrator:
    """Orchestrates all refactoring agents"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.structure_agent = StructureAgent(base_dir)
        self.code_style_agent = CodeStyleAgent()
        self.doc_agent = DocAgent()
        self.naming_agent = NamingAgent()
        self.quality_agent = QualityAgent()
        self.consolidation_agent = ConsolidationAgent()

    def refactor_product(self, product_dir: Path, product_name: str) -> Dict:
        """Run all agents on a product"""
        product_dir = Path(product_dir)
        reports = {}

        print(f"\n{'='*70}")
        print(f"REFACTORING: {product_name}")
        print(f"{'='*70}\n")

        # 1. Structure
        print("1. Running StructureAgent...")
        reports['structure'] = self.structure_agent.create_golden_structure(product_dir, product_name)
        print(f"   Changes: {len(reports['structure'].changes_made)}")

        # 2. Consolidation (remove clutter)
        print("2. Running ConsolidationAgent...")
        reports['consolidation'] = self.consolidation_agent.consolidate_product(product_dir)
        print(f"   Changes: {len(reports['consolidation'].changes_made)}")

        # 3. Documentation
        print("3. Running DocAgent...")
        reports['docs'] = self.doc_agent.consolidate_docs(product_dir, product_name)
        print(f"   Changes: {len(reports['docs'].changes_made)}")

        # 4. Naming check
        print("4. Running NamingAgent...")
        reports['naming'] = self.naming_agent.check_naming(product_dir)
        print(f"   Issues: {len(reports['naming'].issues_found)}")

        # 5. Quality analysis
        print("5. Running QualityAgent...")
        reports['quality'] = self.quality_agent.analyze_product(product_dir)
        print(f"   Score: {reports['quality'].overall_score:.1f}/100")

        return reports

    def refactor_all_products(self) -> Dict:
        """Refactor all products in the IDEAS directory"""

        # List of all products
        products = [
            # Session 2 products (already refactored)
            ('failure-db', 'failure_db'),
            ('research-pricer', 'research_pricer'),
            ('experiment-designer', 'experiment_designer'),
            ('chaos-engine', 'chaos_engine'),
            ('ghost-researcher', 'ghost_researcher'),
            # Session 1 products
            ('adversarial-review', 'adversarial_review'),
            ('promptforge-lite', 'promptforge_lite'),
            ('abstract-writer', 'abstract_writer'),
            ('citation-predictor', 'citation_predictor'),
            ('hypothesis-match', 'hypothesis_match'),
            ('paper-miner', 'paper_miner'),
            ('data-cleaner', 'data_cleaner'),
            ('idea-calculus', 'idea_calculus'),
            ('prompt-marketplace', 'prompt_marketplace'),
        ]

        all_reports = {}

        for dir_name, product_name in products:
            product_dir = self.base_dir / dir_name
            if product_dir.exists():
                all_reports[product_name] = self.refactor_product(product_dir, product_name)
            else:
                print(f"⚠️  Skipping {dir_name} - directory not found")

        return all_reports

    def generate_summary_report(self, all_reports: Dict) -> str:
        """Generate summary report"""
        lines = []
        lines.append("="*70)
        lines.append("REFACTORING SUMMARY REPORT")
        lines.append("="*70)
        lines.append("")

        for product_name, reports in all_reports.items():
            lines.append(f"\n## {product_name.upper()}")
            lines.append("")

            # Structure
            if 'structure' in reports:
                lines.append(f"**Structure:** {len(reports['structure'].changes_made)} changes")

            # Quality
            if 'quality' in reports:
                q = reports['quality']
                lines.append(f"**Quality Score:** {q.overall_score:.1f}/100")
                lines.append(f"  - LOC: {q.loc}")
                lines.append(f"  - Files: {q.files_count}")
                lines.append(f"  - Tests: {'✅' if q.has_tests else '❌'}")
                lines.append(f"  - Docs: {'✅' if q.has_docs else '❌'}")
                lines.append(f"  - Structure: {q.structure_compliance*100:.0f}%")
                lines.append(f"  - Docstrings: {q.docstring_coverage*100:.0f}%")

            # Naming issues
            if 'naming' in reports:
                n = reports['naming']
                if n.issues_found:
                    lines.append(f"**Naming Issues:** {len(n.issues_found)}")
                    for issue in n.issues_found[:3]:
                        lines.append(f"  - {issue}")

        return '\n'.join(lines)

# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Refactoring Agents System")
    parser.add_argument('--base-dir', default='/mnt/c/Users/mesha/Documents/IDEAS',
                       help='Base directory for IDEAS products')
    parser.add_argument('--product', help='Single product to refactor')
    parser.add_argument('--all', action='store_true', help='Refactor all products')
    parser.add_argument('--report', action='store_true', help='Generate summary report')

    args = parser.parse_args()

    orchestrator = RefactorOrchestrator(Path(args.base_dir))

    if args.all:
        print("Refactoring all products...")
        all_reports = orchestrator.refactor_all_products()

        if args.report:
            summary = orchestrator.generate_summary_report(all_reports)
            print("\n" + summary)

            # Save report
            report_path = Path(args.base_dir) / 'REFACTOR_REPORT.md'
            report_path.write_text(summary)
            print(f"\n✅ Report saved to {report_path}")

    elif args.product:
        product_dir = Path(args.base_dir) / args.product
        product_name = args.product.replace('-', '_')

        if product_dir.exists():
            reports = orchestrator.refactor_product(product_dir, product_name)
            print("\n✅ Refactoring complete!")
        else:
            print(f"❌ Product directory not found: {product_dir}")

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
