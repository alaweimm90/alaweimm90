from pathlib import Path
from datetime import datetime
from typing import Dict

from ..core.models import RefactorReport
from ..core.standards import GOLDEN_STRUCTURE, CODE_STANDARDS


class StructureAgent:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def analyze_current_structure(self, product_dir: Path) -> Dict:
        structure = {}
        for item in Path(product_dir).rglob('*'):
            if item.is_file():
                rel_path = item.relative_to(product_dir)
                structure[str(rel_path)] = {
                    'size': item.stat().st_size,
                    'type': 'file'
                }
        return structure

    def create_golden_structure(self, product_dir: Path, product_name: str) -> RefactorReport:
        changes, issues = [], []
        product_dir = Path(product_dir)

        for dir_path in [
            product_dir / 'src' / product_name,
            product_dir / 'tests',
            product_dir / 'examples',
            product_dir / 'docs']:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                changes.append(f"Created directory: {dir_path.relative_to(product_dir)}")

        main_files = list(product_dir.glob('*.py'))
        if main_files:
            main_file = main_files[0]
            new_location = product_dir / 'src' / product_name / 'main.py'
            if main_file != new_location and main_file.name not in ['setup.py', 'refactor_agents.py']:
                if not new_location.exists():
                    import shutil
                    shutil.copy2(main_file, new_location)
                    changes.append(f"Copied {main_file.name} → src/{product_name}/main.py")

        init_files = [
            (product_dir / 'src' / product_name / '__init__.py',
             '"""Package initialization."""\n\nfrom .main import main\n\n__version__ = "0.1.0"\n'),
            (product_dir / 'tests' / '__init__.py', ''),
        ]
        for init_path, content in init_files:
            if not init_path.exists():
                init_path.write_text(content)
                changes.append(f"Created {init_path.relative_to(product_dir)}")

        pyproject_path = product_dir / 'pyproject.toml'
        if not pyproject_path.exists():
            pyproject_content = self._generate_pyproject(product_name)
            pyproject_path.write_text(pyproject_content)
            changes.append("Created pyproject.toml")

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
        return f'''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{product_name}"
version = "0.1.0"
description = "AlaweinOS tooling"
authors = [
    {{name = "AlaweinOS Team"}}
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

