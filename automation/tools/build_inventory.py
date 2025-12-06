from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

WORKSPACE = Path(__file__).resolve().parents[2]
REPORT_DIR = WORKSPACE / "automation" / "reports"
REPORT_JSON = REPORT_DIR / "build_inventory.json"

SCAN_TARGETS: Dict[str, Path] = {
    "librex": Path("alawein-technologies-llc/librex"),
    "mezan": Path("alawein-technologies-llc/mezan"),
    "simcore": Path("alawein-technologies-llc/simcore"),
    "qmlab": Path("alawein-technologies-llc/qmlab"),
    "marketing-automation": Path("alawein-technologies-llc/marketing-automation"),
    "liveiticonic": Path("live-it-iconic-llc/liveiticonic"),
    "repz": Path("repz-llc/repz"),
    "qmatsim": Path("research/qmatsim"),
    "qubeml": Path("research/qubeml"),
    "scicomp": Path("research/scicomp"),
}

MANIFEST_FILES: Sequence[str] = (
    "pyproject.toml",
    "package.json",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "Pipfile",
    "environment.yml",
)

DEPENDENCY_FILES: Sequence[str] = (
    "requirements-dev.txt",
    "poetry.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.toml",
)

BUILD_FILES: Sequence[str] = (
    "Makefile",
    "noxfile.py",
    "tox.ini",
    "Taskfile.yml",
    "justfile",
    "Dockerfile",
    "docker-compose.yml",
)

TEST_DIRS: Sequence[str] = (
    "tests",
    "integration_tests",
    "benchmarks",
    "examples",
)

CI_ROOTS: Sequence[Path] = (
    Path(".github") / "workflows",
    Path(".orchex") / "workflows",
)

SKIP_DIRS: Sequence[str] = (
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
)


@dataclass
class ProjectInventory:
    project: str
    exists: bool
    manifests: List[str]
    dependency_files: List[str]
    build_scripts: List[str]
    test_assets: List[str]
    ci_workflows: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def iter_dirs(root: Path, max_depth: int) -> Iterable[Path]:
    stack: List[tuple[Path, int]] = [(root, 0)]
    seen: set[Path] = set()
    while stack:
        current, depth = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        yield current
        if depth >= max_depth:
            continue
        for child in sorted(current.iterdir()):
            if child.is_dir() and child.name not in SKIP_DIRS:
                stack.append((child, depth + 1))


def find_named_entries(root: Path, names: Sequence[str], max_depth: int = 2) -> List[str]:
    results: List[str] = []
    for base in iter_dirs(root, max_depth):
        for name in names:
            candidate = base / name
            if candidate.exists():
                results.append(candidate.relative_to(WORKSPACE).as_posix())
    return sorted(set(results))


def discover_ci_workflows(root: Path) -> List[str]:
    workflows: List[str] = []
    for ci_root in CI_ROOTS:
        candidate = root / ci_root
        if not candidate.exists() or not candidate.is_dir():
            continue
        workflows.extend(
            wf.relative_to(WORKSPACE).as_posix()
            for wf in candidate.rglob("*.yml")
        )
    return sorted(set(workflows))


def collect_inventory(project: str, relative_path: Path) -> ProjectInventory:
    root = WORKSPACE / relative_path
    if not root.exists():
        return ProjectInventory(
            project=project,
            exists=False,
            manifests=[],
            dependency_files=[],
            build_scripts=[],
            test_assets=[],
            ci_workflows=[],
        )

    manifests = find_named_entries(root, MANIFEST_FILES)
    dependencies = find_named_entries(root, DEPENDENCY_FILES)
    build_scripts = find_named_entries(root, BUILD_FILES)
    test_assets = find_named_entries(root, TEST_DIRS)
    ci_workflows = discover_ci_workflows(root)

    return ProjectInventory(
        project=project,
        exists=True,
        manifests=manifests,
        dependency_files=dependencies,
        build_scripts=build_scripts,
        test_assets=test_assets,
        ci_workflows=ci_workflows,
    )


def generate_report() -> Dict[str, Dict[str, object]]:
    inventory: Dict[str, Dict[str, object]] = {}
    for project, rel_path in SCAN_TARGETS.items():
        inventory[project] = collect_inventory(project, rel_path).to_dict()
    return inventory


def write_report(data: Dict[str, Dict[str, object]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(data, indent=2))


def main() -> None:
    report = generate_report()
    write_report(report)
    print(f"Wrote build inventory for {len(report)} projects -> {REPORT_JSON.relative_to(WORKSPACE)}")


if __name__ == "__main__":
    main()
