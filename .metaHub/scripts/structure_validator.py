#!/usr/bin/env python3
"""
structure_validator.py - Portfolio Structure Validator

Validates and enforces folder/file structure across the portfolio based on
templates defined in portfolio-structure.yaml.

Usage:
    python structure_validator.py                    # Validate all
    python structure_validator.py --org alaweimm90-tools
    python structure_validator.py --fix             # Auto-create missing structure
    python structure_validator.py --report json     # JSON output
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import click
import yaml


class StructureValidator:
    """Validates repository structure against templates."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        self.templates = self._load_templates()
        self.results: Dict[str, Any] = {
            "validated_at": datetime.now().isoformat(),
            "organizations": {},
            "summary": {
                "total_orgs": 0,
                "total_repos": 0,
                "compliant_repos": 0,
                "non_compliant_repos": 0,
                "missing_files": 0,
                "missing_dirs": 0
            }
        }

    def _find_base_path(self) -> Path:
        """Find the central governance repo path."""
        current = Path.cwd()
        while current != current.parent:
            if (current / ".metaHub").exists():
                return current
            current = current.parent
        return Path.cwd()

    def _load_templates(self) -> Dict[str, Any]:
        """Load structure templates from YAML."""
        template_path = self.base_path / ".metaHub/templates/structures/portfolio-structure.yaml"
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def _get_repo_metadata(self, repo_path: Path) -> Optional[Dict[str, Any]]:
        """Read .meta/repo.yaml if it exists."""
        meta_file = repo_path / ".meta" / "repo.yaml"
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return None

    # Directories to always skip during validation (never repos)
    SKIP_DIRS = {
        "node_modules", "__pycache__", ".git", "venv", "env", "dist", "build",
        "vendor", "target", "out", "coverage", ".vscode", ".idea", "cache"
    }

    def _is_likely_repo(self, path: Path) -> bool:
        """Check if a directory looks like an actual repository."""
        # Has README or .meta/repo.yaml = definitely a repo
        if (path / "README.md").exists() or (path / ".meta/repo.yaml").exists():
            return True
        # Has build config = likely a repo
        if any((path / f).exists() for f in [
            "pyproject.toml", "package.json", "go.mod", "Cargo.toml",
            "setup.py", "Makefile", "Dockerfile"
        ]):
            return True
        # Has LICENSE = likely a repo
        if (path / "LICENSE").exists():
            return True
        # Common non-repo directory names without repo indicators
        non_repo_names = {"src", "lib", "bin", "scripts", "examples", "assets",
                         "public", "static", "config", "data", "tmp", "temp"}
        if path.name.lower() in non_repo_names:
            return False
        return True

    def _detect_language(self, repo_path: Path) -> str:
        """Auto-detect primary language from files in repo."""
        # Priority-ordered language indicators
        # Check terraform FIRST (before python since terraform repos might have requirements.txt)
        if list(repo_path.glob("*.tf")) or list(repo_path.glob("**/*.tf")):
            return "terraform"
        if (repo_path / "modules").is_dir() or (repo_path / "environments").is_dir():
            # Likely terraform structure
            if list(repo_path.glob("**/*.tf")):
                return "terraform"

        # Check for mkdocs/docs repos
        if (repo_path / "mkdocs.yml").exists():
            return "docs"
        # Check if it's a docs-only repo (markdown files, no code)
        if repo_path.name.lower() == "docs":
            has_code = any([
                (repo_path / "pyproject.toml").exists(),
                (repo_path / "package.json").exists(),
                (repo_path / "go.mod").exists(),
                (repo_path / "Cargo.toml").exists(),
            ])
            if not has_code:
                return "docs"

        # Standard language indicators
        indicators = [
            ("Cargo.toml", "rust"),
            ("go.mod", "go"),
            ("package.json", "typescript"),
            ("tsconfig.json", "typescript"),
            ("pyproject.toml", "python"),
            ("setup.py", "python"),
            ("requirements.txt", "python"),
        ]

        for indicator, lang in indicators:
            if (repo_path / indicator).exists():
                return lang

        # Check for common source directories
        if (repo_path / "src").is_dir():
            src = repo_path / "src"
            if list(src.glob("**/*.rs")):
                return "rust"
            if list(src.glob("**/*.go")):
                return "go"
            if list(src.glob("**/*.ts")) or list(src.glob("**/*.tsx")):
                return "typescript"
            if list(src.glob("**/*.py")):
                return "python"

        # Check root for language files
        if list(repo_path.glob("*.py")):
            return "python"
        if list(repo_path.glob("*.ts")) or list(repo_path.glob("*.tsx")):
            return "typescript"
        if list(repo_path.glob("*.go")):
            return "go"
        if list(repo_path.glob("*.rs")):
            return "rust"

        return "unknown"

    def _detect_repo_type(self, repo_path: Path, language: str) -> str:
        """Auto-detect repository type from structure."""
        # Type indicators
        if (repo_path / "Dockerfile").exists():
            return "service"
        if (repo_path / "setup.py").exists() or (repo_path / "pyproject.toml").exists():
            # Check if it's a library or tool
            if (repo_path / "src" / "cli").is_dir() or "cli" in repo_path.name.lower():
                return "tool"
            return "library"
        if (repo_path / "package.json").exists():
            pkg_json = repo_path / "package.json"
            try:
                with open(pkg_json) as f:
                    pkg = json.load(f)
                    if "bin" in pkg:
                        return "tool"
                    if pkg.get("private"):
                        return "service"
            except Exception:
                pass
            return "library"
        if language == "docs":
            return "demo"
        if language == "terraform":
            return "tool"

        return "unknown"

    def _get_effective_metadata(self, repo_path: Path) -> Dict[str, Any]:
        """Get metadata, auto-detecting missing fields."""
        metadata = self._get_repo_metadata(repo_path) or {}

        # Always detect language (may override incorrect metadata)
        detected_lang = self._detect_language(repo_path)

        # Override metadata language if detected is more specific
        current_lang = metadata.get("language", "unknown")
        if current_lang in ("unknown", "python") and detected_lang not in ("unknown", "python"):
            # Prefer detected non-python language over generic python
            metadata["language"] = detected_lang
            metadata["_language_detected"] = True
        elif not current_lang or current_lang == "unknown":
            metadata["language"] = detected_lang

        # Auto-detect type if missing or unknown
        if not metadata.get("type") or metadata.get("type") == "unknown":
            metadata["type"] = self._detect_repo_type(repo_path, metadata.get("language", "unknown"))

        # Default tier if missing
        if not metadata.get("tier"):
            metadata["tier"] = 4

        return metadata

    def _get_required_structure(self, metadata: Optional[Dict]) -> Tuple[List[str], List[str]]:
        """Determine required files/dirs based on metadata."""
        required_files = []
        required_dirs = []

        # Base requirements
        base = self.templates.get("repository_base", {})
        required_files.extend(base.get("required", {}).get("files", []))
        required_dirs.extend(base.get("required", {}).get("dirs", []))

        if metadata:
            # Tier requirements
            tier = metadata.get("tier", 4)
            tier_key = f"tier_{tier}"
            tier_reqs = self.templates.get("repository_tiers", {}).get(tier_key, {})
            required_files.extend(tier_reqs.get("additional_required", {}).get("files", []))
            required_dirs.extend(tier_reqs.get("additional_required", {}).get("dirs", []))

            # Language requirements
            language = metadata.get("language", "").lower()
            lang_reqs = self.templates.get("repository_languages", {}).get(language, {})
            required_files.extend(lang_reqs.get("required", {}).get("files", []))
            required_dirs.extend(lang_reqs.get("required", {}).get("dirs", []))

            # Type requirements
            repo_type = metadata.get("type", "").lower()
            type_reqs = self.templates.get("repository_types", {}).get(repo_type, {})
            required_files.extend(type_reqs.get("required", {}).get("files", []))
            required_dirs.extend(type_reqs.get("required", {}).get("dirs", []))

        return list(set(required_files)), list(set(required_dirs))

    def validate_repository(self, repo_path: Path) -> Dict[str, Any]:
        """Validate a single repository against structure requirements."""
        raw_metadata = self._get_repo_metadata(repo_path)
        metadata = self._get_effective_metadata(repo_path)
        required_files, required_dirs = self._get_required_structure(metadata)

        missing_files = []
        missing_dirs = []
        present_files = []
        present_dirs = []

        # Check required files
        for file_path in required_files:
            full_path = repo_path / file_path
            if full_path.exists():
                present_files.append(file_path)
            else:
                missing_files.append(file_path)

        # Check required directories
        for dir_path in required_dirs:
            full_path = repo_path / dir_path.rstrip('/')
            if full_path.is_dir():
                present_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)

        is_compliant = len(missing_files) == 0 and len(missing_dirs) == 0

        return {
            "name": repo_path.name,
            "path": str(repo_path),
            "metadata": metadata,
            "raw_metadata": raw_metadata,
            "has_metadata": raw_metadata is not None,
            "detected_language": metadata.get("language", "unknown"),
            "detected_type": metadata.get("type", "unknown"),
            "compliant": is_compliant,
            "required_files": required_files,
            "required_dirs": required_dirs,
            "present_files": present_files,
            "present_dirs": present_dirs,
            "missing_files": missing_files,
            "missing_dirs": missing_dirs
        }

    def validate_organization(self, org_path: Path) -> Dict[str, Any]:
        """Validate all repositories in an organization."""
        org_result = {
            "name": org_path.name,
            "path": str(org_path),
            "repos": [],
            "summary": {
                "total_repos": 0,
                "compliant": 0,
                "non_compliant": 0,
                "missing_files": 0,
                "missing_dirs": 0
            }
        }

        # Validate each repo
        for repo_dir in sorted(org_path.iterdir()):
            if not repo_dir.is_dir() or repo_dir.name.startswith('.'):
                continue
            # Skip common non-repo directories
            if repo_dir.name in self.SKIP_DIRS:
                continue
            # Check if it looks like an actual repo
            if not self._is_likely_repo(repo_dir):
                continue

            repo_result = self.validate_repository(repo_dir)
            org_result["repos"].append(repo_result)
            org_result["summary"]["total_repos"] += 1

            if repo_result["compliant"]:
                org_result["summary"]["compliant"] += 1
            else:
                org_result["summary"]["non_compliant"] += 1
                org_result["summary"]["missing_files"] += len(repo_result["missing_files"])
                org_result["summary"]["missing_dirs"] += len(repo_result["missing_dirs"])

        return org_result

    def validate_portfolio(self, org_filter: Optional[str] = None) -> Dict[str, Any]:
        """Validate entire portfolio structure."""
        org_path = self.base_path / "organizations"

        if not org_path.exists():
            click.echo(f"Organizations path not found: {org_path}", err=True)
            return self.results

        for org_dir in sorted(org_path.iterdir()):
            if not org_dir.is_dir() or org_dir.name.startswith('.'):
                continue

            if org_filter and org_dir.name != org_filter:
                continue

            org_result = self.validate_organization(org_dir)
            self.results["organizations"][org_dir.name] = org_result
            self.results["summary"]["total_orgs"] += 1
            self.results["summary"]["total_repos"] += org_result["summary"]["total_repos"]
            self.results["summary"]["compliant_repos"] += org_result["summary"]["compliant"]
            self.results["summary"]["non_compliant_repos"] += org_result["summary"]["non_compliant"]
            self.results["summary"]["missing_files"] += org_result["summary"]["missing_files"]
            self.results["summary"]["missing_dirs"] += org_result["summary"]["missing_dirs"]

        return self.results

    def fix_structure(self, repo_path: Path, dry_run: bool = False) -> List[str]:
        """Create missing structure for a repository."""
        changes = []
        result = self.validate_repository(repo_path)

        # Create missing directories
        for dir_path in result["missing_dirs"]:
            full_path = repo_path / dir_path.rstrip('/')
            if dry_run:
                changes.append(f"[DRY-RUN] Would create directory: {dir_path}")
            else:
                full_path.mkdir(parents=True, exist_ok=True)
                # Add .gitkeep to empty directories
                gitkeep = full_path / ".gitkeep"
                gitkeep.touch()
                changes.append(f"Created directory: {dir_path}")

        # Create missing files
        for file_path in result["missing_files"]:
            full_path = repo_path / file_path

            # Skip certain files that need custom content
            if file_path in ["LICENSE", "README.md", ".meta/repo.yaml"]:
                if dry_run:
                    changes.append(f"[DRY-RUN] Would create file: {file_path} (needs manual content)")
                else:
                    if file_path == ".meta/repo.yaml":
                        full_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(full_path, 'w') as f:
                            f.write("# Repository Metadata\n")
                            f.write("type: unknown\n")
                            f.write("language: unknown\n")
                            f.write("tier: 4\n")
                            f.write("# owner: @alaweimm90\n")
                        changes.append(f"Created file: {file_path} (needs update)")
                    elif file_path == "LICENSE":
                        with open(full_path, 'w') as f:
                            f.write("MIT License\n\nCopyright (c) 2025 alaweimm90\n")
                        changes.append(f"Created file: {file_path}")
                    elif file_path == "README.md":
                        with open(full_path, 'w') as f:
                            f.write(f"# {repo_path.name}\n\nTODO: Add description\n")
                        changes.append(f"Created file: {file_path} (needs update)")
                continue

            # Create other required files with useful defaults
            if dry_run:
                changes.append(f"[DRY-RUN] Would create file: {file_path}")
            else:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                # Create files with useful default content
                if file_path == "pyproject.toml":
                    with open(full_path, 'w') as f:
                        f.write(f'''[project]
name = "{repo_path.name}"
version = "0.1.0"
description = ""
requires-python = ">=3.11"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
''')
                elif file_path == "package.json":
                    with open(full_path, 'w') as f:
                        f.write(f'''{{"name": "{repo_path.name}",
  "version": "0.1.0",
  "private": true
}}
''')
                elif file_path == "tsconfig.json":
                    with open(full_path, 'w') as f:
                        f.write('''{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "strict": true,
    "esModuleInterop": true
  }
}
''')
                elif file_path == "SECURITY.md":
                    with open(full_path, 'w') as f:
                        f.write('''# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities via GitHub's private vulnerability reporting.
''')
                else:
                    full_path.touch()
                changes.append(f"Created file: {file_path}")

        return changes


def format_text_report(results: Dict[str, Any]) -> str:
    """Generate text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("PORTFOLIO STRUCTURE VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Validated: {results['validated_at']}")
    lines.append("")

    summary = results["summary"]
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Organizations: {summary['total_orgs']}")
    lines.append(f"Total Repos: {summary['total_repos']}")
    lines.append(f"Compliant: {summary['compliant_repos']}")
    lines.append(f"Non-Compliant: {summary['non_compliant_repos']}")
    lines.append(f"Missing Files: {summary['missing_files']}")
    lines.append(f"Missing Dirs: {summary['missing_dirs']}")

    compliance_pct = (summary['compliant_repos'] / max(summary['total_repos'], 1)) * 100
    lines.append(f"Compliance: {compliance_pct:.1f}%")
    lines.append("")

    for org_name, org_data in results["organizations"].items():
        lines.append(f"\n{org_name.upper()}")
        lines.append("-" * 40)

        for repo in org_data["repos"]:
            status = "[OK]" if repo["compliant"] else "[FAIL]"
            lang = repo.get("detected_language", "?")
            rtype = repo.get("detected_type", "?")
            lines.append(f"  {status} {repo['name']} ({lang}/{rtype})")

            if not repo["compliant"]:
                for f in repo["missing_files"]:
                    lines.append(f"      Missing file: {f}")
                for d in repo["missing_dirs"]:
                    lines.append(f"      Missing dir: {d}")

    return "\n".join(lines)


@click.command()
@click.option('--org', help='Validate specific organization only')
@click.option('--repo', help='Validate specific repository only')
@click.option('--fix', is_flag=True, help='Auto-create missing structure')
@click.option('--dry-run', is_flag=True, help='Show what would be fixed without making changes')
@click.option('--report', type=click.Choice(['text', 'json', 'summary']), default='text')
@click.option('--output', '-o', help='Output file path')
def main(org: Optional[str], repo: Optional[str], fix: bool, dry_run: bool,
         report: str, output: Optional[str]):
    """Validate portfolio structure against templates."""

    validator = StructureValidator()

    if repo:
        # Validate single repo
        repo_path = Path(repo)
        if not repo_path.exists():
            # Try to find it
            for org_dir in (validator.base_path / "organizations").iterdir():
                candidate = org_dir / repo
                if candidate.exists():
                    repo_path = candidate
                    break

        if repo_path.exists():
            result = validator.validate_repository(repo_path)

            if fix or dry_run:
                changes = validator.fix_structure(repo_path, dry_run)
                for change in changes:
                    click.echo(change)

            if report == 'json':
                click.echo(json.dumps(result, indent=2))
            else:
                status = "COMPLIANT" if result["compliant"] else "NON-COMPLIANT"
                click.echo(f"{result['name']}: {status}")
                if not result["compliant"]:
                    click.echo(f"  Missing files: {result['missing_files']}")
                    click.echo(f"  Missing dirs: {result['missing_dirs']}")
        else:
            click.echo(f"Repository not found: {repo}", err=True)
            return

    else:
        # Validate portfolio
        results = validator.validate_portfolio(org_filter=org)

        if fix or dry_run:
            # Fix all non-compliant repos
            for org_name, org_data in results["organizations"].items():
                for repo_data in org_data["repos"]:
                    if not repo_data["compliant"]:
                        changes = validator.fix_structure(Path(repo_data["path"]), dry_run)
                        if changes:
                            click.echo(f"\n{repo_data['name']}:")
                            for change in changes:
                                click.echo(f"  {change}")

        # Output report
        if report == 'json':
            output_text = json.dumps(results, indent=2)
        elif report == 'summary':
            s = results["summary"]
            pct = (s['compliant_repos'] / max(s['total_repos'], 1)) * 100
            output_text = (f"Orgs: {s['total_orgs']} | Repos: {s['total_repos']} | "
                          f"Compliant: {s['compliant_repos']} ({pct:.1f}%) | "
                          f"Missing: {s['missing_files']} files, {s['missing_dirs']} dirs")
        else:
            output_text = format_text_report(results)

        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(output_text)
            click.echo(f"Report saved to: {output}")
        else:
            click.echo(output_text)


if __name__ == "__main__":
    main()
