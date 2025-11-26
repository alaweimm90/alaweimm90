#!/usr/bin/env python3
"""
enforce.py - Master Idempotent Enforcement Script

Enforces Golden Path governance policies across repositories:
- Schema validation for .meta/repo.yaml
- Docker security best practices
- Repository structure compliance
- CODEOWNERS and CI/CD configuration

Usage:
    python enforce.py ./organizations/my-org/         # Enforce on org
    python enforce.py ./organizations/my-org/ --report json --output results.json
    python enforce.py ./organizations/my-org/ --strict --fail-on-warnings
"""

import json
import sys
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import click
import yaml
import jsonschema
from tabulate import tabulate


class PolicyEnforcer:
    """Enforces governance policies on repositories."""

    # Schema-defined required files by tier
    TIER_REQUIREMENTS = {
        1: {  # Mission-critical
            "required_files": [".meta/repo.yaml", "README.md", ".github/CODEOWNERS",
                               ".github/workflows/ci.yml", "tests/"],
            "coverage_min": 90,
            "docs_profile": "standard"
        },
        2: {  # Important
            "required_files": [".meta/repo.yaml", "README.md", ".github/CODEOWNERS",
                               ".github/workflows/ci.yml"],
            "coverage_min": 80,
            "docs_profile": "standard"
        },
        3: {  # Experimental
            "required_files": [".meta/repo.yaml", "README.md"],
            "coverage_min": 60,
            "docs_profile": "minimal"
        },
        4: {  # Unknown
            "required_files": [".meta/repo.yaml", "README.md"],
            "coverage_min": 0,
            "docs_profile": "minimal"
        }
    }

    # Docker security patterns
    DOCKER_SECURITY_PATTERNS = {
        "has_user": re.compile(r'^USER\s+(?!root)', re.MULTILINE),
        "has_healthcheck": re.compile(r'^HEALTHCHECK\s+', re.MULTILINE),
        "latest_tag": re.compile(r'^FROM\s+\S+:latest', re.MULTILINE),
        "untagged_from": re.compile(r'^FROM\s+([^:\s@]+)\s*$', re.MULTILINE),
        "add_not_copy": re.compile(r'^ADD\s+(?!https?://)[^.]*(?<!\.tar)(?<!\.tar\.gz)(?<!\.zip)\s', re.MULTILINE),
        "apt_no_y": re.compile(r'apt-get\s+install\s+(?!.*-y)', re.MULTILINE),
        "secret_in_env": re.compile(r'^ENV\s+\S*(PASSWORD|SECRET|TOKEN|API_KEY|PRIVATE_KEY)', re.MULTILINE | re.IGNORECASE),
        "privileged_port": re.compile(r'^EXPOSE\s+(\d+)', re.MULTILINE),
    }

    def __init__(self, repo_path: Path, strict: bool = False,
                 schema_path: Optional[Path] = None, central_path: Optional[Path] = None):
        self.repo_path = Path(repo_path)
        self.strict = strict
        self.violations: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.info: List[Dict[str, Any]] = []
        self.metadata: Optional[Dict[str, Any]] = None

        # Auto-detect central path
        self.central_path = central_path or self._find_central_path()
        self.schema_path = schema_path or (self.central_path / ".metaHub" / "schemas" / "repo-schema.json")
        self.schema = self._load_schema()

    def _find_central_path(self) -> Path:
        """Find the central governance repo path."""
        # Check environment variable
        if env_path := os.environ.get("GOLDEN_PATH_ROOT"):
            path = Path(env_path)
            if path.exists() and (path / ".metaHub").exists():
                return path

        # Search up from current directory
        current = Path.cwd()
        while current != current.parent:
            if (current / ".metaHub").exists():
                return current
            current = current.parent

        # Fallback to script location
        script_path = Path(__file__).resolve().parent.parent.parent
        if (script_path / ".metaHub").exists():
            return script_path

        raise RuntimeError("Could not find central governance repo (.metaHub directory)")

    def _load_schema(self) -> Optional[Dict[str, Any]]:
        """Load JSON schema for validation."""
        if self.schema_path and self.schema_path.exists():
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _add_violation(self, check: str, message: str, file_path: Optional[str] = None,
                       line: Optional[int] = None, severity: str = "error"):
        """Record a policy violation."""
        entry = {
            "check": check,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        if file_path:
            entry["file"] = file_path
        if line:
            entry["line"] = line
        self.violations.append(entry)

    def _add_warning(self, check: str, message: str, file_path: Optional[str] = None):
        """Record a policy warning."""
        self.warnings.append({
            "check": check,
            "message": message,
            "file": file_path,
            "severity": "warning"
        })

    def _add_info(self, check: str, message: str):
        """Record informational message."""
        self.info.append({
            "check": check,
            "message": message,
            "severity": "info"
        })

    def check_all(self) -> Tuple[int, int]:
        """Run all enforcement checks."""
        if not self.repo_path.exists():
            self._add_violation("repo_exists", f"Repository path does not exist: {self.repo_path}")
            return len(self.violations), len(self.warnings)

        # Core checks
        self.check_metadata()
        self.check_repo_structure()
        self.check_docker()
        self.check_codeowners()
        self.check_workflows()
        self.check_readme()

        return len(self.violations), len(self.warnings)

    def check_metadata(self) -> bool:
        """Validate .meta/repo.yaml against schema."""
        meta_file = self.repo_path / ".meta" / "repo.yaml"

        if not meta_file.exists():
            self._add_violation("metadata", ".meta/repo.yaml is required", str(meta_file))
            return False

        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                self.metadata = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self._add_violation("metadata", f"Invalid YAML in .meta/repo.yaml: {e}", str(meta_file))
            return False

        if not self.metadata:
            self._add_violation("metadata", ".meta/repo.yaml is empty", str(meta_file))
            return False

        # Schema validation
        if self.schema:
            try:
                jsonschema.validate(self.metadata, self.schema)
                self._add_info("metadata", "Schema validation passed")
            except jsonschema.ValidationError as e:
                self._add_violation("metadata", f"Schema validation failed: {e.message}", str(meta_file))
                return False

        # Required fields check
        required_fields = ["type", "language"]
        for field in required_fields:
            if field not in self.metadata:
                self._add_violation("metadata", f"Missing required field: {field}", str(meta_file))

        # Tier validation
        tier = self.metadata.get("tier", 4)
        if not isinstance(tier, int) or tier < 1 or tier > 4:
            self._add_warning("metadata", f"Tier should be integer 1-4, got: {tier}", str(meta_file))

        return True

    def check_repo_structure(self) -> bool:
        """Validate repository structure based on tier requirements."""
        if not self.metadata:
            return False

        tier = self.metadata.get("tier", 4)
        requirements = self.TIER_REQUIREMENTS.get(tier, self.TIER_REQUIREMENTS[4])

        missing_files = []
        for required in requirements["required_files"]:
            path = self.repo_path / required
            # Handle directory requirements (ending with /)
            if required.endswith("/"):
                if not path.parent.exists() and not (self.repo_path / required.rstrip("/")).is_dir():
                    missing_files.append(required)
            elif not path.exists():
                missing_files.append(required)

        if missing_files:
            for f in missing_files:
                if self.strict:
                    self._add_violation("structure", f"Missing required file: {f}", f)
                else:
                    self._add_warning("structure", f"Missing recommended file: {f}", f)

        return len(missing_files) == 0

    def check_docker(self) -> bool:
        """Validate Dockerfiles against security policies."""
        dockerfiles = list(self.repo_path.glob("**/Dockerfile*"))

        if not dockerfiles:
            self._add_info("docker", "No Dockerfiles found")
            return True

        all_passed = True
        for dockerfile in dockerfiles:
            try:
                content = dockerfile.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = dockerfile.read_text(encoding='latin-1')

            rel_path = str(dockerfile.relative_to(self.repo_path))

            # Check for USER directive (non-root)
            if not self.DOCKER_SECURITY_PATTERNS["has_user"].search(content):
                self._add_violation("docker", "Dockerfile must run as non-root user (USER directive)", rel_path)
                all_passed = False

            # Check for HEALTHCHECK
            if not self.DOCKER_SECURITY_PATTERNS["has_healthcheck"].search(content):
                self._add_warning("docker", "Dockerfile should include HEALTHCHECK", rel_path)

            # Check for :latest tag
            if self.DOCKER_SECURITY_PATTERNS["latest_tag"].search(content):
                self._add_violation("docker", "Do not use :latest tag in FROM directive", rel_path)
                all_passed = False

            # Check for untagged FROM
            if self.DOCKER_SECURITY_PATTERNS["untagged_from"].search(content):
                self._add_violation("docker", "FROM directive must specify version tag", rel_path)
                all_passed = False

            # Check for ADD instead of COPY
            if self.DOCKER_SECURITY_PATTERNS["add_not_copy"].search(content):
                self._add_warning("docker", "Use COPY instead of ADD unless extracting archives", rel_path)

            # Check for secrets in ENV
            if self.DOCKER_SECURITY_PATTERNS["secret_in_env"].search(content):
                self._add_violation("docker", "Do not hardcode secrets in ENV directives", rel_path)
                all_passed = False

            # Check for privileged ports
            for match in self.DOCKER_SECURITY_PATTERNS["privileged_port"].finditer(content):
                port = int(match.group(1))
                if port < 1024:
                    self._add_violation("docker", f"Do not EXPOSE privileged port {port}", rel_path)
                    all_passed = False

        return all_passed

    def check_codeowners(self) -> bool:
        """Check CODEOWNERS file exists and is valid."""
        codeowners = self.repo_path / ".github" / "CODEOWNERS"

        if not codeowners.exists():
            if self.metadata and self.metadata.get("tier", 4) <= 2:
                self._add_violation("codeowners", "CODEOWNERS required for tier 1-2 repos", str(codeowners))
                return False
            else:
                self._add_warning("codeowners", "CODEOWNERS recommended", str(codeowners))
                return True

        # Validate CODEOWNERS content
        try:
            content = codeowners.read_text(encoding='utf-8')
            if not content.strip():
                self._add_violation("codeowners", "CODEOWNERS file is empty", str(codeowners))
                return False

            # Check for at least one valid ownership rule
            has_rule = False
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    has_rule = True
                    break

            if not has_rule:
                self._add_violation("codeowners", "CODEOWNERS has no ownership rules", str(codeowners))
                return False

        except Exception as e:
            self._add_violation("codeowners", f"Error reading CODEOWNERS: {e}", str(codeowners))
            return False

        return True

    def check_workflows(self) -> bool:
        """Check CI/CD workflow configuration."""
        workflows_dir = self.repo_path / ".github" / "workflows"

        if not workflows_dir.exists():
            if self.metadata and self.metadata.get("tier", 4) <= 2:
                self._add_violation("workflows", "CI/CD workflows required for tier 1-2 repos")
                return False
            else:
                self._add_warning("workflows", "CI/CD workflows recommended")
                return True

        # Check for ci.yml or similar
        ci_files = list(workflows_dir.glob("ci*.yml")) + list(workflows_dir.glob("ci*.yaml"))
        if not ci_files:
            self._add_warning("workflows", "No CI workflow found (ci.yml)")

        # Validate workflow files
        for wf_file in workflows_dir.glob("*.y*ml"):
            try:
                with open(wf_file, 'r', encoding='utf-8') as f:
                    workflow = yaml.safe_load(f)

                if workflow:
                    # Check for permissions block (security hardening)
                    if 'permissions' not in workflow:
                        self._add_warning("workflows",
                            f"Workflow {wf_file.name} missing permissions block", str(wf_file))

                    # Check for continue-on-error (anti-pattern)
                    self._check_continue_on_error(workflow, wf_file.name)

            except yaml.YAMLError as e:
                self._add_violation("workflows", f"Invalid YAML in {wf_file.name}: {e}", str(wf_file))

        return True

    def _check_continue_on_error(self, workflow: Dict, filename: str, path: str = ""):
        """Recursively check for continue-on-error usage."""
        if isinstance(workflow, dict):
            if workflow.get("continue-on-error") is True:
                self._add_warning("workflows",
                    f"{filename}: continue-on-error=true at {path or 'root'} may hide failures")
            for key, value in workflow.items():
                self._check_continue_on_error(value, filename, f"{path}.{key}" if path else key)
        elif isinstance(workflow, list):
            for i, item in enumerate(workflow):
                self._check_continue_on_error(item, filename, f"{path}[{i}]")

    def check_readme(self) -> bool:
        """Check README.md exists and has minimum content."""
        readme = self.repo_path / "README.md"

        if not readme.exists():
            self._add_violation("readme", "README.md is required", str(readme))
            return False

        try:
            content = readme.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = readme.read_text(encoding='latin-1')

        # Check minimum content
        if len(content.strip()) < 50:
            self._add_warning("readme", "README.md is too short (< 50 chars)", str(readme))

        # Check for title
        if not content.strip().startswith('#'):
            self._add_warning("readme", "README.md should start with a title (#)", str(readme))

        return True

    def report(self, fmt: str = 'text') -> str:
        """Generate enforcement report."""
        if fmt == 'json':
            return json.dumps({
                'repo': str(self.repo_path),
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'violations': len(self.violations),
                    'warnings': len(self.warnings),
                    'info': len(self.info),
                    'passed': len(self.violations) == 0
                },
                'metadata': self.metadata,
                'violations': self.violations,
                'warnings': self.warnings,
                'info': self.info
            }, indent=2)

        # Text format
        lines = [
            f"\n{'='*60}",
            f"Enforcement Report: {self.repo_path.name}",
            f"{'='*60}",
            f"\nSummary:",
            f"  Violations: {len(self.violations)}",
            f"  Warnings:   {len(self.warnings)}",
            f"  Info:       {len(self.info)}",
        ]

        if self.violations:
            lines.append(f"\n[VIOLATIONS]")
            for v in self.violations:
                lines.append(f"  [ERROR] {v['check']}: {v['message']}")
                if v.get('file'):
                    lines.append(f"          File: {v['file']}")

        if self.warnings:
            lines.append(f"\n[WARNINGS]")
            for w in self.warnings:
                lines.append(f"  [WARN] {w['check']}: {w['message']}")
                if w.get('file'):
                    lines.append(f"         File: {w['file']}")

        if self.info:
            lines.append(f"\n[INFO]")
            for i in self.info:
                lines.append(f"  [INFO] {i['check']}: {i['message']}")

        status = "[PASS]" if len(self.violations) == 0 else "[FAIL]"
        lines.append(f"\n{status} Enforcement {'passed' if len(self.violations) == 0 else 'failed'}")

        return '\n'.join(lines)


def enforce_organization(org_path: Path, strict: bool = False,
                        schema_path: Optional[Path] = None,
                        central_path: Optional[Path] = None) -> Dict[str, Any]:
    """Enforce policies across all repos in an organization."""
    results = {
        "organization": org_path.name,
        "timestamp": datetime.now().isoformat(),
        "repos": [],
        "summary": {
            "total_repos": 0,
            "passed": 0,
            "failed": 0,
            "total_violations": 0,
            "total_warnings": 0
        }
    }

    # Iterate through all repo directories
    for repo_dir in sorted(org_path.iterdir()):
        if not repo_dir.is_dir():
            continue
        if repo_dir.name.startswith('.'):
            continue

        results["summary"]["total_repos"] += 1

        enforcer = PolicyEnforcer(repo_dir, strict=strict,
                                   schema_path=schema_path,
                                   central_path=central_path)
        violations, warnings = enforcer.check_all()

        repo_result = {
            "name": repo_dir.name,
            "path": str(repo_dir),
            "passed": violations == 0,
            "violations": enforcer.violations,
            "warnings": enforcer.warnings,
            "info": enforcer.info,
            "metadata": enforcer.metadata
        }
        results["repos"].append(repo_result)

        if violations == 0:
            results["summary"]["passed"] += 1
        else:
            results["summary"]["failed"] += 1

        results["summary"]["total_violations"] += violations
        results["summary"]["total_warnings"] += warnings

    return results


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--strict', is_flag=True, help='Treat warnings as violations')
@click.option('--report', 'report_fmt', type=click.Choice(['text', 'json']),
              default='text', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--fail-on-warnings', is_flag=True, help='Exit with error if warnings exist')
@click.option('--schema', type=click.Path(exists=True), help='Path to JSON schema')
def main(path: str, strict: bool, report_fmt: str, output: Optional[str],
         fail_on_warnings: bool, schema: Optional[str]):
    """
    Enforce Golden Path governance policies.

    PATH can be a single repository or an organization directory containing multiple repos.

    Examples:

        python enforce.py ./organizations/my-org/

        python enforce.py ./organizations/my-org/ --report json --output results.json

        python enforce.py ./my-repo --strict --fail-on-warnings
    """
    path_obj = Path(path)
    schema_path = Path(schema) if schema else None

    # Determine if this is an org directory or single repo
    is_org = (path_obj / ".meta").exists() is False and any(
        (d / ".meta").exists() for d in path_obj.iterdir() if d.is_dir()
    )

    if is_org:
        # Organization-level enforcement
        results = enforce_organization(path_obj, strict=strict, schema_path=schema_path)

        if report_fmt == 'json':
            report_output = json.dumps(results, indent=2)
        else:
            # Generate text summary
            lines = [
                f"\n{'='*60}",
                f"Organization Enforcement Report: {results['organization']}",
                f"{'='*60}",
                f"\nSummary:",
                f"  Total Repos:      {results['summary']['total_repos']}",
                f"  Passed:           {results['summary']['passed']}",
                f"  Failed:           {results['summary']['failed']}",
                f"  Total Violations: {results['summary']['total_violations']}",
                f"  Total Warnings:   {results['summary']['total_warnings']}",
            ]

            # List failed repos
            failed_repos = [r for r in results['repos'] if not r['passed']]
            if failed_repos:
                lines.append(f"\n[FAILED REPOS]")
                for repo in failed_repos:
                    lines.append(f"  - {repo['name']}: {len(repo['violations'])} violations")

            # List repos with warnings only
            warning_repos = [r for r in results['repos'] if r['passed'] and r['warnings']]
            if warning_repos:
                lines.append(f"\n[REPOS WITH WARNINGS]")
                for repo in warning_repos:
                    lines.append(f"  - {repo['name']}: {len(repo['warnings'])} warnings")

            status = "[PASS]" if results['summary']['failed'] == 0 else "[FAIL]"
            lines.append(f"\n{status} Organization enforcement complete")

            report_output = '\n'.join(lines)

        exit_code = 1 if results['summary']['failed'] > 0 else 0
        if fail_on_warnings and results['summary']['total_warnings'] > 0:
            exit_code = 1

    else:
        # Single repo enforcement
        enforcer = PolicyEnforcer(path_obj, strict=strict, schema_path=schema_path)
        violations, warnings = enforcer.check_all()
        report_output = enforcer.report(fmt=report_fmt)

        exit_code = 1 if violations > 0 else 0
        if fail_on_warnings and warnings > 0:
            exit_code = 1

    # Output handling
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(report_output)
        click.echo(f"Report written to: {output}")
    else:
        click.echo(report_output)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
