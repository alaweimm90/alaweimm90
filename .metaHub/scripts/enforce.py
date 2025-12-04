"""
Policy Enforcement Module.

Validates repositories against governance policies including:
- Schema validation
- Docker security checks
- Repository structure
- CODEOWNERS validation
- Workflow checks
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


class PolicyEnforcer:
    """Enforces governance policies on repositories."""

    # Secrets patterns to detect in Dockerfiles
    SECRETS_PATTERNS = [
        r"PASSWORD",
        r"SECRET",
        r"API_KEY",
        r"TOKEN",
        r"PRIVATE_KEY",
        r"CREDENTIAL",
    ]

    def __init__(self, repo_path: Path, central_path: Path = None):
        """
        Initialize policy enforcer.

        Args:
            repo_path: Path to repository to check
            central_path: Path to central governance repo
        """
        self.repo_path = Path(repo_path).resolve()
        self.central_path = (Path(central_path) if central_path else Path.cwd()).resolve()
        self.schema_path = self.central_path / ".metaHub" / "schemas"
        
        # Validate paths are within expected boundaries
        if not self._is_safe_path(self.repo_path):
            raise ValueError(f"Unsafe repository path: {repo_path}")
        if not self._is_safe_path(self.central_path):
            raise ValueError(f"Unsafe central path: {central_path}")
            
        self.violations: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []

    def _is_safe_path(self, path: Path) -> bool:
        """Validate path is safe and within expected boundaries."""
        try:
            resolved = path.resolve()
            # Ensure path doesn't contain traversal attempts
            return ".." not in str(path) and resolved.exists()
        except (OSError, ValueError):
            return False

    def check_all(self) -> Tuple[int, int]:
        """
        Run all policy checks.

        Returns:
            Tuple of (violation_count, warning_count)
        """
        self.violations = []
        self.warnings = []

        self.check_metadata()
        self.check_readme()
        self.check_docker()
        self.check_workflows()
        self.check_codeowners()

        return len(self.violations), len(self.warnings)

    def check_metadata(self) -> None:
        """Check .meta/repo.yaml exists and validates against schema."""
        meta_file = self.repo_path / ".meta" / "repo.yaml"

        if not meta_file.exists():
            self.violations.append({
                "check": "metadata",
                "message": "Missing .meta/repo.yaml metadata file",
                "severity": "error"
            })
            return

        # Load and validate metadata
        try:
            with open(meta_file, encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
        except (yaml.YAMLError, OSError, UnicodeDecodeError) as e:
            self.violations.append({
                "check": "metadata",
                "message": f"Invalid YAML in .meta/repo.yaml: {e}",
                "severity": "error"
            })
            return

        # Schema validation
        schema_file = self.schema_path / "repo-schema.json"
        if schema_file.exists() and HAS_JSONSCHEMA:
            try:
                with open(schema_file, encoding='utf-8') as f:
                    schema = json.load(f)
                jsonschema.validate(metadata, schema)
            except (jsonschema.ValidationError, OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                self.violations.append({
                    "check": "metadata",
                    "message": f"Schema validation failed: {e.message}",
                    "severity": "error"
                })

    def check_readme(self) -> None:
        """Check README.md exists and has minimum content."""
        readme_file = self.repo_path / "README.md"

        if not readme_file.exists():
            self.violations.append({
                "check": "readme",
                "message": "Missing README.md file",
                "severity": "error"
            })
            return

        try:
            content = readme_file.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            self.violations.append({
                "check": "readme",
                "message": f"Cannot read README.md: {e}",
                "severity": "error"
            })
            return
            
        if len(content) < 50:
            self.warnings.append({
                "check": "readme",
                "message": "README.md is too short (less than 50 characters)",
                "severity": "warning"
            })

    def check_docker(self) -> None:
        """Check Dockerfile for security best practices."""
        dockerfile = self.repo_path / "Dockerfile"
        if not dockerfile.exists():
            return

        try:
            content = dockerfile.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            self.violations.append({
                "check": "docker",
                "message": f"Cannot read Dockerfile: {e}",
                "severity": "error"
            })
            return
            
        lines = content.split("\n")

        # Check for :latest tag
        for line in lines:
            if line.strip().startswith("FROM") and ":latest" in line:
                self.violations.append({
                    "check": "docker",
                    "message": "Using :latest tag in FROM instruction is not recommended",
                    "severity": "error"
                })

        # Check for non-root user
        has_user = any(line.strip().startswith("USER") and "root" not in line.lower()
                       for line in lines)
        if not has_user:
            self.violations.append({
                "check": "docker",
                "message": "Dockerfile should specify a non-root USER",
                "severity": "error"
            })

        # Check for secrets in ENV
        for line in lines:
            if line.strip().startswith("ENV"):
                for pattern in self.SECRETS_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        self.violations.append({
                            "check": "docker",
                            "message": f"Potential secrets in ENV instruction: {pattern}",
                            "severity": "error"
                        })
                        break

        # Check for HEALTHCHECK
        has_healthcheck = any(line.strip().startswith("HEALTHCHECK") for line in lines)
        if not has_healthcheck:
            self.warnings.append({
                "check": "docker",
                "message": "Dockerfile should include a HEALTHCHECK instruction",
                "severity": "warning"
            })

    def check_workflows(self) -> None:
        """Check GitHub Actions workflows for best practices."""
        workflows_dir = self.repo_path / ".github" / "workflows"
        if not workflows_dir.exists():
            return

        for workflow_file in workflows_dir.glob("*.yml"):
            try:
                with open(workflow_file, encoding='utf-8') as f:
                    workflow = yaml.safe_load(f)
            except (yaml.YAMLError, OSError, UnicodeDecodeError):
                self.warnings.append({
                    "check": "workflow",
                    "message": f"Invalid YAML in {workflow_file.name}",
                    "severity": "warning"
                })
                continue

            if not workflow:
                continue

            # Check for permissions block
            if "permissions" not in workflow:
                self.warnings.append({
                    "check": "workflow",
                    "message": f"{workflow_file.name}: Missing permissions block",
                    "severity": "warning"
                })

            # Check for continue-on-error
            jobs = workflow.get("jobs", {})
            for job_name, job in jobs.items():
                steps = job.get("steps", [])
                for step in steps:
                    if step.get("continue-on-error"):
                        self.warnings.append({
                            "check": "workflow",
                            "message": f"{workflow_file.name}: continue-on-error in job '{job_name}'",
                            "severity": "warning"
                        })

    def check_codeowners(self) -> None:
        """Check CODEOWNERS file exists."""
        codeowners = self.repo_path / ".github" / "CODEOWNERS"
        if not codeowners.exists():
            self.warnings.append({
                "check": "codeowners",
                "message": "Missing .github/CODEOWNERS file",
                "severity": "warning"
            })

    def report(self, fmt: str = "text") -> str:
        """
        Generate enforcement report.

        Args:
            fmt: Output format ('text', 'markdown', or 'json')

        Returns:
            Formatted report string
        """
        if fmt == "json":
            return json.dumps({
                "repository": str(self.repo_path),
                "violations": self.violations,
                "warnings": self.warnings,
                "summary": {
                    "violation_count": len(self.violations),
                    "warning_count": len(self.warnings),
                    "passed": len(self.violations) == 0
                }
            }, indent=2)

        lines = [
            "=" * 60,
            "Enforcement Report",
            "=" * 60,
            f"Repository: {self.repo_path}",
            "",
            f"Violations: {len(self.violations)}",
            f"Warnings: {len(self.warnings)}",
            ""
        ]

        if self.violations:
            lines.append("VIOLATIONS:")
            for v in self.violations:
                lines.append(f"  [{v['check']}] {v['message']}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  [{w['check']}] {w['message']}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


def enforce_organization(
    org_path: Path,
    central_path: Path = None
) -> Dict[str, Any]:
    """
    Enforce policies across all repositories in an organization.

    Args:
        org_path: Path to organization directory
        central_path: Path to central governance repo

    Returns:
        Enforcement results dictionary
    """
    org_path = Path(org_path)
    results = {
        "organization": org_path.name,
        "repositories": {},
        "summary": {
            "total_repos": 0,
            "passed": 0,
            "failed": 0,
            "total_violations": 0,
            "total_warnings": 0
        }
    }

    for repo_dir in org_path.iterdir():
        if not repo_dir.is_dir():
            continue

        enforcer = PolicyEnforcer(repo_dir, central_path=central_path)
        violations, warnings = enforcer.check_all()

        results["repositories"][repo_dir.name] = {
            "violations": violations,
            "warnings": warnings,
            "passed": violations == 0
        }

        results["summary"]["total_repos"] += 1
        results["summary"]["total_violations"] += violations
        results["summary"]["total_warnings"] += warnings

        if violations == 0:
            results["summary"]["passed"] += 1
        else:
            results["summary"]["failed"] += 1

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Policy enforcement")
    parser.add_argument("path", help="Repository or organization path")
    parser.add_argument("--org", action="store_true", help="Enforce on organization")
    parser.add_argument("--format", "-f", choices=["text", "json"], default="text")
    parser.add_argument("--central", help="Central repo path")

    args = parser.parse_args()

    if args.org:
        results = enforce_organization(Path(args.path), Path(args.central) if args.central else None)
        if args.format == "json":
            print(json.dumps(results, indent=2))
        else:
            print(f"Organization: {results['organization']}")
            print(f"Total Repos: {results['summary']['total_repos']}")
            print(f"Passed: {results['summary']['passed']}")
            print(f"Failed: {results['summary']['failed']}")
    else:
        enforcer = PolicyEnforcer(Path(args.path), Path(args.central) if args.central else None)
        enforcer.check_all()
        print(enforcer.report(fmt=args.format))
