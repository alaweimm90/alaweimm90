#!/usr/bin/env python3
"""
meta.py - Meta Auditor & Project Promotion System

Portfolio-level auditing and project promotion capabilities:
- Scan projects for governance gaps
- Promote projects to full repositories
- Cross-reference projects with repository inventory
- Generate compliance reports

Usage:
    python meta.py scan-projects                    # Scan all projects
    python meta.py scan-projects --org alaweimm90-tools
    python meta.py promote-project my-project --org alaweimm90-tools
    python meta.py audit --output audit-report.md
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import click
import yaml


class GapSeverity(Enum):
    """Severity levels for governance gaps."""
    P0 = "P0"  # Critical - blocks deployment
    P1 = "P1"  # High - should fix soon
    P2 = "P2"  # Medium - nice to have
    P3 = "P3"  # Low - informational


@dataclass
class GovernanceGap:
    """Represents a governance compliance gap."""
    severity: GapSeverity
    category: str
    message: str
    file_path: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class ProjectAudit:
    """Audit result for a single project."""
    name: str
    path: str
    organization: str
    language: str
    repo_type: str
    has_metadata: bool
    gaps: List[GovernanceGap] = field(default_factory=list)
    compliance_score: float = 0.0
    promotion_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "organization": self.organization,
            "language": self.language,
            "type": self.repo_type,
            "has_metadata": self.has_metadata,
            "gaps": [
                {
                    "severity": g.severity.value,
                    "category": g.category,
                    "message": g.message,
                    "file_path": g.file_path,
                    "recommendation": g.recommendation
                }
                for g in self.gaps
            ],
            "compliance_score": self.compliance_score,
            "promotion_ready": self.promotion_ready
        }


class MetaAuditor:
    """Portfolio-level auditor for governance compliance."""

    # Required files by tier
    TIER_REQUIREMENTS = {
        1: ["README.md", ".meta/repo.yaml", ".github/CODEOWNERS",
            ".github/workflows/ci.yml", "tests/", "LICENSE"],
        2: ["README.md", ".meta/repo.yaml", ".github/CODEOWNERS",
            ".github/workflows/ci.yml"],
        3: ["README.md", ".meta/repo.yaml"],
        4: ["README.md"]
    }

    # Language detection markers
    LANGUAGE_MARKERS = {
        "python": ["pyproject.toml", "setup.py", "requirements.txt"],
        "typescript": ["package.json", "tsconfig.json"],
        "go": ["go.mod"],
        "rust": ["Cargo.toml"],
        "java": ["pom.xml", "build.gradle"]
    }

    # Type inference from name prefixes
    TYPE_PREFIXES = {
        "lib-": "library",
        "tool-": "tool",
        "adapter-": "adapter",
        "demo-": "demo",
        "paper-": "research",
        "core-": "tool",
        "infra-": "tool",
        "template-": "demo"
    }

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        self.org_path = self.base_path / "organizations"
        self.audits: List[ProjectAudit] = []

    def _find_base_path(self) -> Path:
        """Find the central governance repo path."""
        if env_path := os.environ.get("GOLDEN_PATH_ROOT"):
            path = Path(env_path)
            if path.exists() and (path / ".metaHub").exists():
                return path

        current = Path.cwd()
        while current != current.parent:
            if (current / ".metaHub").exists():
                return current
            current = current.parent

        script_path = Path(__file__).resolve().parent.parent.parent
        if (script_path / ".metaHub").exists():
            return script_path

        raise RuntimeError("Could not find central governance repo (.metaHub directory)")

    def scan_all_projects(self, org_filter: Optional[str] = None) -> List[ProjectAudit]:
        """Scan all projects across organizations."""
        self.audits = []

        if not self.org_path.exists():
            raise RuntimeError(f"Organizations path not found: {self.org_path}")

        for org_dir in sorted(self.org_path.iterdir()):
            if not org_dir.is_dir() or org_dir.name.startswith('.'):
                continue

            if org_filter and org_dir.name != org_filter:
                continue

            self._scan_organization(org_dir)

        return self.audits

    def _scan_organization(self, org_dir: Path) -> None:
        """Scan all projects in an organization."""
        for project_dir in sorted(org_dir.iterdir()):
            if not project_dir.is_dir() or project_dir.name.startswith('.'):
                continue

            audit = self._audit_project(project_dir, org_dir.name)
            self.audits.append(audit)

    def _audit_project(self, project_dir: Path, org_name: str) -> ProjectAudit:
        """Audit a single project for governance compliance."""
        audit = ProjectAudit(
            name=project_dir.name,
            path=str(project_dir),
            organization=org_name,
            language=self._detect_language(project_dir),
            repo_type=self._infer_type(project_dir.name),
            has_metadata=(project_dir / ".meta" / "repo.yaml").exists()
        )

        # Check for governance gaps
        self._check_metadata_gaps(project_dir, audit)
        self._check_structure_gaps(project_dir, audit)
        self._check_ci_gaps(project_dir, audit)
        self._check_security_gaps(project_dir, audit)
        self._check_documentation_gaps(project_dir, audit)

        # Calculate compliance score
        audit.compliance_score = self._calculate_score(audit)
        audit.promotion_ready = self._is_promotion_ready(audit)

        return audit

    def _detect_language(self, project_dir: Path) -> str:
        """Detect primary language from project files."""
        for lang, markers in self.LANGUAGE_MARKERS.items():
            for marker in markers:
                if (project_dir / marker).exists():
                    return lang
        return "unknown"

    def _infer_type(self, name: str) -> str:
        """Infer project type from name prefix."""
        name_lower = name.lower()
        for prefix, repo_type in self.TYPE_PREFIXES.items():
            if name_lower.startswith(prefix):
                return repo_type
        return "unknown"

    def _check_metadata_gaps(self, project_dir: Path, audit: ProjectAudit) -> None:
        """Check for metadata-related gaps."""
        meta_file = project_dir / ".meta" / "repo.yaml"

        if not meta_file.exists():
            audit.gaps.append(GovernanceGap(
                severity=GapSeverity.P0,
                category="metadata",
                message=".meta/repo.yaml is missing",
                file_path=str(meta_file),
                recommendation="Create .meta/repo.yaml with type, language, tier fields"
            ))
            return

        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f) or {}
        except Exception as e:
            audit.gaps.append(GovernanceGap(
                severity=GapSeverity.P0,
                category="metadata",
                message=f"Invalid YAML in .meta/repo.yaml: {e}",
                file_path=str(meta_file)
            ))
            return

        # Check required fields
        required = ["type", "language"]
        for req_field in required:
            if req_field not in metadata:
                audit.gaps.append(GovernanceGap(
                    severity=GapSeverity.P0,
                    category="metadata",
                    message=f"Missing required field: {req_field}",
                    file_path=str(meta_file)
                ))

        # Check tier
        if "tier" not in metadata:
            audit.gaps.append(GovernanceGap(
                severity=GapSeverity.P1,
                category="metadata",
                message="Missing tier field",
                file_path=str(meta_file),
                recommendation="Add tier: 1-4 based on criticality"
            ))

    def _check_structure_gaps(self, project_dir: Path, audit: ProjectAudit) -> None:
        """Check for structure-related gaps."""
        # README check
        if not (project_dir / "README.md").exists():
            audit.gaps.append(GovernanceGap(
                severity=GapSeverity.P0,
                category="structure",
                message="README.md is missing",
                recommendation="Create README.md with project description"
            ))

        # LICENSE check
        if not (project_dir / "LICENSE").exists():
            audit.gaps.append(GovernanceGap(
                severity=GapSeverity.P1,
                category="structure",
                message="LICENSE file is missing",
                recommendation="Add LICENSE file (MIT recommended)"
            ))

        # Tests directory check
        has_tests = (project_dir / "tests").is_dir() or (project_dir / "test").is_dir()
        if not has_tests and audit.language in ["python", "typescript", "go", "rust"]:
            audit.gaps.append(GovernanceGap(
                severity=GapSeverity.P1,
                category="structure",
                message="No tests directory found",
                recommendation="Create tests/ directory with test files"
            ))

    def _check_ci_gaps(self, project_dir: Path, audit: ProjectAudit) -> None:
        """Check for CI/CD-related gaps."""
        workflows_dir = project_dir / ".github" / "workflows"

        if not workflows_dir.exists():
            audit.gaps.append(GovernanceGap(
                severity=GapSeverity.P1,
                category="ci",
                message="No CI/CD workflows found",
                recommendation="Add .github/workflows/ci.yml"
            ))
            return

        ci_files = list(workflows_dir.glob("ci*.yml")) + list(workflows_dir.glob("ci*.yaml"))
        if not ci_files:
            audit.gaps.append(GovernanceGap(
                severity=GapSeverity.P1,
                category="ci",
                message="No CI workflow found",
                recommendation="Add ci.yml workflow"
            ))

    def _check_security_gaps(self, project_dir: Path, audit: ProjectAudit) -> None:
        """Check for security-related gaps."""
        # CODEOWNERS check
        codeowners = project_dir / ".github" / "CODEOWNERS"
        if not codeowners.exists():
            audit.gaps.append(GovernanceGap(
                severity=GapSeverity.P1,
                category="security",
                message="CODEOWNERS file is missing",
                recommendation="Add .github/CODEOWNERS with ownership rules"
            ))

        # Check for secrets in code (basic check)
        for pattern in ["*.py", "*.ts", "*.js", "*.go", "*.rs"]:
            for file in project_dir.rglob(pattern):
                if file.is_file() and self._check_file_for_secrets(file):
                    audit.gaps.append(GovernanceGap(
                        severity=GapSeverity.P0,
                        category="security",
                        message="Potential hardcoded secret detected",
                        file_path=str(file),
                        recommendation="Remove hardcoded secrets, use environment variables"
                    ))
                    break  # Only report once per project

    def _check_file_for_secrets(self, file_path: Path) -> bool:
        """Basic check for hardcoded secrets."""
        # Skip test files, config templates, and documentation
        skip_patterns = ["test_", "_test.", ".test.", "spec.", "mock", "fixture", "example", "template", ".md"]
        file_name = file_path.name.lower()
        if any(p in file_name for p in skip_patterns):
            return False
        
        # Skip files in test directories
        path_str = str(file_path).lower()
        if any(d in path_str for d in ["/tests/", "/test/", "/__tests__/", "/spec/", "/fixtures/"]):
            return False
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Skip if file is mostly comments or documentation
            lines = content.split('\n')
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith(('#', '//', '/*', '*', '"""', "'''"))]
            if len(code_lines) < 5:
                return False
            
            secret_patterns = [
                "password=",
                "api_key=",
                "secret_key=",
                "private_key=",
                "AWS_SECRET",
                "GITHUB_TOKEN="
            ]
            content_lower = content.lower()
            for pattern in secret_patterns:
                if pattern.lower() in content_lower:
                    # Skip if it's just a variable reference or placeholder
                    if any(safe in content for safe in ["os.environ", "process.env", "getenv", "${", "{{", "<YOUR_", "xxx", "placeholder"]):
                        continue
                    return True
        except Exception:
            pass
        return False

    def _check_documentation_gaps(self, project_dir: Path, audit: ProjectAudit) -> None:
        """Check for documentation-related gaps."""
        readme = project_dir / "README.md"
        if readme.exists():
            try:
                content = readme.read_text(encoding='utf-8', errors='ignore')
                if len(content.strip()) < 100:
                    audit.gaps.append(GovernanceGap(
                        severity=GapSeverity.P2,
                        category="documentation",
                        message="README.md is too short",
                        recommendation="Expand README with installation, usage, and examples"
                    ))
            except Exception:
                pass

        # Check for CONTRIBUTING.md in larger projects
        if audit.repo_type in ["library", "tool"]:
            if not (project_dir / "CONTRIBUTING.md").exists():
                audit.gaps.append(GovernanceGap(
                    severity=GapSeverity.P2,
                    category="documentation",
                    message="CONTRIBUTING.md is missing",
                    recommendation="Add contribution guidelines"
                ))

    def _calculate_score(self, audit: ProjectAudit) -> float:
        """Calculate compliance score (0-100)."""
        if not audit.gaps:
            return 100.0

        # Weight by severity
        weights = {
            GapSeverity.P0: 25,
            GapSeverity.P1: 15,
            GapSeverity.P2: 5,
            GapSeverity.P3: 2
        }

        total_penalty = sum(weights.get(g.severity, 0) for g in audit.gaps)
        score = max(0, 100 - total_penalty)
        return round(score, 1)

    def _is_promotion_ready(self, audit: ProjectAudit) -> bool:
        """Check if project is ready for promotion to full repo."""
        # No P0 gaps allowed
        p0_gaps = [g for g in audit.gaps if g.severity == GapSeverity.P0]
        if p0_gaps:
            return False

        # Must have metadata
        if not audit.has_metadata:
            return False

        # Minimum score of 70
        if audit.compliance_score < 70:
            return False

        return True

    def generate_report(self, fmt: str = 'text') -> str:
        """Generate audit report."""
        if fmt == 'json':
            return json.dumps({
                "generated_at": datetime.now().isoformat(),
                "total_projects": len(self.audits),
                "summary": self._generate_summary(),
                "projects": [a.to_dict() for a in self.audits]
            }, indent=2)

        elif fmt == 'markdown':
            return self._generate_markdown_report()

        else:
            return self._generate_text_report()

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        total = len(self.audits)
        if total == 0:
            return {}

        promotion_ready = sum(1 for a in self.audits if a.promotion_ready)
        avg_score = sum(a.compliance_score for a in self.audits) / total

        gap_counts = {s.value: 0 for s in GapSeverity}
        for audit in self.audits:
            for gap in audit.gaps:
                gap_counts[gap.severity.value] += 1

        return {
            "total_projects": total,
            "promotion_ready": promotion_ready,
            "average_score": round(avg_score, 1),
            "gaps_by_severity": gap_counts
        }

    def _generate_text_report(self) -> str:
        """Generate text format report."""
        lines = [
            "=" * 60,
            "META AUDITOR - Portfolio Compliance Report",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            f"Total Projects: {len(self.audits)}",
            ""
        ]

        summary = self._generate_summary()
        if summary:
            lines.extend([
                "SUMMARY",
                "-" * 40,
                f"  Promotion Ready: {summary['promotion_ready']}/{summary['total_projects']}",
                f"  Average Score: {summary['average_score']}%",
                f"  P0 Gaps: {summary['gaps_by_severity']['P0']}",
                f"  P1 Gaps: {summary['gaps_by_severity']['P1']}",
                f"  P2 Gaps: {summary['gaps_by_severity']['P2']}",
                ""
            ])

        # Projects with issues
        projects_with_gaps = [a for a in self.audits if a.gaps]
        if projects_with_gaps:
            lines.append("PROJECTS WITH GAPS")
            lines.append("-" * 40)
            for audit in sorted(projects_with_gaps, key=lambda x: x.compliance_score):
                lines.append(f"\n  {audit.organization}/{audit.name}")
                lines.append(f"    Score: {audit.compliance_score}% | Type: {audit.repo_type} | Lang: {audit.language}")
                for gap in audit.gaps:
                    lines.append(f"    [{gap.severity.value}] {gap.category}: {gap.message}")

        return '\n'.join(lines)

    def _generate_markdown_report(self) -> str:
        """Generate markdown format report."""
        summary = self._generate_summary()

        lines = [
            "# Portfolio Compliance Audit Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Projects | {summary.get('total_projects', 0)} |",
            f"| Promotion Ready | {summary.get('promotion_ready', 0)} |",
            f"| Average Score | {summary.get('average_score', 0)}% |",
            f"| P0 (Critical) Gaps | {summary.get('gaps_by_severity', {}).get('P0', 0)} |",
            f"| P1 (High) Gaps | {summary.get('gaps_by_severity', {}).get('P1', 0)} |",
            "",
            "## Projects by Organization",
            ""
        ]

        # Group by organization
        by_org: Dict[str, List[ProjectAudit]] = {}
        for audit in self.audits:
            by_org.setdefault(audit.organization, []).append(audit)

        for org, audits in sorted(by_org.items()):
            lines.append(f"### {org}")
            lines.append("")
            lines.append("| Project | Score | Type | Language | Status |")
            lines.append("|---------|-------|------|----------|--------|")

            for audit in sorted(audits, key=lambda x: -x.compliance_score):
                status = "Ready" if audit.promotion_ready else f"{len(audit.gaps)} gaps"
                lines.append(
                    f"| {audit.name} | {audit.compliance_score}% | "
                    f"{audit.repo_type} | {audit.language} | {status} |"
                )

            lines.append("")

        # Critical gaps section
        critical_gaps = [
            (a, g) for a in self.audits for g in a.gaps
            if g.severity == GapSeverity.P0
        ]

        if critical_gaps:
            lines.extend([
                "## Critical Gaps (P0)",
                "",
                "These must be fixed before deployment:",
                ""
            ])
            for audit, gap in critical_gaps:
                lines.append(f"- **{audit.organization}/{audit.name}**: {gap.message}")
                if gap.recommendation:
                    lines.append(f"  - Recommendation: {gap.recommendation}")

        return '\n'.join(lines)


class ProjectPromoter:
    """Promotes projects to full repository status."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        self.templates_path = self.base_path / ".metaHub" / "templates"

    def _find_base_path(self) -> Path:
        """Find the central governance repo path."""
        if env_path := os.environ.get("GOLDEN_PATH_ROOT"):
            path = Path(env_path)
            if path.exists() and (path / ".metaHub").exists():
                return path

        current = Path.cwd()
        while current != current.parent:
            if (current / ".metaHub").exists():
                return current
            current = current.parent

        raise RuntimeError("Could not find central governance repo")

    def promote_project(self, project_path: Path, org_name: str,
                       dry_run: bool = False) -> Dict[str, Any]:
        """Promote a project to full repository status."""
        result = {
            "project": project_path.name,
            "organization": org_name,
            "actions": [],
            "success": True
        }

        if not project_path.exists():
            result["success"] = False
            result["error"] = f"Project path not found: {project_path}"
            return result

        # Detect language
        language = self._detect_language(project_path)
        repo_type = self._infer_type(project_path.name)

        # Create .meta/repo.yaml if missing
        meta_file = project_path / ".meta" / "repo.yaml"
        if not meta_file.exists():
            action = self._create_metadata(project_path, org_name, language, repo_type, dry_run)
            result["actions"].append(action)

        # Create .github/CODEOWNERS if missing
        codeowners = project_path / ".github" / "CODEOWNERS"
        if not codeowners.exists():
            action = self._create_codeowners(project_path, org_name, dry_run)
            result["actions"].append(action)

        # Create CI workflow if missing
        ci_file = project_path / ".github" / "workflows" / "ci.yml"
        if not ci_file.exists():
            action = self._create_ci_workflow(project_path, language, dry_run)
            result["actions"].append(action)

        # Create pre-commit config if missing
        precommit = project_path / ".pre-commit-config.yaml"
        if not precommit.exists():
            action = self._create_precommit(project_path, language, dry_run)
            result["actions"].append(action)

        return result

    def _detect_language(self, project_path: Path) -> str:
        """Detect primary language."""
        markers = {
            "python": ["pyproject.toml", "setup.py", "requirements.txt"],
            "typescript": ["package.json", "tsconfig.json"],
            "go": ["go.mod"],
            "rust": ["Cargo.toml"]
        }
        for lang, files in markers.items():
            for f in files:
                if (project_path / f).exists():
                    return lang
        return "unknown"

    def _infer_type(self, name: str) -> str:
        """Infer project type from name."""
        prefixes = {
            "lib-": "library",
            "tool-": "tool",
            "adapter-": "adapter",
            "demo-": "demo",
            "paper-": "research"
        }
        for prefix, ptype in prefixes.items():
            if name.lower().startswith(prefix):
                return ptype
        return "unknown"

    def _create_metadata(self, project_path: Path, org_name: str,
                        language: str, repo_type: str, dry_run: bool) -> Dict:
        """Create .meta/repo.yaml."""
        meta_dir = project_path / ".meta"
        meta_file = meta_dir / "repo.yaml"

        metadata = {
            "type": repo_type,
            "language": language,
            "tier": 3,
            "owner": org_name,
            "description": f"Repository: {project_path.name}",
            "status": "active",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "coverage": {"target": 70},
            "docs": {"profile": "minimal"}
        }

        if not dry_run:
            meta_dir.mkdir(parents=True, exist_ok=True)
            with open(meta_file, 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

        return {
            "action": "create",
            "file": str(meta_file),
            "dry_run": dry_run
        }

    def _create_codeowners(self, project_path: Path, org_name: str, dry_run: bool) -> Dict:
        """Create .github/CODEOWNERS."""
        codeowners_file = project_path / ".github" / "CODEOWNERS"

        content = f"""# Repository ownership and approval requirements

* @{org_name}

# Critical paths require additional review
/.github/workflows/ @{org_name}
/src/ @{org_name}
/tests/ @{org_name}
"""

        if not dry_run:
            codeowners_file.parent.mkdir(parents=True, exist_ok=True)
            with open(codeowners_file, 'w', encoding='utf-8') as f:
                f.write(content)

        return {
            "action": "create",
            "file": str(codeowners_file),
            "dry_run": dry_run
        }

    def _create_ci_workflow(self, project_path: Path, language: str, dry_run: bool) -> Dict:
        """Create .github/workflows/ci.yml."""
        ci_file = project_path / ".github" / "workflows" / "ci.yml"

        # Use reusable workflows from central repo
        if language == "python":
            workflow = """name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  python:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
"""
        elif language == "typescript":
            workflow = """name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  typescript:
    uses: alaweimm90/.github/.github/workflows/reusable-ts-ci.yml@main
    with:
      node-version: '20'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
"""
        else:
            workflow = """name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
"""

        if not dry_run:
            ci_file.parent.mkdir(parents=True, exist_ok=True)
            with open(ci_file, 'w', encoding='utf-8') as f:
                f.write(workflow)

        return {
            "action": "create",
            "file": str(ci_file),
            "dry_run": dry_run
        }

    def _create_precommit(self, project_path: Path, language: str, dry_run: bool) -> Dict:
        """Create .pre-commit-config.yaml from template."""
        precommit_file = project_path / ".pre-commit-config.yaml"

        # Find appropriate template
        template_name = f"{language}.yaml" if language in ["python", "typescript", "go", "rust"] else "generic.yaml"
        template_path = self.templates_path / "pre-commit" / template_name

        if template_path.exists():
            content = template_path.read_text(encoding='utf-8')
        else:
            # Fallback to generic
            generic_path = self.templates_path / "pre-commit" / "generic.yaml"
            if generic_path.exists():
                content = generic_path.read_text(encoding='utf-8')
            else:
                content = "# Pre-commit configuration\nrepos: []\n"

        if not dry_run:
            with open(precommit_file, 'w', encoding='utf-8') as f:
                f.write(content)

        return {
            "action": "create",
            "file": str(precommit_file),
            "template": template_name,
            "dry_run": dry_run
        }


# CLI Commands
@click.group()
def cli():
    """Meta Auditor - Portfolio governance and project promotion."""
    pass


@cli.command('scan-projects')
@click.option('--org', help='Filter by organization name')
@click.option('--format', 'fmt', type=click.Choice(['text', 'json', 'markdown']),
              default='text', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--min-score', type=float, default=0, help='Minimum compliance score to include')
def scan_projects(org: Optional[str], fmt: str, output: Optional[str], min_score: float):
    """Scan all projects for governance compliance."""
    try:
        auditor = MetaAuditor()
        click.echo(f"Scanning projects at: {auditor.org_path}")

        audits = auditor.scan_all_projects(org_filter=org)

        # Filter by minimum score
        if min_score > 0:
            auditor.audits = [a for a in audits if a.compliance_score >= min_score]

        report = auditor.generate_report(fmt=fmt)

        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(report)
            click.echo(f"Report written to: {output}")
        else:
            click.echo(report)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command('promote-project')
@click.argument('project_name')
@click.option('--org', required=True, help='Organization name')
@click.option('--dry-run', is_flag=True, help='Preview changes without writing')
def promote_project(project_name: str, org: str, dry_run: bool):
    """Promote a project to full repository status."""
    try:
        promoter = ProjectPromoter()
        project_path = promoter.base_path / "organizations" / org / project_name

        if not project_path.exists():
            click.echo(f"Error: Project not found: {project_path}", err=True)
            raise SystemExit(1)

        click.echo(f"Promoting project: {org}/{project_name}")
        if dry_run:
            click.echo("[DRY-RUN] No files will be modified")

        result = promoter.promote_project(project_path, org, dry_run=dry_run)

        if result["success"]:
            click.echo("\nActions performed:")
            for action in result["actions"]:
                status = "[DRY-RUN]" if action.get("dry_run") else "[OK]"
                click.echo(f"  {status} {action['action']}: {action['file']}")
            click.echo("\n[SUCCESS] Project promotion complete")
        else:
            click.echo(f"\n[FAILED] {result.get('error', 'Unknown error')}", err=True)
            raise SystemExit(1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command('audit')
@click.option('--output', '-o', type=click.Path(), default='audit-report.md',
              help='Output file path')
@click.option('--format', 'fmt', type=click.Choice(['text', 'json', 'markdown']),
              default='markdown', help='Output format')
def audit(output: str, fmt: str):
    """Generate comprehensive audit report."""
    try:
        auditor = MetaAuditor()
        click.echo("Running portfolio audit...")

        auditor.scan_all_projects()
        report = auditor.generate_report(fmt=fmt)

        with open(output, 'w', encoding='utf-8') as f:
            f.write(report)

        click.echo(f"Audit report written to: {output}")

        # Print summary
        summary = auditor._generate_summary()
        click.echo("\nSummary:")
        click.echo(f"  Total Projects: {summary.get('total_projects', 0)}")
        click.echo(f"  Promotion Ready: {summary.get('promotion_ready', 0)}")
        click.echo(f"  Average Score: {summary.get('average_score', 0)}%")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == '__main__':
    cli()
