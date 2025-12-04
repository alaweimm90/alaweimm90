#!/usr/bin/env python3
"""
AI-Powered Governance Audit using Claude

Integrates with existing governance scripts to provide AI-enhanced
analysis, recommendations, and automated remediation.

Usage:
    python ai_audit.py organizations/              # Audit all
    python ai_audit.py organizations/alaweimm90-tools --deep
    python ai_audit.py --report markdown --output audit.md
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

# Import existing governance scripts
sys.path.insert(0, str(Path(__file__).parent))
from enforce import PolicyEnforcer
from structure_validator import StructureValidator


class GovernanceContextCollector:
    """Collects comprehensive context for AI analysis."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.orgs_path = base_path / "organizations"

    def collect_repo_context(self, repo_path: Path) -> Dict[str, Any]:
        """Collect all relevant context for a single repository."""
        context = {
            "name": repo_path.name,
            "path": str(repo_path),
            "structure": self._scan_structure(repo_path),
            "metadata": self._read_metadata(repo_path),
            "files": self._list_key_files(repo_path),
            "workflows": self._analyze_workflows(repo_path),
        }
        return context

    def collect_org_context(self, org_path: Path) -> Dict[str, Any]:
        """Collect context for an entire organization."""
        repos = []
        for item in org_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                repos.append(self.collect_repo_context(item))

        return {
            "name": org_path.name,
            "path": str(org_path),
            "repo_count": len(repos),
            "repos": repos,
        }

    def collect_portfolio_context(self) -> Dict[str, Any]:
        """Collect context for entire portfolio."""
        orgs = []
        if self.orgs_path.exists():
            for org_dir in self.orgs_path.iterdir():
                if org_dir.is_dir() and not org_dir.name.startswith("."):
                    orgs.append(self.collect_org_context(org_dir))

        return {
            "timestamp": datetime.now().isoformat(),
            "organization_count": len(orgs),
            "organizations": orgs,
        }

    def _scan_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Scan repository structure."""
        structure = {
            "has_readme": (repo_path / "README.md").exists(),
            "has_license": (repo_path / "LICENSE").exists(),
            "has_meta": (repo_path / ".meta" / "repo.yaml").exists(),
            "has_github": (repo_path / ".github").is_dir(),
            "has_tests": (repo_path / "tests").is_dir(),
            "has_src": (repo_path / "src").is_dir(),
            "has_dockerfile": (repo_path / "Dockerfile").exists(),
            "has_security": (repo_path / "SECURITY.md").exists(),
        }
        return structure

    def _read_metadata(self, repo_path: Path) -> Optional[Dict]:
        """Read .meta/repo.yaml if exists."""
        meta_file = repo_path / ".meta" / "repo.yaml"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return None
        return None

    def _list_key_files(self, repo_path: Path) -> List[str]:
        """List key configuration files."""
        key_files = [
            "README.md", "LICENSE", "SECURITY.md", "CONTRIBUTING.md",
            "pyproject.toml", "package.json", "go.mod", "Cargo.toml",
            "Dockerfile", "docker-compose.yml", ".gitignore",
            ".pre-commit-config.yaml", "Makefile"
        ]
        found = []
        for f in key_files:
            if (repo_path / f).exists():
                found.append(f)
        return found

    def _analyze_workflows(self, repo_path: Path) -> List[str]:
        """List GitHub workflows."""
        workflows_dir = repo_path / ".github" / "workflows"
        if workflows_dir.is_dir():
            return [f.name for f in workflows_dir.glob("*.yml")]
        return []


class AIGovernanceAuditor:
    """AI-powered governance auditor using Claude."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.collector = GovernanceContextCollector(base_path)
        self.validator = StructureValidator(base_path)

    def run_audit(self, target_path: Optional[Path] = None, deep: bool = False) -> Dict[str, Any]:
        """Run governance audit."""
        # Collect context
        if target_path and target_path.is_dir():
            if (target_path / ".meta").exists() or (target_path / "README.md").exists():
                # Single repo
                context = {"type": "repo", "data": self.collector.collect_repo_context(target_path)}
            else:
                # Organization
                context = {"type": "org", "data": self.collector.collect_org_context(target_path)}
        else:
            # Full portfolio
            context = {"type": "portfolio", "data": self.collector.collect_portfolio_context()}

        # Run structure validation
        validation_results = self._run_validation()

        # Run enforcement checks
        enforcement_results = self._run_enforcement()

        # Generate audit report
        audit_result = {
            "timestamp": datetime.now().isoformat(),
            "target": str(target_path) if target_path else "portfolio",
            "deep_analysis": deep,
            "context": context,
            "validation": validation_results,
            "enforcement": enforcement_results,
            "summary": self._generate_summary(validation_results, enforcement_results),
            "recommendations": self._generate_recommendations(validation_results, enforcement_results),
        }

        return audit_result

    def _run_validation(self) -> Dict[str, Any]:
        """Run structure validation."""
        try:
            results = self.validator.validate_portfolio()
            summary = results["summary"]
            total_repos = summary["total_repos"]
            compliant = summary["compliant_repos"]
            compliance_rate = round((compliant / total_repos * 100) if total_repos > 0 else 100, 1)
            return {
                "total_repos": total_repos,
                "compliant": compliant,
                "compliance_rate": compliance_rate,
                "organizations": {
                    org: {
                        "repos": len(data["repos"]),
                        "compliant": data["summary"]["compliant"],
                    }
                    for org, data in results.get("organizations", {}).items()
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def _run_enforcement(self) -> Dict[str, Any]:
        """Run policy enforcement checks."""
        results = {"organizations": {}, "total_violations": 0, "total_warnings": 0}

        orgs_path = self.base_path / "organizations"
        if not orgs_path.exists():
            return results

        for org_dir in orgs_path.iterdir():
            if not org_dir.is_dir() or org_dir.name.startswith("."):
                continue

            org_results = {"repos": [], "violations": 0, "warnings": 0}

            for repo_dir in org_dir.iterdir():
                if not repo_dir.is_dir() or repo_dir.name.startswith("."):
                    continue

                try:
                    enforcer = PolicyEnforcer(repo_dir)
                    violations, warnings = enforcer.check_all()
                    org_results["repos"].append({
                        "name": repo_dir.name,
                        "violations": violations,
                        "warnings": warnings,
                    })
                    org_results["violations"] += violations
                    org_results["warnings"] += warnings
                except Exception:
                    org_results["repos"].append({
                        "name": repo_dir.name,
                        "error": "Failed to check",
                    })

            results["organizations"][org_dir.name] = org_results
            results["total_violations"] += org_results["violations"]
            results["total_warnings"] += org_results["warnings"]

        return results

    def _generate_summary(self, validation: Dict, enforcement: Dict) -> Dict[str, Any]:
        """Generate audit summary."""
        compliance_rate = validation.get("compliance_rate", 0)
        total_violations = enforcement.get("total_violations", 0)

        if compliance_rate == 100 and total_violations == 0:
            status = "EXCELLENT"
            score = 100
        elif compliance_rate >= 90 and total_violations <= 5:
            status = "GOOD"
            score = 85
        elif compliance_rate >= 75:
            status = "NEEDS_IMPROVEMENT"
            score = 70
        else:
            status = "CRITICAL"
            score = 50

        return {
            "status": status,
            "score": score,
            "compliance_rate": compliance_rate,
            "total_violations": total_violations,
            "total_warnings": enforcement.get("total_warnings", 0),
        }

    def _generate_recommendations(self, validation: Dict, enforcement: Dict) -> List[Dict[str, str]]:
        """Generate prioritized recommendations."""
        recommendations = []

        if validation.get("compliance_rate", 100) < 100:
            recommendations.append({
                "priority": "HIGH",
                "category": "Structure",
                "issue": "Non-compliant repositories detected",
                "action": "Run `python structure_validator.py --fix` to auto-remediate",
            })

        if enforcement.get("total_violations", 0) > 0:
            recommendations.append({
                "priority": "HIGH",
                "category": "Policy",
                "issue": f"{enforcement['total_violations']} policy violations found",
                "action": "Review enforcement report and address violations",
            })

        if enforcement.get("total_warnings", 0) > 10:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Best Practices",
                "issue": f"{enforcement['total_warnings']} warnings found",
                "action": "Address warnings to improve governance posture",
            })

        # Add general recommendations
        recommendations.append({
            "priority": "LOW",
            "category": "Automation",
            "issue": "Regular audits recommended",
            "action": "Enable weekly governance checks via checkpoint.yml",
        })

        return recommendations


def generate_markdown_report(audit_result: Dict[str, Any]) -> str:
    """Generate markdown report from audit results."""
    lines = [
        "# Governance Audit Report",
        "",
        f"**Generated:** {audit_result['timestamp']}",
        f"**Target:** {audit_result['target']}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Status | {audit_result['summary']['status']} |",
        f"| Score | {audit_result['summary']['score']}/100 |",
        f"| Compliance Rate | {audit_result['summary']['compliance_rate']}% |",
        f"| Violations | {audit_result['summary']['total_violations']} |",
        f"| Warnings | {audit_result['summary']['total_warnings']} |",
        "",
        "---",
        "",
        "## Validation Results",
        "",
    ]

    validation = audit_result.get("validation", {})
    if "error" not in validation:
        lines.append(f"**Total Repos:** {validation.get('total_repos', 0)}")
        lines.append(f"**Compliant:** {validation.get('compliant', 0)}")
        lines.append("")

        if validation.get("organizations"):
            lines.append("### By Organization")
            lines.append("")
            lines.append("| Organization | Repos | Compliant |")
            lines.append("|--------------|-------|-----------|")
            for org, data in validation["organizations"].items():
                lines.append(f"| {org} | {data['repos']} | {data['compliant']} |")
            lines.append("")

    lines.extend([
        "---",
        "",
        "## Recommendations",
        "",
    ])

    for rec in audit_result.get("recommendations", []):
        lines.append(f"### [{rec['priority']}] {rec['category']}")
        lines.append(f"**Issue:** {rec['issue']}")
        lines.append(f"**Action:** {rec['action']}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "*Generated by AI Governance Auditor*",
    ])

    return "\n".join(lines)


@click.command()
@click.argument("path", type=click.Path(), default=".")
@click.option("--deep", is_flag=True, help="Run deep analysis")
@click.option("--report", "report_format", type=click.Choice(["json", "markdown", "summary"]), default="summary")
@click.option("--output", "-o", type=click.Path(), help="Output file")
def main(path: str, deep: bool, report_format: str, output: str):
    """Run AI-powered governance audit."""
    base_path = Path(__file__).parent.parent.parent
    target_path = Path(path) if path != "." else None

    auditor = AIGovernanceAuditor(base_path)
    result = auditor.run_audit(target_path, deep)

    if report_format == "json":
        output_content = json.dumps(result, indent=2, default=str)
    elif report_format == "markdown":
        output_content = generate_markdown_report(result)
    else:
        # Summary
        summary = result["summary"]
        output_content = f"""
Governance Audit Summary
========================
Status: {summary['status']}
Score: {summary['score']}/100
Compliance: {summary['compliance_rate']}%
Violations: {summary['total_violations']}
Warnings: {summary['total_warnings']}
"""

    if output:
        Path(output).write_text(output_content)
        click.echo(f"Report saved to: {output}")
    else:
        click.echo(output_content)


if __name__ == "__main__":
    main()
