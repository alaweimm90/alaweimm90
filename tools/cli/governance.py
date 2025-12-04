#!/usr/bin/env python3
"""Unified Governance CLI - Validation, enforcement, and compliance."""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import click
import yaml

# Import shared libraries
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.checkpoint import CheckpointManager
from lib.validation import Validator
from lib.telemetry import Telemetry

# Import governance modules from .metaHub/scripts
metahub_scripts = Path(__file__).parent.parent.parent / ".metaHub" / "scripts"
sys.path.insert(0, str(metahub_scripts))
sys.path.insert(0, str(metahub_scripts / "compliance"))
sys.path.insert(0, str(metahub_scripts / "utils"))

from enforce import PolicyEnforcer
from catalog import CatalogBuilder
from meta import MetaAuditor, ProjectPromoter

# Import stubs for modules that may not exist yet
from governance_stubs import (
    enforce_organization,
    AIGovernanceAuditor,
    generate_markdown_report,
    GovernanceSyncer,
)


@click.group()
@click.version_option(version='2.0.0')
def cli():
    """Unified Governance CLI - Validation, enforcement, and compliance"""
    pass


@cli.command('enforce')
@click.argument('path', type=click.Path(exists=True))
@click.option('--strict', is_flag=True, help='Treat warnings as violations')
@click.option('--report', 'report_fmt', type=click.Choice(['text', 'json']),
              default='text', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--fail-on-warnings', is_flag=True, help='Exit with error if warnings exist')
@click.option('--schema', type=click.Path(exists=True), help='Path to JSON schema')
def cmd_enforce(path: str, strict: bool, report_fmt: str, output: Optional[str],
                fail_on_warnings: bool, schema: Optional[str]):
    """
    Enforce Golden Path governance policies.

    PATH can be a single repository or an organization directory containing multiple repos.

    Examples:

        python governance.py enforce ./organizations/my-org/

        python governance.py enforce ./organizations/my-org/ --report json --output results.json

        python governance.py enforce ./my-repo --strict --fail-on-warnings
    """
    telemetry = Telemetry()
    telemetry.start_operation("enforce")

    try:
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
                    "\nSummary:",
                    f"  Total Repos:      {results['summary']['total_repos']}",
                    f"  Passed:           {results['summary']['passed']}",
                    f"  Failed:           {results['summary']['failed']}",
                    f"  Total Violations: {results['summary']['total_violations']}",
                    f"  Total Warnings:   {results['summary']['total_warnings']}",
                ]

                # List failed repos
                failed_repos = [r for r in results['repos'] if not r['passed']]
                if failed_repos:
                    lines.append("\n[FAILED REPOS]")
                    for repo in failed_repos:
                        lines.append(f"  - {repo['name']}: {len(repo['violations'])} violations")

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

        telemetry.end_operation("enforce", success=exit_code == 0)
        sys.exit(exit_code)

    except Exception as e:
        telemetry.end_operation("enforce", success=False, error=str(e))
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('checkpoint')
@click.option('--baseline', is_flag=True, help='Create new baseline (skip comparison)')
@click.option('--report', 'report_fmt', type=click.Choice(['text', 'markdown', 'json']),
              default='text', help='Report format')
@click.option('--output', '-o', type=click.Path(), help='Output file for report')
@click.option('--checkpoint', type=click.Path(exists=True),
              help='Specific checkpoint file to compare against')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress output')
def cmd_checkpoint(baseline: bool, report_fmt: str, output: Optional[str],
                   checkpoint: Optional[str], quiet: bool):
    """
    Detect drift between governance checkpoints.

    Creates a snapshot of current repository states and compares against
    the previous checkpoint to identify changes.

    Examples:

        python governance.py checkpoint                     # Normal drift detection

        python governance.py checkpoint --baseline          # Create new baseline

        python governance.py checkpoint --report markdown   # Markdown report
    """
    telemetry = Telemetry()
    telemetry.start_operation("checkpoint")

    try:
        # Import checkpoint module for drift detection
        from checkpoint import CheckpointManager as GovCheckpointManager

        mgr = GovCheckpointManager()

        if not quiet:
            click.echo("Generating current state snapshot...")

        mgr.generate_current_state()

        if not baseline:
            if checkpoint:
                loaded = mgr.load_previous_checkpoint(Path(checkpoint))
            else:
                loaded = mgr.load_previous_checkpoint()

            if not loaded and not quiet:
                click.echo("No previous checkpoint found - this will be the baseline")

            if not quiet:
                click.echo("Detecting drift...")

            mgr.detect_drift()

        # Save checkpoint
        checkpoint_file = mgr.save_checkpoint()
        if not quiet:
            click.echo(f"Checkpoint saved: {checkpoint_file}")

        # Generate report
        report = mgr.generate_report(fmt=report_fmt)

        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(report)
            if not quiet:
                click.echo(f"Report written to: {output}")
        else:
            click.echo(report)

        # Also save drift report
        if mgr.drift.get("has_drift"):
            drift_file = mgr.checkpoint_dir / f"drift-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            with open(drift_file, 'w', encoding='utf-8') as f:
                json.dump(mgr.drift, f, indent=2)
            if not quiet:
                click.echo(f"Drift report saved: {drift_file}")

        telemetry.end_operation("checkpoint", success=True)
        sys.exit(0)

    except Exception as e:
        telemetry.end_operation("checkpoint", success=False, error=str(e))
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('catalog')
@click.option('--org-path', type=click.Path(exists=True),
              help='Path to organizations directory')
@click.option('--format', 'fmt', type=click.Choice(['json', 'markdown', 'html']),
              default='json', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress output')
def cmd_catalog(org_path: Optional[str], fmt: str, output: Optional[str], quiet: bool):
    """
    Generate a service catalog from repository metadata.

    Scans all organizations and repositories, reading .meta/repo.yaml
    files to build a comprehensive catalog.

    Examples:

        python governance.py catalog

        python governance.py catalog --format markdown --output catalog.md

        python governance.py catalog --format html --output service-catalog.html
    """
    telemetry = Telemetry()
    telemetry.start_operation("catalog")

    try:
        builder = CatalogBuilder(org_path=org_path)

        if not quiet:
            click.echo(f"Scanning organizations at: {builder.org_path}")

        catalog = builder.scan_organizations()

        if not quiet:
            click.echo(f"\nFound {catalog['summary']['total_organizations']} organizations")
            click.echo(f"Found {catalog['summary']['total_repositories']} repositories")

        # Generate output
        output_path = Path(output) if output else None

        if fmt == 'json':
            result = builder.generate_json(output_path)
            default_file = builder.base_path / ".metaHub" / "catalog" / "catalog.json"
        elif fmt == 'markdown':
            result = builder.generate_markdown(output_path)
            default_file = builder.base_path / ".metaHub" / "catalog" / "catalog.md"
        else:  # html
            result = builder.generate_html(output_path)
            default_file = builder.base_path / ".metaHub" / "catalog" / "catalog.html"

        # Write to default location if no output specified
        if not output_path:
            default_file.parent.mkdir(parents=True, exist_ok=True)
            with open(default_file, 'w', encoding='utf-8') as f:
                f.write(result)
            if not quiet:
                click.echo(f"\nCatalog written to: {default_file}")
        else:
            if not quiet:
                click.echo(f"\nCatalog written to: {output_path}")

        telemetry.end_operation("catalog", success=True)
        sys.exit(0)

    except Exception as e:
        telemetry.end_operation("catalog", success=False, error=str(e))
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.group('meta')
def meta_group():
    """Repository metadata management and auditing"""
    pass


@meta_group.command('scan')
@click.option('--org', help='Filter by organization name')
@click.option('--format', 'fmt', type=click.Choice(['text', 'json', 'markdown']),
              default='text', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--min-score', type=float, default=0, help='Minimum compliance score to include')
def meta_scan(org: Optional[str], fmt: str, output: Optional[str], min_score: float):
    """Scan all projects for governance compliance."""
    telemetry = Telemetry()
    telemetry.start_operation("meta_scan")

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

        telemetry.end_operation("meta_scan", success=True)

    except Exception as e:
        telemetry.end_operation("meta_scan", success=False, error=str(e))
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@meta_group.command('promote')
@click.argument('project_name')
@click.option('--org', required=True, help='Organization name')
@click.option('--dry-run', is_flag=True, help='Preview changes without writing')
def meta_promote(project_name: str, org: str, dry_run: bool):
    """Promote a project to full repository status."""
    telemetry = Telemetry()
    telemetry.start_operation("meta_promote")

    try:
        promoter = ProjectPromoter()
        project_path = promoter.base_path / "organizations" / org / project_name

        if not project_path.exists():
            click.echo(f"Error: Project not found: {project_path}", err=True)
            sys.exit(1)

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
            telemetry.end_operation("meta_promote", success=True)
        else:
            click.echo(f"\n[FAILED] {result.get('error', 'Unknown error')}", err=True)
            telemetry.end_operation("meta_promote", success=False)
            sys.exit(1)

    except Exception as e:
        telemetry.end_operation("meta_promote", success=False, error=str(e))
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('audit')
@click.argument("path", type=click.Path(), default=".")
@click.option("--deep", is_flag=True, help="Run deep analysis")
@click.option("--report", "report_format", type=click.Choice(["json", "markdown", "summary"]),
              default="summary")
@click.option("--output", "-o", type=click.Path(), help="Output file")
def cmd_audit(path: str, deep: bool, report_format: str, output: str):
    """
    Run AI-powered governance audit.

    Examples:

        python governance.py audit

        python governance.py audit --report markdown --output audit.md

        python governance.py audit organizations/my-org --deep
    """
    telemetry = Telemetry()
    telemetry.start_operation("audit")

    try:
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

        telemetry.end_operation("audit", success=True)

    except Exception as e:
        telemetry.end_operation("audit", success=False, error=str(e))
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('sync')
@click.option('--org', help='Specific organization to sync')
@click.option('--all', 'sync_all', is_flag=True, help='Sync all organizations')
@click.option('--dry-run', is_flag=True, help='Show what would be synced')
def cmd_sync(org: Optional[str], sync_all: bool, dry_run: bool):
    """
    Sync governance rules to repositories.

    Examples:

        python governance.py sync --org my-org

        python governance.py sync --all

        python governance.py sync --all --dry-run
    """
    telemetry = Telemetry()
    telemetry.start_operation("sync")

    try:
        syncer = GovernanceSyncer()

        if dry_run:
            click.echo("DRY RUN - No changes will be made")
            telemetry.end_operation("sync", success=True)
            return

        if org:
            results = syncer.sync_organization(org)
            successful = results["successful_syncs"]
            total = results["total_repos"]
            click.echo(f"Organization {org}: {successful}/{total} repos synced")
        elif sync_all:
            results = syncer.sync_all_organizations()
            successful = results["total_successful"]
            total = results["total_repos"]
            click.echo(f"All organizations: {successful}/{total} repos synced")
        else:
            click.echo("Error: Must specify --org or --all", err=True)
            click.echo("Use --help for usage information")
            sys.exit(1)

        telemetry.end_operation("sync", success=True)

    except Exception as e:
        telemetry.end_operation("sync", success=False, error=str(e))
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
