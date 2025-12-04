#!/usr/bin/env python3
"""
Unified Governance Compliance Validator

Validates compliance across:
- Organizations directory (original functionality)
- DevOps CLI patterns
- SuperTool cross-reference integrity
- Cross-project governance alignment

Usage:
    python compliance_validator.py                    # Full validation
    python compliance_validator.py --organizations   # Organizations only
    python compliance_validator.py --devops          # DevOps CLI only
    python compliance_validator.py --supertool       # SuperTool refs only
    python compliance_validator.py --unified         # Cross-project report
    python compliance_validator.py --json            # JSON output
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =============================================================================
# CONFIGURATION
# =============================================================================

SUPERTOOL_PATH = Path("C:/Users/mesha/Desktop/Projects/SuperTool")
GITHUB_REPO_PATH = Path("C:/Users/mesha/Desktop/GitHub")

DEVOPS_CLI_MODULES = [
    "tools/devops/builder.ts",
    "tools/devops/coder.ts",
    "tools/devops/config.ts",
    "tools/devops/fs.ts",
    "tools/devops/install.ts",
    "tools/devops/bootstrap.ts",
]

DEVOPS_TEMPLATE_CATEGORIES = [
    "cicd",
    "db",
    "demos",
    "iac",
    "k8s",
    "logging",
    "monitoring",
    "ui",
]

REQUIRED_SUPERTOOL_REFS = [
    ".metaHub/references/supertool/SUMMARY.md",
    ".metaHub/references/supertool/devops-validation-rules.yaml",
]

# =============================================================================
# ORIGINAL ORGANIZATION VALIDATION
# =============================================================================

def validate_metadata(repo_path: Path) -> Tuple[bool, Dict[str, bool]]:
    """Validate .meta/repo.yaml exists and is valid."""
    meta_file = repo_path / ".meta" / "repo.yaml"
    results = {
        "metadata_exists": meta_file.exists(),
    }
    return meta_file.exists(), results


def validate_documentation(repo_path: Path) -> Tuple[bool, Dict[str, bool]]:
    """Validate required documentation files."""
    required_files = [
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "LICENSE",
    ]
    results = {
        f"{file}_exists": (repo_path / file).exists()
        for file in required_files
    }
    all_exist = all(results.values())
    return all_exist, results


def validate_github_templates(repo_path: Path) -> Tuple[bool, Dict[str, bool]]:
    """Validate GitHub templates."""
    github_dir = repo_path / ".github"
    issue_template_dir = github_dir / "ISSUE_TEMPLATE"

    results = {
        "issue_template_dir_exists": issue_template_dir.exists(),
        "bug_report_template_exists": (issue_template_dir / "bug_report.md").exists() if issue_template_dir.exists() else False,
        "feature_request_template_exists": (issue_template_dir / "feature_request.md").exists() if issue_template_dir.exists() else False,
        "pr_template_exists": (github_dir / "PULL_REQUEST_TEMPLATE.md").exists(),
        "codeowners_exists": (github_dir / "CODEOWNERS").exists(),
    }
    all_exist = all(results.values())
    return all_exist, results


def validate_ci_cd(repo_path: Path) -> Tuple[bool, Dict[str, bool]]:
    """Validate CI/CD configuration."""
    workflows_dir = repo_path / ".github" / "workflows"

    results = {
        "workflows_dir_exists": workflows_dir.exists(),
        "has_test_workflow": False,
        "has_lint_workflow": False,
        "has_security_workflow": False,
    }

    if workflows_dir.exists():
        workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        workflow_names = [f.name for f in workflow_files]

        results["has_test_workflow"] = any("test" in name for name in workflow_names)
        results["has_lint_workflow"] = any("lint" in name for name in workflow_names)
        results["has_security_workflow"] = any("security" in name or "scan" in name for name in workflow_names)

    all_exist = results["workflows_dir_exists"] and results["has_test_workflow"]
    return all_exist, results


def validate_pre_commit(repo_path: Path) -> Tuple[bool, Dict[str, bool]]:
    """Validate pre-commit configuration."""
    pre_commit_file = repo_path / ".pre-commit-config.yaml"

    results = {
        "pre_commit_config_exists": pre_commit_file.exists(),
    }

    return pre_commit_file.exists(), results


def validate_project(project_path: Path) -> Dict[str, Any]:
    """Validate a single project."""
    project_name = project_path.name

    metadata_valid, metadata_results = validate_metadata(project_path)
    docs_valid, docs_results = validate_documentation(project_path)
    templates_valid, templates_results = validate_github_templates(project_path)
    ci_cd_valid, ci_cd_results = validate_ci_cd(project_path)
    pre_commit_valid, pre_commit_results = validate_pre_commit(project_path)

    all_valid = all([metadata_valid, docs_valid, templates_valid, ci_cd_valid, pre_commit_valid])

    return {
        "project": project_name,
        "path": str(project_path),
        "valid": all_valid,
        "metadata": metadata_results,
        "documentation": docs_results,
        "github_templates": templates_results,
        "ci_cd": ci_cd_results,
        "pre_commit": pre_commit_results,
    }


def validate_organizations(orgs_path: Path) -> Dict[str, Any]:
    """Validate all organizations."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "organizations": [],
        "summary": {
            "total_projects": 0,
            "valid_projects": 0,
            "invalid_projects": 0,
            "compliance_percentage": 0.0,
        }
    }

    if not orgs_path.exists():
        return results

    # Find all organization directories
    org_dirs = [d for d in orgs_path.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for org_dir in sorted(org_dirs):
        org_result = validate_project(org_dir)
        results["organizations"].append(org_result)

        results["summary"]["total_projects"] += 1
        if org_result["valid"]:
            results["summary"]["valid_projects"] += 1
        else:
            results["summary"]["invalid_projects"] += 1

    if results["summary"]["total_projects"] > 0:
        results["summary"]["compliance_percentage"] = (
            results["summary"]["valid_projects"] / results["summary"]["total_projects"] * 100
        )

    return results


# =============================================================================
# DEVOPS CLI VALIDATION
# =============================================================================

def validate_devops_cli_modules(repo_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate DevOps CLI module structure."""
    results = {
        "modules": {},
        "total_modules": len(DEVOPS_CLI_MODULES),
        "existing_modules": 0,
        "missing_modules": [],
    }

    for module in DEVOPS_CLI_MODULES:
        module_path = repo_path / module
        exists = module_path.exists()
        results["modules"][module] = {
            "exists": exists,
            "size": module_path.stat().st_size if exists else 0,
        }
        if exists:
            results["existing_modules"] += 1
        else:
            results["missing_modules"].append(module)

    all_exist = results["existing_modules"] == results["total_modules"]
    return all_exist, results


def validate_devops_templates(repo_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate DevOps template library."""
    templates_dir = repo_path / "templates" / "devops"

    results = {
        "templates_dir_exists": templates_dir.exists(),
        "categories": {},
        "total_categories": len(DEVOPS_TEMPLATE_CATEGORIES),
        "existing_categories": 0,
        "total_templates": 0,
        "missing_categories": [],
    }

    if templates_dir.exists():
        for category in DEVOPS_TEMPLATE_CATEGORIES:
            category_path = templates_dir / category
            exists = category_path.exists()
            template_count = 0

            if exists:
                # Count subdirectories as templates
                template_count = len([d for d in category_path.iterdir() if d.is_dir()])
                results["existing_categories"] += 1
            else:
                results["missing_categories"].append(category)

            results["categories"][category] = {
                "exists": exists,
                "template_count": template_count,
            }
            results["total_templates"] += template_count

    all_valid = (
        results["templates_dir_exists"] and
        results["existing_categories"] == results["total_categories"]
    )
    return all_valid, results


def validate_devops_tests(repo_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate DevOps CLI test coverage."""
    tests_dir = repo_path / "tests"

    expected_test_files = [
        "devops_cli.test.ts",
        "devops_config.test.ts",
        "devops_validate.test.ts",
        "devops_coder.test.ts",
    ]

    results = {
        "tests_dir_exists": tests_dir.exists(),
        "test_files": {},
        "total_expected": len(expected_test_files),
        "existing_tests": 0,
        "missing_tests": [],
    }

    for test_file in expected_test_files:
        test_path = tests_dir / test_file
        exists = test_path.exists()
        results["test_files"][test_file] = {
            "exists": exists,
            "size": test_path.stat().st_size if exists else 0,
        }
        if exists:
            results["existing_tests"] += 1
        else:
            results["missing_tests"].append(test_file)

    all_exist = results["existing_tests"] == results["total_expected"]
    return all_exist, results


def validate_devops_config(repo_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate DevOps development configuration."""
    config_files = {
        "package.json": repo_path / "package.json",
        "tsconfig.json": repo_path / "tsconfig.json",
        "eslint.config.js": repo_path / "eslint.config.js",
        "vitest.config.ts": repo_path / "vitest.config.ts",
        ".prettierrc": repo_path / ".prettierrc",
    }

    results = {
        "config_files": {},
        "total_configs": len(config_files),
        "existing_configs": 0,
        "missing_configs": [],
    }

    for name, path in config_files.items():
        exists = path.exists()
        results["config_files"][name] = {"exists": exists}
        if exists:
            results["existing_configs"] += 1
        else:
            results["missing_configs"].append(name)

    # Minimum required configs
    required = ["package.json", "tsconfig.json"]
    all_required = all(results["config_files"].get(r, {}).get("exists", False) for r in required)

    return all_required, results


def validate_devops_cli(repo_path: Path) -> Dict[str, Any]:
    """Full DevOps CLI validation."""
    modules_valid, modules_results = validate_devops_cli_modules(repo_path)
    templates_valid, templates_results = validate_devops_templates(repo_path)
    tests_valid, tests_results = validate_devops_tests(repo_path)
    config_valid, config_results = validate_devops_config(repo_path)

    all_valid = all([modules_valid, templates_valid, tests_valid, config_valid])

    return {
        "timestamp": datetime.now().isoformat(),
        "valid": all_valid,
        "modules": modules_results,
        "templates": templates_results,
        "tests": tests_results,
        "config": config_results,
        "summary": {
            "modules_complete": modules_valid,
            "templates_complete": templates_valid,
            "tests_complete": tests_valid,
            "config_complete": config_valid,
            "overall_status": "COMPLIANT" if all_valid else "NON-COMPLIANT",
        }
    }


# =============================================================================
# SUPERTOOL CROSS-REFERENCE VALIDATION
# =============================================================================

def validate_supertool_references(repo_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate SuperTool reference files exist."""
    results = {
        "reference_files": {},
        "total_required": len(REQUIRED_SUPERTOOL_REFS),
        "existing_refs": 0,
        "missing_refs": [],
    }

    for ref_file in REQUIRED_SUPERTOOL_REFS:
        ref_path = repo_path / ref_file
        exists = ref_path.exists()
        results["reference_files"][ref_file] = {
            "exists": exists,
            "size": ref_path.stat().st_size if exists else 0,
        }
        if exists:
            results["existing_refs"] += 1
        else:
            results["missing_refs"].append(ref_file)

    all_exist = results["existing_refs"] == results["total_required"]
    return all_exist, results


def validate_supertool_project(supertool_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate SuperTool project exists and has governance."""
    results = {
        "project_exists": supertool_path.exists(),
        "governance_dir": False,
        "validation_scripts": [],
        "makefile_targets": 0,
        "devops_files": 0,
    }

    if not supertool_path.exists():
        return False, results

    # Check governance directory
    governance_dir = supertool_path / "devops" / "governance"
    results["governance_dir"] = governance_dir.exists()

    # Check validation scripts
    scripts_dir = governance_dir / "scripts"
    if scripts_dir.exists():
        results["validation_scripts"] = [f.name for f in scripts_dir.glob("*.sh")]

    # Count devops files
    devops_dir = supertool_path / "devops"
    if devops_dir.exists():
        results["devops_files"] = len(list(devops_dir.rglob("*")))

    # Check Makefile for governance targets
    makefile = supertool_path / "Makefile"
    if makefile.exists():
        content = makefile.read_text(encoding="utf-8", errors="ignore")
        governance_targets = [
            line for line in content.split("\n")
            if line.startswith("governance-")
        ]
        results["makefile_targets"] = len(governance_targets)

    all_valid = results["governance_dir"] and len(results["validation_scripts"]) > 0
    return all_valid, results


def validate_governance_sync(repo_path: Path, supertool_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate governance sync between projects."""
    results = {
        "sync_script_exists": False,
        "unified_governance_exists": False,
        "validation_rules_synced": False,
        "last_sync_status": "unknown",
    }

    # Check sync script in SuperTool
    sync_script = supertool_path / "devops" / "governance" / "scripts" / "sync-governance.sh"
    results["sync_script_exists"] = sync_script.exists()

    # Check unified governance doc
    unified_gov = supertool_path / "devops" / "governance" / "UNIFIED_GOVERNANCE.md"
    results["unified_governance_exists"] = unified_gov.exists()

    # Check if validation rules reference is in sync
    central_ref = repo_path / ".metaHub" / "references" / "supertool" / "devops-validation-rules.yaml"
    supertool_rules = supertool_path / "devops" / "governance" / "validation-rules.yaml"

    if central_ref.exists() and supertool_rules.exists():
        results["validation_rules_synced"] = True
        results["last_sync_status"] = "synced"
    elif central_ref.exists() or supertool_rules.exists():
        results["last_sync_status"] = "partial"
    else:
        results["last_sync_status"] = "not_configured"

    all_valid = (
        results["sync_script_exists"] and
        results["unified_governance_exists"] and
        results["validation_rules_synced"]
    )
    return all_valid, results


def validate_supertool(repo_path: Path, supertool_path: Path) -> Dict[str, Any]:
    """Full SuperTool cross-reference validation."""
    refs_valid, refs_results = validate_supertool_references(repo_path)
    project_valid, project_results = validate_supertool_project(supertool_path)
    sync_valid, sync_results = validate_governance_sync(repo_path, supertool_path)

    all_valid = all([refs_valid, project_valid, sync_valid])

    return {
        "timestamp": datetime.now().isoformat(),
        "valid": all_valid,
        "central_references": refs_results,
        "supertool_project": project_results,
        "governance_sync": sync_results,
        "summary": {
            "references_complete": refs_valid,
            "project_configured": project_valid,
            "sync_configured": sync_valid,
            "overall_status": "SYNCED" if all_valid else "OUT_OF_SYNC",
        }
    }


# =============================================================================
# UNIFIED CROSS-PROJECT REPORT
# =============================================================================

def generate_unified_report(
    repo_path: Path,
    supertool_path: Path
) -> Dict[str, Any]:
    """Generate unified compliance report across all domains."""

    # Run all validations
    orgs_results = validate_organizations(repo_path / "organizations")
    devops_results = validate_devops_cli(repo_path)
    supertool_results = validate_supertool(repo_path, supertool_path)

    # Calculate overall scores
    scores = {
        "organizations": orgs_results["summary"]["compliance_percentage"] if orgs_results["summary"]["total_projects"] > 0 else 100.0,
        "devops_cli": 100.0 if devops_results["valid"] else 50.0,
        "supertool_sync": 100.0 if supertool_results["valid"] else 25.0,
    }

    overall_score = sum(scores.values()) / len(scores)

    return {
        "report_type": "unified_compliance",
        "timestamp": datetime.now().isoformat(),
        "repository": str(repo_path),
        "supertool": str(supertool_path),
        "domains": {
            "organizations": orgs_results,
            "devops_cli": devops_results,
            "supertool_integration": supertool_results,
        },
        "scores": scores,
        "overall": {
            "score": round(overall_score, 1),
            "status": "COMPLIANT" if overall_score >= 80 else "NEEDS_ATTENTION" if overall_score >= 50 else "NON_COMPLIANT",
            "grade": "A" if overall_score >= 90 else "B" if overall_score >= 80 else "C" if overall_score >= 70 else "D" if overall_score >= 60 else "F",
        }
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_organizations_report(results: Dict[str, Any]) -> None:
    """Print organizations compliance report."""
    print("\n" + "=" * 80)
    print("ORGANIZATIONS COMPLIANCE REPORT")
    print("=" * 80 + "\n")

    summary = results["summary"]
    print(f"Total Projects: {summary['total_projects']}")
    print(f"Valid Projects: {summary['valid_projects']}")
    print(f"Invalid Projects: {summary['invalid_projects']}")
    print(f"Compliance: {summary['compliance_percentage']:.1f}%\n")

    if results["organizations"]:
        print("-" * 80)
        print("ORGANIZATION DETAILS")
        print("-" * 80 + "\n")

        for org in results["organizations"]:
            status = "[PASS]" if org["valid"] else "[FAIL]"
            print(f"{status} {org['project']}")

            if not org["valid"]:
                for category, items in org.items():
                    if category not in ["project", "path", "valid"]:
                        for item, exists in items.items():
                            if not exists:
                                print(f"       Missing: {category}/{item}")
            print()


def print_devops_report(results: Dict[str, Any]) -> None:
    """Print DevOps CLI compliance report."""
    print("\n" + "=" * 80)
    print("DEVOPS CLI COMPLIANCE REPORT")
    print("=" * 80 + "\n")

    summary = results["summary"]
    print(f"Overall Status: {summary['overall_status']}")
    print(f"  Modules: {'PASS' if summary['modules_complete'] else 'FAIL'}")
    print(f"  Templates: {'PASS' if summary['templates_complete'] else 'FAIL'}")
    print(f"  Tests: {'PASS' if summary['tests_complete'] else 'FAIL'}")
    print(f"  Config: {'PASS' if summary['config_complete'] else 'FAIL'}")

    # Module details
    print(f"\nModules: {results['modules']['existing_modules']}/{results['modules']['total_modules']}")
    if results['modules']['missing_modules']:
        for missing in results['modules']['missing_modules']:
            print(f"  Missing: {missing}")

    # Template details
    print(f"\nTemplates: {results['templates']['total_templates']} across {results['templates']['existing_categories']} categories")
    if results['templates']['missing_categories']:
        for missing in results['templates']['missing_categories']:
            print(f"  Missing category: {missing}")

    # Test details
    print(f"\nTests: {results['tests']['existing_tests']}/{results['tests']['total_expected']}")
    if results['tests']['missing_tests']:
        for missing in results['tests']['missing_tests']:
            print(f"  Missing: {missing}")


def print_supertool_report(results: Dict[str, Any]) -> None:
    """Print SuperTool cross-reference report."""
    print("\n" + "=" * 80)
    print("SUPERTOOL CROSS-REFERENCE REPORT")
    print("=" * 80 + "\n")

    summary = results["summary"]
    print(f"Overall Status: {summary['overall_status']}")
    print(f"  Central References: {'PASS' if summary['references_complete'] else 'FAIL'}")
    print(f"  Project Configured: {'PASS' if summary['project_configured'] else 'FAIL'}")
    print(f"  Sync Configured: {'PASS' if summary['sync_configured'] else 'FAIL'}")

    # Reference details
    refs = results["central_references"]
    print(f"\nCentral References: {refs['existing_refs']}/{refs['total_required']}")
    if refs['missing_refs']:
        for missing in refs['missing_refs']:
            print(f"  Missing: {missing}")

    # Project details
    proj = results["supertool_project"]
    print("\nSuperTool Project:")
    print(f"  Exists: {'Yes' if proj['project_exists'] else 'No'}")
    print(f"  Governance Dir: {'Yes' if proj['governance_dir'] else 'No'}")
    print(f"  Validation Scripts: {len(proj['validation_scripts'])}")
    print(f"  Makefile Targets: {proj['makefile_targets']}")
    print(f"  DevOps Files: {proj['devops_files']}")

    # Sync status
    sync = results["governance_sync"]
    print("\nGovernance Sync:")
    print(f"  Sync Script: {'Yes' if sync['sync_script_exists'] else 'No'}")
    print(f"  Unified Governance: {'Yes' if sync['unified_governance_exists'] else 'No'}")
    print(f"  Rules Synced: {'Yes' if sync['validation_rules_synced'] else 'No'}")
    print(f"  Status: {sync['last_sync_status']}")


def print_unified_report(results: Dict[str, Any]) -> None:
    """Print unified cross-project compliance report."""
    print("\n" + "=" * 80)
    print("UNIFIED GOVERNANCE COMPLIANCE REPORT")
    print("=" * 80 + "\n")

    overall = results["overall"]
    print(f"Overall Score: {overall['score']}%")
    print(f"Grade: {overall['grade']}")
    print(f"Status: {overall['status']}")

    print("\n" + "-" * 80)
    print("DOMAIN SCORES")
    print("-" * 80 + "\n")

    scores = results["scores"]
    for domain, score in scores.items():
        bar_length = int(score / 5)
        bar = "#" * bar_length + "-" * (20 - bar_length)
        print(f"{domain:25} [{bar}] {score:.1f}%")

    # Print sub-reports
    print_organizations_report(results["domains"]["organizations"])
    print_devops_report(results["domains"]["devops_cli"])
    print_supertool_report(results["domains"]["supertool_integration"])


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run compliance validation."""
    args = sys.argv[1:] if len(sys.argv) > 1 else []

    # Determine repo path
    repo_path = Path.cwd()
    if not (repo_path / ".metaHub").exists():
        repo_path = GITHUB_REPO_PATH

    supertool_path = SUPERTOOL_PATH

    # Parse arguments
    if "--organizations" in args:
        results = validate_organizations(repo_path / "organizations")
        print_organizations_report(results)
    elif "--devops" in args:
        results = validate_devops_cli(repo_path)
        print_devops_report(results)
    elif "--supertool" in args:
        results = validate_supertool(repo_path, supertool_path)
        print_supertool_report(results)
    elif "--unified" in args or not args:
        results = generate_unified_report(repo_path, supertool_path)
        print_unified_report(results)
    elif "--json" in args:
        results = generate_unified_report(repo_path, supertool_path)
        print(json.dumps(results, indent=2))
    elif "--help" in args:
        print(__doc__)
        sys.exit(0)
    else:
        print("Usage: compliance_validator.py [OPTIONS]")
        print("\nOptions:")
        print("  --organizations  Validate organizations directory only")
        print("  --devops         Validate DevOps CLI only")
        print("  --supertool      Validate SuperTool cross-references only")
        print("  --unified        Generate unified cross-project report (default)")
        print("  --json           Output unified report as JSON")
        print("  --help           Show this help message")
        sys.exit(0)

    # Save report
    report_dir = repo_path / ".metaHub" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / f"compliance-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {report_file}")

    # Exit code based on compliance
    if isinstance(results.get("overall", {}).get("score"), (int, float)):
        sys.exit(0 if results["overall"]["score"] >= 80 else 1)
    elif "valid" in results:
        sys.exit(0 if results["valid"] else 1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
