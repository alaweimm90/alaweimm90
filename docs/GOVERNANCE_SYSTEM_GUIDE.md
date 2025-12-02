# Governance System Guide

## Overview

This governance system provides automated compliance checking, structure validation, and policy enforcement across 80+ repositories in 5 organizations. The system ensures consistent development practices, security standards, and project structure.

## Architecture

### Core Components

1. **Structure Validator** (`structure_validator.py`)
   - Validates repository structure against templates
   - Auto-detects language and project type
   - Generates compliance reports

2. **Governance Enforcer** (`enforce.py`)
   - Applies governance policies across repositories
   - Handles policy violations and remediation

3. **Repository Catalog** (`catalog.py`)
   - Maintains inventory of all repositories
   - Tracks metadata and compliance status

4. **Checkpoint System** (`checkpoint.py`)
   - Detects configuration drift
   - Monitors for policy violations

### Automation Scripts

- `setup_repo_ci.py` - Adds CI/CD workflows to repositories
- `sync_governance.py` - Syncs governance rules across all repos
- `push_monorepos.py` - Pushes organization monorepos to GitHub

## Repository Structure

### Governance Repository Layout

```
alaweimm90/alaweimm90/
├── .metaHub/                    # Governance infrastructure
│   ├── scripts/                 # Automation scripts
│   ├── schemas/                 # JSON schemas
│   ├── templates/               # Project templates
│   ├── policies/                # OPA policies
│   └── docs/                    # Governance docs
├── .github/workflows/           # CI/CD workflows
├── organizations/               # Local working copies
│   ├── alaweimm90-business/     # Business projects
│   ├── alaweimm90-science/      # Scientific computing
│   ├── alaweimm90-tools/        # Developer tools
│   ├── AlaweinOS/              # OS & infrastructure
│   └── MeatheadPhysicist/      # Physics education
└── docs/                        # Documentation
```

### Individual Repository Structure

Each repository must follow this structure:

```
repository/
├── .meta/repo.yaml              # Repository metadata
├── .github/
│   ├── workflows/ci.yml         # CI/CD pipeline
│   └── CODEOWNERS               # Ownership rules
├── src/ or lib/                 # Source code
├── tests/                       # Test suite
├── docs/                        # Documentation
├── pyproject.toml or package.json  # Build config
├── LICENSE                      # License file
├── README.md                    # Documentation
└── SECURITY.md                  # Security policy
```

## Compliance Requirements

### Tier-Based Requirements

| Tier                 | Requirements                                          |
| -------------------- | ----------------------------------------------------- |
| **1** (Critical)     | Full test coverage, SLO monitoring, incident runbooks |
| **2** (Important)    | 80%+ coverage, basic monitoring                       |
| **3** (Experimental) | Metadata only                                         |
| **4** (Unknown)      | Basic structure compliance                            |

### Language-Specific Requirements

#### Python Repositories

- `pyproject.toml` with proper configuration
- `src/` directory structure
- `tests/` with pytest
- Type hints and docstrings
- Pre-commit hooks (ruff, mypy, black)

#### TypeScript Repositories

- `package.json` with scripts
- `tsconfig.json` configuration
- `src/` and `tests/` directories
- ESLint and Prettier configuration

#### Go Repositories

- Standard Go project layout
- `go.mod` and `go.sum`
- Proper package structure

## Usage Guide

### Daily Operations

#### Check Compliance

```bash
# Validate all repositories
python .metaHub/scripts/structure_validator.py --report text

# Check specific organization
python .metaHub/scripts/structure_validator.py --org alaweimm90-tools

# JSON output for automation
python .metaHub/scripts/structure_validator.py --report json --output report.json
```

#### Auto-Fix Structure Issues

```bash
# Fix missing files/directories
python .metaHub/scripts/structure_validator.py --fix

# Dry run first
python .metaHub/scripts/structure_validator.py --fix --dry-run
```

#### Sync Governance Rules

```bash
# Update all repositories with latest governance rules
python .metaHub/scripts/sync_governance.py --all

# Update specific organization
python .metaHub/scripts/sync_governance.py --org alaweimm90-business
```

### Repository Onboarding

#### Add New Repository

1. Create repository in appropriate organization directory
2. Add `.meta/repo.yaml` with metadata
3. Run structure validator to check compliance
4. Run CI setup script
5. Push to GitHub

#### Repository Metadata Template

```yaml
type: library # library | service | tool | demo | research
language: python # python | typescript | go | rust | docs
tier: 2 # 1-4 (1=critical, 4=experimental)
owner: '@alaweimm90' # GitHub username or team
status: active # active | archived | deprecated
description: 'Brief project description'
```

### CI/CD Setup

#### Automated CI Setup

```bash
# Set up CI for all repositories
python .metaHub/scripts/setup_repo_ci.py --all

# Set up specific organization
python .metaHub/scripts/setup_repo_ci.py --org alaweimm90-tools
```

#### Manual CI Configuration

Each repository gets appropriate CI workflow based on:

- Language (Python, TypeScript, Go, etc.)
- Tier (security scans for tier 1-2)
- Type (different workflows for libraries vs services)

### GitHub Repository Management

#### Push Monorepos to GitHub

```bash
# Push all organization monorepos
python .metaHub/scripts/push_monorepos.py --all

# Push specific organization
python .metaHub/scripts/push_monorepos.py --org alaweimm90-business
```

## Monitoring and Reporting

### Weekly Compliance Checks

The system runs automated weekly checks via GitHub Actions:

- **Schedule**: Every Monday at 9 AM UTC
- **Checks**: Structure validation, policy compliance
- **Reporting**: GitHub Issues for violations
- **Artifacts**: Detailed JSON reports

### Manual Monitoring

#### Check Repository Drift

```bash
# Detect configuration changes
python .metaHub/scripts/checkpoint.py --scan
```

#### Generate Compliance Reports

```bash
# Full portfolio report
python .metaHub/scripts/structure_validator.py --report json --output compliance.json

# Governance policy check
python .metaHub/scripts/enforce.py organizations/ --report json --output policy.json
```

## Policy Enforcement

### OPA Policies

The system uses Open Policy Agent for policy enforcement:

- **Repository Structure** (`repo-structure.rego`)
- **Security Requirements** (`docker-security.rego`, `dependency-security.rego`)
- **Service SLOs** (`service-slo.rego`)
- **ADR Compliance** (`adr-policy.rego`)

### Policy Evaluation

```bash
# Test policies against repositories
python .metaHub/scripts/enforce.py organizations/ --policy repo-structure

# Generate policy reports
python .metaHub/scripts/enforce.py organizations/ --report json
```

## Troubleshooting

### Common Issues

#### Repository Not Detected

- Ensure `.meta/repo.yaml` exists
- Check file has valid YAML syntax
- Verify repository has build config (pyproject.toml, package.json, etc.)

#### CI/CD Not Working

- Check `.github/workflows/ci.yml` exists
- Verify workflow references correct reusable workflows
- Ensure governance repo has required workflows

#### Compliance Failures

- Run `structure_validator.py --fix` to auto-create missing files
- Check repository metadata is correct
- Verify language detection is working

### Debug Commands

#### Validate Single Repository

```bash
python .metaHub/scripts/structure_validator.py --repo organizations/alaweimm90-tools/my-repo
```

#### Test Policy Against Repository

```bash
python .metaHub/scripts/enforce.py organizations/alaweimm90-tools/my-repo --policy repo-structure
```

#### Check GitHub CLI Authentication

```bash
gh auth status
```

## Development

### Adding New Policies

1. Create `.rego` file in `.metaHub/policies/`
2. Update enforcement scripts to include new policy
3. Test against sample repositories
4. Update documentation

### Modifying Templates

1. Edit files in `.metaHub/templates/`
2. Update `portfolio-structure.yaml` for new requirements
3. Test with `structure_validator.py --fix --dry-run`
4. Update documentation

### Extending Automation

1. Add new scripts to `.metaHub/scripts/`
2. Update main CI workflow to include new checks
3. Add tests for new functionality
4. Update this documentation

## Security Considerations

- All automation scripts run with minimal permissions
- GitHub CLI authentication required for repository operations
- Policies prevent security violations
- Regular security scans via CI/CD pipelines

## Support

For issues or questions:

1. Check this documentation
2. Review GitHub Issues in governance repository
3. Run debug commands and provide output
4. Create new issue with detailed information

---

**Last Updated**: 2025-11-27
**Version**: 1.0.0
**Maintained by**: @alaweimm90
