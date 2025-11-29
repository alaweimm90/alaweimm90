# GOVERNANCE IMPLEMENTATION GUIDE

## Quick Start

This guide provides step-by-step instructions to implement the governance improvements identified in the audit report.

### Prerequisites

- Python 3.9+
- Git
- Access to all organization repositories
- Administrator permissions for GitHub settings

---

## Phase 1: Foundation (Weeks 1-2)

### Week 1: Metadata & Templates

#### Step 1.1: Create Metadata Schema

1. Copy the template to each organization:

```bash
cp .meta/repo.yaml.template organizations/alaweimm90-tools/.meta/repo.yaml
cp .meta/repo.yaml.template organizations/AlaweinOS/.meta/repo.yaml
cp .meta/repo.yaml.template organizations/MeatheadPhysicist/.meta/repo.yaml
cp .meta/repo.yaml.template organizations/alaweimm90-business/.meta/repo.yaml
cp .meta/repo.yaml.template organizations/alaweimm90-science/.meta/repo.yaml
```

2. Edit each `.meta/repo.yaml` with organization-specific details:
   - Update `organization` field
   - Update `project_name` field
   - Update `description` field
   - Update `owner` and `team` fields
   - Update `languages`, `frameworks`, `databases`
   - Set appropriate `tier` (1, 2, or 3)

#### Step 1.2: Create GitHub Templates

The following templates have been created in `.github/`:

- `.github/ISSUE_TEMPLATE/bug_report.md` ✅
- `.github/ISSUE_TEMPLATE/feature_request.md` ✅
- `.github/PULL_REQUEST_TEMPLATE.md` ✅

**Action:** Copy these templates to each organization's `.github/` directory.

#### Step 1.3: Expand CODEOWNERS

Update `.github/CODEOWNERS` in each organization:

```
# Default owner
* @alaweimm90

# Organization-specific owners
organizations/alaweimm90-tools/ @alaweimm90 @team-tools
organizations/AlaweinOS/ @alaweimm90 @team-research
organizations/MeatheadPhysicist/ @alaweimm90 @team-physics
organizations/alaweimm90-business/ @alaweimm90 @team-business
organizations/alaweimm90-science/ @alaweimm90 @team-science

# Domain-specific owners
.github/ @alaweimm90
.metaHub/ @alaweimm90
docs/ @alaweimm90
SECURITY.md @alaweimm90
```

### Week 2: CI/CD & Dependencies

#### Step 2.1: Create Reusable CI/CD Workflows

Create `.github/workflows/python-test.yml`:

```yaml
name: Python Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Lint with Ruff
        run: ruff check .

      - name: Format check with Ruff
        run: ruff format --check .

      - name: Type check with mypy
        run: mypy src/

      - name: Run tests with pytest
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

#### Step 2.2: Add Dependency Management

Create `renovate.json` in root:

```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": ["config:base", ":dependencyDashboard", ":semanticCommits"],
  "python": {
    "major": {
      "enabled": false
    }
  },
  "packageRules": [
    {
      "matchUpdateTypes": ["minor", "patch"],
      "automerge": true
    },
    {
      "matchDatasources": ["npm"],
      "automerge": true
    }
  ]
}
```

#### Step 2.3: Implement Compliance Checks

Run the compliance validator:

```bash
python .metaHub/scripts/compliance_validator.py
```

This will generate a report showing:

- Metadata compliance
- Documentation completeness
- GitHub templates presence
- CI/CD configuration
- Pre-commit hooks

---

## Phase 2: Standardization (Weeks 3-4)

### Week 3: Code Style & Testing

#### Step 3.1: Standardize Code Style

Create shared configuration files:

**`.ruff.toml`** (Python):

```toml
[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "SIM", "TCH", "PTH"]
ignore = ["E501", "B008"]

[tool.ruff.lint.mccabe]
max-complexity = 10
```

**`.eslintrc.json`** (JavaScript):

```json
{
  "extends": ["eslint:recommended"],
  "parserOptions": {
    "ecmaVersion": 2021,
    "sourceType": "module"
  },
  "rules": {
    "indent": ["error", 2],
    "linebreak-style": ["error", "unix"],
    "quotes": ["error", "single"],
    "semi": ["error", "always"]
  }
}
```

#### Step 3.2: Enforce Testing Standards

Update `pyproject.toml` in each Python project:

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
addopts = "--cov=src --cov-report=html --cov-report=term-missing"
```

#### Step 3.3: Update Pre-commit Hooks

Update `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-added-large-files
      - id: detect-private-key

  - repo: https://github.com/adrienverge/yamllint
    rev: v1.35.1
    hooks:
      - id: yamllint
        args: [-c, .yamllint.yml]
```

### Week 4: Documentation & Security

#### Step 4.1: Create Documentation Templates

Create `ARCHITECTURE.md` template:

```markdown
# Architecture

## Overview

High-level system architecture description.

## Components

- Component 1: Description
- Component 2: Description

## Data Flow

Describe how data flows through the system.

## Design Decisions

Document key architectural decisions.

## Deployment

Deployment architecture and considerations.
```

#### Step 4.2: Enhance Security Policies

Create security checklist in `SECURITY.md`:

```markdown
# Security Policy

## Pre-Deployment Checklist

- [ ] All dependencies scanned for vulnerabilities
- [ ] Secrets not committed to repository
- [ ] CORS properly configured
- [ ] Rate limiting enabled
- [ ] Input validation implemented
- [ ] SQL injection protection verified
- [ ] Authentication/authorization tested
- [ ] Audit logging enabled

## Incident Response

1. Identify the vulnerability
2. Assess impact
3. Create fix
4. Test fix
5. Deploy fix
6. Notify users if necessary
```

#### Step 4.3: Implement Branch Protection

For each repository on GitHub:

1. Go to Settings → Branches
2. Add rule for `main` branch:
   - Require pull request reviews (minimum 1)
   - Require status checks to pass
   - Require branches to be up to date
   - Dismiss stale PR approvals
   - Include administrators

---

## Phase 3: Automation (Weeks 5-6)

### Week 5: Automated Compliance

#### Step 5.1: Create Compliance Dashboard

Create `.metaHub/compliance_dashboard.py`:

```python
#!/usr/bin/env python3
"""Compliance dashboard generator."""

import json
from pathlib import Path
from compliance_validator import validate_organizations

def generate_dashboard():
    """Generate compliance dashboard."""
    orgs_path = Path.cwd() / "organizations"
    results = validate_organizations(orgs_path)

    # Generate HTML dashboard
    html = f"""
    <html>
    <head>
        <title>Governance Compliance Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .pass {{ color: green; }}
            .fail {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Governance Compliance Dashboard</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Projects: {results['summary']['total_projects']}</p>
            <p class="pass">Valid: {results['summary']['valid_projects']}</p>
            <p class="fail">Invalid: {results['summary']['invalid_projects']}</p>
            <p>Compliance: {results['summary']['compliance_percentage']:.1f}%</p>
        </div>
    </body>
    </html>
    """

    with open("compliance_dashboard.html", "w") as f:
        f.write(html)

if __name__ == "__main__":
    generate_dashboard()
```

#### Step 5.2: Implement Automated Checks

Add to CI/CD pipeline:

```yaml
- name: Run Compliance Check
  run: python .metaHub/scripts/compliance_validator.py
```

### Week 6: Documentation & Training

#### Step 6.1: Create Governance Documentation

Create `docs/GOVERNANCE_GUIDE.md`:

```markdown
# Governance Guide

## Overview

This guide explains the governance standards for all projects.

## Compliance Tiers

### Tier 1 (Critical)

- 90%+ test coverage
- SLO monitoring
- Incident runbooks

### Tier 2 (Important)

- 80%+ test coverage
- Basic monitoring

### Tier 3 (Experimental)

- Metadata only

## Compliance Checklist

- [ ] `.meta/repo.yaml` created
- [ ] README.md complete
- [ ] CONTRIBUTING.md present
- [ ] SECURITY.md present
- [ ] GitHub templates created
- [ ] CI/CD workflows configured
- [ ] Pre-commit hooks configured
- [ ] Branch protection enabled
- [ ] CODEOWNERS defined
- [ ] Tests passing
```

#### Step 6.2: Create Training Materials

Create video tutorials covering:

1. Metadata schema setup
2. GitHub templates usage
3. CI/CD configuration
4. Compliance validation
5. Best practices

---

## Verification Checklist

After completing each phase, verify:

### Phase 1 Verification

- [ ] All organizations have `.meta/repo.yaml`
- [ ] GitHub templates present in all repos
- [ ] CODEOWNERS expanded with team members
- [ ] Compliance validator runs successfully

### Phase 2 Verification

- [ ] Code style standards applied
- [ ] Testing standards enforced
- [ ] Pre-commit hooks working
- [ ] Documentation templates created
- [ ] Security policies enhanced
- [ ] Branch protection enabled

### Phase 3 Verification

- [ ] Compliance dashboard accessible
- [ ] Automated checks running in CI/CD
- [ ] Governance documentation complete
- [ ] Team trained on new standards
- [ ] 100% compliance achieved

---

## Troubleshooting

### Issue: Compliance validator fails

**Solution:**

```bash
# Check for missing files
python .metaHub/scripts/compliance_validator.py

# Create missing files
touch organizations/[org]/.meta/repo.yaml
touch organizations/[org]/.github/ISSUE_TEMPLATE/bug_report.md
```

### Issue: Pre-commit hooks not running

**Solution:**

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Issue: CI/CD workflows not triggering

**Solution:**

1. Check workflow file syntax
2. Verify branch names match
3. Check GitHub Actions permissions
4. Review workflow logs

---

## Success Metrics

Track these metrics to measure success:

| Metric                     | Target | Current | Status |
| -------------------------- | ------ | ------- | ------ |
| Metadata Compliance        | 100%   | 0%      | ⏳     |
| Documentation Completeness | 95%    | 0%      | ⏳     |
| Test Coverage              | 80%+   | 60-95%  | ⏳     |
| CI/CD Coverage             | 100%   | 50%     | ⏳     |
| Security Scanning          | 100%   | 0%      | ⏳     |
| Branch Protection          | 100%   | 0%      | ⏳     |

---

## Support

For questions or issues:

1. Check the [Governance Audit Report](./GOVERNANCE_AUDIT_REPORT.md)
2. Review the [Compliance Validator](../.metaHub/scripts/compliance_validator.py)
3. Contact the governance team

---

**Last Updated:** November 28, 2025  
**Next Review:** December 28, 2025
