# Governance Automation Guide

**Purpose:** Understanding and maintaining automated governance systems
**Last Updated:** 2025-12-04

---

## Overview

This repository uses automated governance to maintain code quality, security, and compliance. This guide explains how the governance automation works and how to maintain it.

## Governance Architecture

```
┌─────────────────────────────────────────────────────┐
│              Governance Orchestration                │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Pre-commit   │  │   CI/CD      │  │  Automated │ │
│  │    Hooks     │  │  Workflows   │  │   Agents   │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│         │                 │                  │       │
│         ▼                 ▼                  ▼       │
│  ┌───────────────────────────────────────────────┐  │
│  │         Governance Policy Engine              │  │
│  │  (.metaHub/policies/, automation/governance/) │  │
│  └───────────────────────────────────────────────┘  │
│                        │                             │
│                        ▼                             │
│              ┌──────────────────┐                    │
│              │  Enforcement      │                    │
│              │  - Block commits  │                    │
│              │  - Fail CI        │                    │
│              │  - Create issues  │                    │
│              └──────────────────┘                    │
└─────────────────────────────────────────────────────┘
```

---

## Governance Layers

### Layer 1: Pre-commit Enforcement

**Location:** `.husky/pre-commit`, `package.json` (lint-staged)

**What it does:**

- Runs on every git commit before code is committed
- Enforces code formatting and linting
- Scans for secrets and security issues
- Validates YAML files
- Checks file sizes

**Tools:**

- `lint-staged`: Runs linters on staged files
- `prettier`: Code formatting
- `eslint`: JavaScript/TypeScript linting
- `npm audit`: Security vulnerability scanning

**Configuration:**

```json
// package.json
{
  "lint-staged": {
    "!(templates/**/*)*.{ts,tsx,js,jsx}": ["eslint --fix", "prettier --write"],
    "!(templates/**/*)*.{json,md}": ["prettier --write"]
  }
}
```

**How to modify:**

1. Edit `package.json` → `lint-staged` section
2. Test changes: `npx lint-staged`
3. Commit and verify pre-commit hook runs

**Bypass (emergency only):**

```bash
git commit --no-verify  # Skips pre-commit hooks
```

⚠️ **Only use --no-verify in genuine emergencies!**

---

### Layer 2: CI/CD Governance

**Location:** `.github/workflows/`

**What it does:**

- Validates code quality on every push/PR
- Runs automated tests
- Performs security scanning
- Enforces technical debt gates
- Validates governance policies

**Key Workflows:**

| Workflow                       | Purpose                      | Frequency     |
| ------------------------------ | ---------------------------- | ------------- |
| `ci.yml`                       | Main CI pipeline             | Every push/PR |
| `super-linter.yml`             | Comprehensive linting        | Every push    |
| `structure-validation.yml`     | Repository structure checks  | Every push    |
| `orchestration-governance.yml` | Governance policy validation | Every push    |
| `weekly-governance-check.yml`  | Comprehensive audit          | Weekly        |
| `scorecard.yml`                | OSSF security scorecard      | Weekly        |

**Technical Debt Gate:**

```yaml
# .github/workflows/ci.yml (excerpt)
technical-debt:
  runs-on: ubuntu-latest
  steps:
    - name: Run technical-debt scan
      run: python debt_cli.py scan --path . --json debt_scan.json

    - name: Apply CI technical-debt gate
      run: python debt_gate.py --scan debt_scan.json --env ci
```

**Gate Thresholds:**

```yaml
# automation/governance/policies/technical_debt.yaml
thresholds:
  ci:
    max_total_items: 150
    max_high_severity_percentage: 60
    block_on_p0: true
```

**How to adjust thresholds:**

1. Edit `automation/governance/policies/technical_debt.yaml`
2. Test locally: `python automation/debt_gate.py --scan automation/debt_scan.json --env ci`
3. Create PR for review
4. Monitor first few CI runs after merge

---

### Layer 3: Automated Governance Agents

**Location:** `automation/agents/governance/`, `.ai/mcp/workflows/`

**What it does:**

- Monitors for governance violations
- Auto-files issues for detected problems
- Generates compliance reports
- Tracks technical debt trends

**Key Agents:**

1. **Structure Monitor** (`automation/agents/governance/structure_monitor.py`)
   - Validates repository structure against policies
   - Creates issues for violations
   - Runs: On-demand or triggered by workflows

2. **Compliance Tracker** (`automation/agents/governance/compliance_tracker.py`)
   - Tracks compliance metrics over time
   - Generates weekly reports
   - Archives historical data

3. **Policy Validator** (`automation/agents/governance/policy_validator.py`)
   - Validates all policy files are well-formed
   - Checks for policy conflicts
   - Updates policy documentation

**How to run agents manually:**

```bash
# Structure validation
python automation/agents/governance/structure_monitor.py --check

# Compliance report
python automation/agents/governance/compliance_tracker.py --report

# Policy validation
python automation/agents/governance/policy_validator.py --validate-all
```

---

## Governance Policies

### Policy Files

**Location:** `.metaHub/policies/`, `automation/governance/policies/`

**Policy Structure:**

```yaml
# Example: .metaHub/policies/root-structure.yaml
version: '1.0'
description: 'Root directory structure policy'

required_directories:
  - path: '.github/workflows'
    purpose: 'CI/CD workflows'
  - path: 'docs'
    purpose: 'Documentation'

required_files:
  - path: 'README.md'
    purpose: 'Project overview'
  - path: 'LICENSE'
    purpose: 'License information'

prohibited_patterns:
  - '*.env'
  - '**/*.key'
  - '**/*.pem'
```

### Adding a New Policy

1. **Create policy file:**

   ```bash
   # Create YAML policy
   cat > .metaHub/policies/new-policy.yaml << 'EOF'
   version: "1.0"
   description: "Description of policy"

   rules:
     - name: "Rule 1"
       severity: "high"
       pattern: "pattern to match"
   EOF
   ```

2. **Register policy:**

   ```bash
   # Add to policy index
   echo "  - new-policy.yaml" >> .metaHub/policies/index.yaml
   ```

3. **Test policy:**

   ```bash
   # Validate policy syntax
   python automation/agents/governance/policy_validator.py --file .metaHub/policies/new-policy.yaml

   # Dry run enforcement
   python automation/agents/governance/structure_monitor.py --check --dry-run
   ```

4. **Document policy:**
   - Add entry to `docs/governance/POLICIES.md`
   - Explain rationale and consequences
   - Provide examples

5. **Deploy policy:**
   ```bash
   git add .metaHub/policies/new-policy.yaml
   git commit -m "feat(governance): add new-policy for [purpose]"
   git push
   ```

---

## Technical Debt Management

### Debt Scanning

**Command:**

```bash
python automation/debt_cli.py scan --path . --json debt_scan.json
```

**What it scans:**

- TODO comments
- FIXME comments
- XXX comments
- HACK comments
- Type errors
- Lint errors
- Test skips
- Deprecated API usage

**Debt Categories:**

| Priority | Description | Examples                                        |
| -------- | ----------- | ----------------------------------------------- |
| P0       | Critical    | Security vulnerabilities, data corruption risks |
| P1       | High        | Broken functionality, flaky tests               |
| P2       | Medium      | Performance issues, missing tests               |
| P3       | Low         | Code style, minor refactoring                   |

### Debt Gate

**Purpose:** Prevent technical debt from accumulating beyond acceptable thresholds

**Environments:**

| Environment | Strictness | Use Case                                |
| ----------- | ---------- | --------------------------------------- |
| `dev`       | Lenient    | Local development, warnings only        |
| `ci`        | Moderate   | Pull requests, blocks on excessive debt |
| `prod`      | Strict     | Production deployments, zero tolerance  |

**Running manually:**

```bash
# Development mode (warnings)
python automation/debt_gate.py --scan automation/debt_scan.json --env dev

# CI mode (blocks on threshold)
python automation/debt_gate.py --scan automation/debt_scan.json --env ci

# Production mode (strict)
python automation/debt_gate.py --scan automation/debt_scan.json --env prod
```

**Exit codes:**

- `0`: Pass (debt within thresholds)
- `1`: Fail (debt exceeds thresholds)
- `2`: Error (scan file missing or invalid)

### Viewing Debt Reports

```bash
# View summary
cat automation/debt_scan.md

# View detailed JSON
python -m json.tool automation/debt_scan.json | less

# Filter by priority
jq '.items[] | select(.priority == "P0")' automation/debt_scan.json
```

---

## Monitoring & Alerts

### GitHub Actions Dashboard

**URL:** `https://github.com/YOUR_ORG/YOUR_REPO/actions`

**What to monitor:**

- Workflow success rate
- Average execution time
- Failed runs (investigate immediately)
- Flaky tests

### Weekly Governance Report

**Generated by:** `.github/workflows/weekly-governance-check.yml`

**Schedule:** Every Monday at 9:00 AM UTC

**Contents:**

- Repository health score
- Governance policy compliance
- Technical debt summary
- Security vulnerabilities
- Dependency updates needed

**Location:** GitHub Actions artifacts, or auto-creates issue with summary

### Setting Up Slack Notifications (Optional)

```yaml
# Add to .github/workflows/ci.yml
- name: Notify Slack on failure
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "CI Failed: ${{ github.repository }}",
        "blocks": [...]
      }
```

---

## Maintenance Tasks

### Daily

- [ ] Monitor CI status for failures
- [ ] Review Dependabot PRs
- [ ] Address P0 technical debt items

### Weekly

- [ ] Review weekly governance report
- [ ] Run manual compliance checks: `npm run governance check`
- [ ] Update dependencies: See [dependency-updates.md](../operations/runbooks/dependency-updates.md)

### Monthly

- [ ] Review and update governance policies
- [ ] Audit CODEOWNERS for accuracy
- [ ] Review technical debt trends
- [ ] Security audit: `npm audit && npm run ai:security:scan`

### Quarterly

- [ ] Review ADRs and mark outdated ones as Superseded
- [ ] Update governance thresholds based on team growth
- [ ] Performance review of CI/CD pipelines
- [ ] Update governance documentation

---

## Troubleshooting

### Issue: Pre-commit hook failing

**Diagnosis:**

```bash
# Run lint-staged manually
npx lint-staged

# Check specific linter
npm run lint
npm run type-check
```

**Common causes:**

- Code formatting issues → Run `npm run format`
- Type errors → Fix TypeScript errors
- Secrets detected → Remove secrets, add to .gitignore
- Large files → Remove or add to `.gitattributes`

**Temporary bypass (emergency only):**

```bash
git commit --no-verify
```

### Issue: CI failing on technical debt gate

**Diagnosis:**

```bash
# Run locally
python automation/debt_cli.py scan --path .
python automation/debt_gate.py --scan automation/debt_scan.json --env ci
```

**Solutions:**

1. Address high-priority debt items
2. Adjust thresholds if appropriate
3. Document exceptions in ADR

### Issue: Governance workflow failing

**Diagnosis:**

```bash
# Check workflow logs
gh run list --workflow=orchestration-governance.yml --limit 5
gh run view <run-id> --log

# Validate policies locally
python automation/agents/governance/policy_validator.py --validate-all
```

**Common causes:**

- Malformed YAML policy files
- Missing required directories/files
- Policy conflicts

---

## Advanced Configuration

### Custom Debt Detectors

Add custom patterns to scan for:

```python
# automation/debt_cli.py (excerpt)
CUSTOM_PATTERNS = [
    {
        'pattern': r'// DEPRECATED:',
        'priority': 'P1',
        'category': 'deprecated',
        'message': 'Using deprecated API'
    },
    {
        'pattern': r'console\.log\(',
        'priority': 'P2',
        'category': 'debug',
        'message': 'Debug console.log found'
    }
]
```

### Policy Exemptions

For legitimate exceptions:

```yaml
# .metaHub/policies/exemptions.yaml
exemptions:
  - path: 'templates/**'
    reason: 'Templates intentionally violate structure'
    approved_by: '@alaweimm90'
    expires: '2026-01-01'
```

---

## References

- [Technical Debt Policies](../../automation/governance/policies/technical_debt.yaml)
- [Repository Structure Policy](../../.metaHub/policies/root-structure.yaml)
- [ADR Documentation](../architecture/decisions/README.md)
- [Incident Response](../operations/incident-response.md)
- [GOVERNANCE.md](../../GOVERNANCE.md)
