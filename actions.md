# Prioritized Remediation Actions

**Generated:** 2025-11-25
**Scope:** 35 repositories under github.com/alaweimm90
**Estimated Total Effort:** 120-150 hours over 4-6 weeks

---

## Priority Levels

- **P0:** Critical - Breaks Golden Path rules, security risk, or blocks library/tool usage (fix within 1 week)
- **P1:** High - Missing required files or governance integration (fix within 1 month)
- **P2:** Medium - Quality improvements and standardization (fix within 3 months)

---

## Executive Summary

### By Priority

| Priority | Actions | Affected Repos | Estimated Effort |
|----------|---------|----------------|------------------|
| P0 | 18 actions | 15 repos | 60-80 hours |
| P1 | 12 actions | 20 repos | 40-50 hours |
| P2 | 8 actions | 35 repos | 20-30 hours |

### Quick Wins (P0 - Week 1)

1. Add `.meta/repo.yaml` to AlaweinOS (1 hour) - Makes 2nd compliant repo
2. Add tests to mag-logic and spin-circ (16 hours) - Fixes library compliance
3. Add tests to HELIOS and TalAI (16 hours) - Fixes featured AI tools
4. Archive or fix 3 dead repos (4 hours) - Cleans portfolio

### Strategic Initiatives (P1-P2 - Months 2-3)

1. Bootstrap `.meta/repo.yaml` across all repos (8 hours)
2. Migrate to reusable workflows (20 hours)
3. Add CODEOWNERS universally (6 hours)

---

## P0 Actions (Critical - Fix Within 1 Week)

### P0-1: Add Tests to Libraries (mag-logic, spin-circ)

**Impact:** 2 libraries missing tests - violates Golden Path requirement of ≥80%
**Effort:** 16 hours (8 hours per library)
**Affected:** `mag-logic`, `spin-circ`

**Rationale:**
- Libraries without tests cannot be trusted
- Golden Path mandates ≥80% coverage for all libs
- These are scientific computing libraries - correctness is critical

**Implementation:**

```bash
# For mag-logic
cd organizations/alaweimm90-science/mag-logic
mkdir -p tests
```

Create `organizations/alaweimm90-science/mag-logic/tests/test_core.py`:
```python
"""Core functionality tests for mag-logic."""
import pytest

# TODO: Import actual modules once structure is understood
# from mag_logic import core

def test_simulation_initialization():
    """Test nanomagnetic logic simulation initialization."""
    # TODO: Implement based on actual API
    pass

def test_simulation_step():
    """Test simulation step execution."""
    # TODO: Implement based on actual API
    pass

def test_simulation_results():
    """Test simulation results output."""
    # TODO: Implement based on actual API
    pass

# Target: Achieve ≥80% coverage
```

Create `organizations/alaweimm90-science/mag-logic/.github/workflows/coverage.yml`:
```yaml
name: Coverage

on:
  push:
    branches: [main, master]
  pull_request:

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install pytest pytest-cov
          pip install -e .
      - name: Run tests with coverage
        run: |
          pytest --cov=mag_logic --cov-report=term --cov-report=xml --cov-fail-under=80
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: always()
```

**Repeat for spin-circ** with appropriate module names.

**Verification:**
```bash
cd organizations/alaweimm90-science/mag-logic
pytest --cov=mag_logic --cov-report=term
# Must show ≥80% coverage
```

---

### P0-2: Add Tests to Tools (HELIOS, TalAI)

**Impact:** 2 featured AI tools missing tests - violates Golden Path requirement of ≥80%
**Effort:** 16 hours (8 hours per tool)
**Affected:** `HELIOS`, `TalAI`

**Rationale:**
- Tools require ≥80% coverage per Golden Path
- These are featured AI capabilities - quality is critical
- HELIOS is AI orchestration system - must be reliable

**Implementation:**

Create `organizations/alaweimm90-tools/HELIOS/tests/test_orchestration.py`:
```python
"""Tests for HELIOS AI orchestration."""
import pytest

# TODO: Import actual modules
# from helios import orchestrator

def test_hypothesis_generation():
    """Test hypothesis generation from prompts."""
    pass

def test_learning_pipeline():
    """Test learning intelligence pipeline."""
    pass

def test_orchestration_workflow():
    """Test complete orchestration workflow."""
    pass

# Target: ≥80% coverage
```

Add coverage workflow (similar to P0-1).

**Repeat for TalAI** with AI talent platform specifics.

---

### P0-3: Fix or Archive Dead Repositories

**Impact:** 3 repos with no CI, no tests, minimal docs
**Effort:** 4 hours decision + 2 hours per repo if fixing
**Affected:** `calla-lily-couture`, `dr-alowein-portfolio`, `marketing-automation`

**Decision Matrix:**

| Repo | Status | Recommendation | Effort if Keeping |
|------|--------|----------------|-------------------|
| calla-lily-couture | No activity signals | **ARCHIVE** | N/A |
| dr-alowein-portfolio | Personal portfolio | **EXEMPT** (add .golden-path-exempt) | 1 hour |
| marketing-automation | Tool without code | **ARCHIVE or FIX** | 10 hours |

**Implementation - Archive:**

```bash
# Add DEPRECATED.md to repo root
```

`organizations/alaweimm90-business/calla-lily-couture/DEPRECATED.md`:
```markdown
# DEPRECATED

This repository has been archived as of 2025-11-25.

**Reason:** Inactive project, no longer maintained

**Alternatives:** See live-it-iconic for active e-commerce platform

**Archive Date:** 2025-11-25
```

Update README.md to add deprecation notice at top.

**Implementation - Exempt:**

`organizations/alaweimm90-business/dr-alowein-portfolio/.golden-path-exempt`:
```yaml
# Golden Path Exemption
repo: dr-alowein-portfolio
reason: Personal portfolio site - does not require full governance
exempt_from:
  - ci_required
  - tests_required
  - security_md
  - contributing_md
approved_by: "@alaweimm90"
approved_date: "2025-11-25"
```

---

### P0-4: Add Coverage Tracking to Libraries

**Impact:** 3 libraries cannot verify ≥80% requirement
**Effort:** 6 hours (2 hours per library)
**Affected:** `qmat-sim`, `qube-ml`, `sci-comp`

**Implementation:**

For each library, add `.github/workflows/coverage.yml` (see P0-1 template).

Update existing CI to include coverage requirements:

```yaml
# Add to existing .github/workflows/ci.yml
- name: Run tests with coverage
  run: |
    pytest --cov=<module_name> --cov-fail-under=80
```

---

### P0-5: Bootstrap CI for Critical Tools

**Impact:** 3 critical tools with no CI at all
**Effort:** 12 hours (4 hours per repo)
**Affected:** `marketing-automation` (if not archived), `admin-dashboard`, `CrazyIdeas`

**Implementation:**

Create `.github/workflows/ci.yml` calling governance:

```yaml
name: Governance

on:
  push:
    branches: [main, master]
  pull_request:

jobs:
  lint:
    uses: alaweimm90/alaweimm90/.github/workflows/super-linter.yml@master

  policies:
    uses: alaweimm90/alaweimm90/.github/workflows/opa-conftest.yml@master

  security:
    uses: alaweimm90/alaweimm90/.github/workflows/scorecard.yml@master
    permissions:
      security-events: write
      id-token: write
```

---

## P1 Actions (High Priority - Fix Within 1 Month)

### P1-1: Universal `.meta/repo.yaml` Deployment

**Impact:** 34/35 repos missing required file
**Effort:** 8 hours (15 min per repo)
**Affected:** All repos except `AlaweinOS`

**Rationale:**
- Required by Golden Path
- Enables programmatic validation
- Documents repo metadata canonically

**Template** (use `AlaweinOS/.meta/repo.yaml` as reference):

```yaml
# .meta/repo.yaml - Repository Metadata
# Required by Golden Path - github.com/alaweimm90

repo:
  name: "{repo_name}"
  prefix: "{core|lib|tool|demo|adapter|template|paper|archive|infra}"
  type: "{specific_type}"
  description: "{one_line_description}"

ownership:
  primary_owner: "@alaweimm90"
  team: "{team_name}"
  organization: "{organization}"

policies:
  docs_profile: "{minimal|standard}"
  test_coverage_required: "{0|70|80|90}%"
  ci_required: true
  opa_policies: true

structure:
  language: ["{Python|JavaScript|TypeScript|Mixed}"]
  framework: ["{React|Express|FastAPI|etc}"]
  monorepo: false

compliance:
  golden_path_version: "1.0"
  last_audit: "2025-11-25"
  exemptions: []
```

**Batch Script:**

```bash
#!/bin/bash
# generate-meta-yaml.sh

REPOS=(
  "organizations/alaweimm90-business/benchbarrier:demo:e-commerce:70"
  "organizations/alaweimm90-business/repz:core:platform:80"
  "organizations/alaweimm90-science/mag-logic:lib:scientific-library:80"
  # ... add all 34 repos
)

for repo_config in "${REPOS[@]}"; do
  IFS=':' read -r path prefix type coverage <<< "$repo_config"
  repo_name=$(basename "$path")

  mkdir -p "$path/.meta"

  cat > "$path/.meta/repo.yaml" <<EOF
repo:
  name: "$repo_name"
  prefix: "$prefix"
  type: "$type"
  description: "TODO: Add from README"

ownership:
  primary_owner: "@alaweimm90"
  team: "platform"
  organization: "alaweimm90"

policies:
  docs_profile: "standard"
  test_coverage_required: "${coverage}%"
  ci_required: true
  opa_policies: true

structure:
  language: ["TODO"]
  framework: []
  monorepo: false

compliance:
  golden_path_version: "1.0"
  last_audit: "2025-11-25"
  exemptions: []
EOF

  echo "Created .meta/repo.yaml for $repo_name"
done
```

**Verification:**

Add OPA policy to validate `.meta/repo.yaml`:

```rego
# .metaHub/policies/repo-meta.rego
package repo_meta

import future.keywords.contains
import future.keywords.if
import future.keywords.in

# Check .meta/repo.yaml exists
deny[msg] {
    not file_exists(".meta/repo.yaml")
    msg := ".meta/repo.yaml is required by Golden Path"
}

# Validate required fields
deny[msg] {
    meta := yaml.unmarshal(input[".meta/repo.yaml"])
    not meta.repo.name
    msg := ".meta/repo.yaml missing required field: repo.name"
}

deny[msg] {
    meta := yaml.unmarshal(input[".meta/repo.yaml"])
    not meta.repo.prefix
    msg := ".meta/repo.yaml missing required field: repo.prefix"
}

# Validate prefix taxonomy
deny[msg] {
    meta := yaml.unmarshal(input[".meta/repo.yaml"])
    prefix := meta.repo.prefix
    valid_prefixes := {"core", "lib", "tool", "demo", "adapter", "template", "paper", "archive", "infra"}
    not prefix in valid_prefixes
    msg := sprintf(".meta/repo.yaml invalid prefix '%s' - must be one of: %v", [prefix, valid_prefixes])
}
```

---

### P1-2: Universal CODEOWNERS Deployment

**Impact:** 34/35 repos missing required file
**Effort:** 6 hours (10 min per repo)
**Affected:** All repos except `alaweimm90` (meta governance)

**Template:**

```
# CODEOWNERS - Enforce Review Requirements

# Default owner for everything
* @alaweimm90

# Governance files require explicit approval
.meta/ @alaweimm90
.github/ @alaweimm90
SECURITY.md @alaweimm90
LICENSE @alaweimm90

# CI/CD workflows require DevOps review
.github/workflows/ @alaweimm90

# Policy files require security approval
*.rego @alaweimm90
policies/ @alaweimm90
```

**Batch Script:**

```bash
#!/bin/bash
# generate-codeowners.sh

find organizations/ -maxdepth 4 -type d -name .git -prune -o -type d | while read dir; do
  if [ -d "$dir" ] && [ ! -f "$dir/.github/CODEOWNERS" ]; then
    mkdir -p "$dir/.github"
    cp .metaHub/templates/CODEOWNERS "$dir/.github/CODEOWNERS"
    echo "Created CODEOWNERS for $dir"
  fi
done
```

---

### P1-3: Add Missing LICENSE Files

**Impact:** 9 repos without LICENSE
**Effort:** 2 hours (13 min per repo)
**Affected:** See inventory.json

**Standard License:** MIT (consistent with existing repos)

**Template:**

```
MIT License

Copyright (c) 2025 Mohammed Alowein

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Batch Script:**

```bash
#!/bin/bash
# add-licenses.sh

REPOS_WITHOUT_LICENSE=(
  "organizations/alaweimm90-business/calla-lily-couture"
  "organizations/alaweimm90-business/dr-alowein-portfolio"
  "organizations/alaweimm90-business/marketing-automation"
  "organizations/alaweimm90-tools/admin-dashboard"
  "organizations/alaweimm90-tools/alaweimm90-cli"
  "organizations/alaweimm90-tools/alaweimm90-python-sdk"
  "organizations/alaweimm90-tools/business-intelligence"
  "organizations/alaweimm90-tools/core-framework"
  "organizations/alaweimm90-tools/CrazyIdeas"
)

for repo in "${REPOS_WITHOUT_LICENSE[@]}"; do
  if [ ! -f "$repo/LICENSE" ]; then
    cp .metaHub/templates/LICENSE "$repo/LICENSE"
    echo "Added LICENSE to $repo"
  fi
done
```

---

### P1-4: Add Missing SECURITY.md Files

**Impact:** 16 repos without SECURITY.md
**Effort:** 4 hours (15 min per repo)

**Template:**

```markdown
# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it via:

1. **GitHub Security Advisories** (preferred): [Report a vulnerability](../../security/advisories/new)
2. **Email**: security@alowein.dev (if GitHub not applicable)

**Do NOT** open a public issue for security vulnerabilities.

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity (Critical: 7 days, High: 30 days, Medium: 90 days)

## Security Measures

This repository is protected by:

- ✅ **OpenSSF Scorecard** - Weekly security health checks
- ✅ **Renovate** - Automated dependency updates
- ✅ **CodeQL** - Static analysis for vulnerabilities (if applicable)
- ✅ **Governance Policies** - OPA enforcement from alaweimm90/alaweimm90

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | ✅        |
| Older   | ❌        |

We only support the latest version. Please upgrade to receive security updates.

## Security Best Practices

When contributing:
- Never commit secrets, API keys, or credentials
- Use `.env` files for local secrets (add to `.gitignore`)
- Follow OWASP Top 10 guidelines
- Run `git-secrets` or `gitleaks` before committing

## Related Policies

See [alaweimm90/alaweimm90](https://github.com/alaweimm90/alaweimm90) for:
- Governance policies
- Security scanning workflows
- Compliance frameworks
```

---

### P1-5: Migrate Pilot Repos to Reusable Workflows

**Impact:** Reduces CI duplication, ensures consistent enforcement
**Effort:** 12 hours (6 hours per repo)
**Pilot Repos:** `repz`, `live-it-iconic`

**Rationale:**
- These are gold standard repos with extensive CI
- Success here validates pattern for all repos
- Reduces 30 workflows → 5 reusable workflows

**Implementation:**

**Step 1:** Create reusable workflow in governance repo:

`.github/workflows/reusable-node-ci.yml` (in `alaweimm90/alaweimm90`):
```yaml
name: Reusable Node.js CI

on:
  workflow_call:
    inputs:
      node-version:
        required: false
        type: string
        default: '18'
      coverage-threshold:
        required: false
        type: number
        default: 80

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
      - run: npm ci
      - run: npm run lint

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
      - run: npm ci
      - run: npm test -- --coverage --coverageThreshold='{"global":{"lines":${{ inputs.coverage-threshold }}}}}'

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
      - run: npm ci
      - run: npm run build
```

**Step 2:** Simplify `repz` CI:

Replace 30 workflows with:

`.github/workflows/governance.yml`:
```yaml
name: Governance & CI

on:
  push:
    branches: [main, master]
  pull_request:

jobs:
  # Call reusable workflows from governance repo
  super-linter:
    uses: alaweimm90/alaweimm90/.github/workflows/super-linter.yml@master
    permissions:
      contents: read
      packages: read
      statuses: write

  opa-policies:
    uses: alaweimm90/alaweimm90/.github/workflows/opa-conftest.yml@master

  scorecard:
    uses: alaweimm90/alaweimm90/.github/workflows/scorecard.yml@master
    permissions:
      security-events: write
      id-token: write

  # Repo-specific CI
  ci:
    uses: alaweimm90/alaweimm90/.github/workflows/reusable-node-ci.yml@master
    with:
      node-version: '18'
      coverage-threshold: 80

  # Keep specialized workflows (E2E, performance, etc.)
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # ... existing E2E steps
```

**Step 3:** Archive replaced workflows:

```bash
cd organizations/alaweimm90-business/repz
mkdir -p .github/workflows-archive
mv .github/workflows/ci.yml .github/workflows-archive/
mv .github/workflows/lint.yml .github/workflows-archive/
# ... archive duplicated workflows
```

**Step 4:** Verify:

```bash
# Trigger CI
git commit -m "test: verify governance workflows"
git push

# Check workflow runs
gh run list --limit 5
```

---

### P1-6: Add Coverage Reporting to Featured Tools

**Impact:** Cannot verify ≥80% for Attributa, qmlab
**Effort:** 4 hours (2 hours per tool)

**Implementation:**

Add coverage workflow (see P0-4 template) to:
- `organizations/alaweimm90-tools/Attributa`
- `organizations/AlaweinOS/qmlab`

---

### P1-7: Add CONTRIBUTING.md to Standard Repos

**Impact:** 15 repos missing CONTRIBUTING.md with standard docs profile
**Effort:** 8 hours (30 min per repo)

**Template:**

```markdown
# Contributing to {PROJECT_NAME}

Thank you for your interest in contributing!

## Code of Conduct

Be respectful, inclusive, and professional.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/{repo}.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make changes and commit: `git commit -m "feat: your feature"`
5. Push and create a Pull Request

## Development Setup

\`\`\`bash
# Install dependencies
{npm install | pip install -e . | etc}

# Run tests
{npm test | pytest | etc}

# Run linting
{npm run lint | flake8 | etc}
\`\`\`

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from `@alaweimm90`

## Testing Requirements

- **Libraries:** ≥80% coverage required
- **Tools:** ≥80% coverage required
- **Demos:** ≥70% coverage required

## Code Review

- All PRs require approval from CODEOWNERS
- CI must pass (Super-Linter, OPA policies, tests)
- No merge until all checks green

## Questions?

Open an issue or contact `@alaweimm90`.
```

---

## P2 Actions (Medium Priority - Fix Within 3 Months)

### P2-1: Systematic Reusable Workflow Migration

**Impact:** All 22 repos with CI should call governance workflows
**Effort:** 24 hours (1 hour per repo after pilots proven)
**Affected:** All repos except `alaweimm90`, `repz` (pilot), `live-it-iconic` (pilot)

**Process:**
1. Complete P1-5 pilots first
2. Document lessons learned
3. Create migration checklist
4. Migrate in batches of 5 repos
5. Verify each batch before proceeding

**Batch 1 (Scientific Libraries):**
- mag-logic
- qmat-sim
- qube-ml
- sci-comp
- spin-circ

**Batch 2 (Tools):**
- Attributa
- HELIOS
- fitness-app
- job-search
- LLMWorks

**Batch 3 (Core Platforms):**
- AlaweinOS
- MEZAN
- optilibria
- QAPlibria-new
- qmlab

**Batch 4 (Remaining):**
- SimCore
- TalAI
- MeatheadPhysicist
- benchbarrier
- CrazyIdeas

---

### P2-2: Add Renovate to All Repositories

**Impact:** Automated dependency updates across portfolio
**Effort:** 4 hours (configuration at org level)

**Implementation:**

Update `.metaHub/renovate.json` to include all repos:

```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": [
    "config:base"
  ],
  "repositories": [
    "alaweimm90/alaweimm90",
    "organizations/*/.*"
  ],
  "labels": ["dependencies"],
  "automerge": true,
  "automergeType": "pr",
  "automergeStrategy": "squash",
  "platformAutomerge": true,
  "schedule": ["every 3 hours"],
  "prConcurrentLimit": 10,
  "prHourlyLimit": 0,
  "packageRules": [
    {
      "matchUpdateTypes": ["minor", "patch"],
      "automerge": true
    },
    {
      "matchUpdateTypes": ["major"],
      "automerge": false,
      "labels": ["major-update"]
    }
  ]
}
```

---

### P2-3: Backstage Catalog Synchronization

**Impact:** Keep service catalog up to date
**Effort:** 6 hours (initial) + 30 min/month (maintenance)

**Implementation:**

Create automated catalog update script:

```bash
#!/bin/bash
# .metaHub/scripts/sync-backstage-catalog.sh

# Generate catalog entries from .meta/repo.yaml files
find organizations/ -name "repo.yaml" -path "*/.meta/*" | while read meta_file; do
  # Parse YAML and generate Backstage component
  # Add to .metaHub/backstage/catalog-info.yaml
  echo "Syncing $(dirname $(dirname $meta_file))"
done
```

Add to GitHub Actions:

`.github/workflows/backstage-sync.yml`:
```yaml
name: Backstage Catalog Sync

on:
  push:
    paths:
      - 'organizations/**/.meta/repo.yaml'
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Sync catalog
        run: .metaHub/scripts/sync-backstage-catalog.sh
      - name: Commit changes
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add .metaHub/backstage/catalog-info.yaml
          git commit -m "chore: sync Backstage catalog" || exit 0
          git push
```

---

### P2-4: Add Pre-commit Hooks

**Impact:** Catch issues before CI runs
**Effort:** 2 hours (org-level setup)

**Implementation:**

`.metaHub/templates/.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=10240']
      - id: check-merge-conflict

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.1
    hooks:
      - id: gitleaks

  - repo: local
    hooks:
      - id: opa-test
        name: OPA Policy Test
        entry: bash -c 'conftest test --policy .metaHub/policies/ .'
        language: system
        pass_filenames: false
```

Add to all repos:

```bash
find organizations/ -type d -maxdepth 3 | while read dir; do
  if [ -d "$dir/.git" ]; then
    cp .metaHub/templates/.pre-commit-config.yaml "$dir/"
    echo "Added pre-commit config to $dir"
  fi
done
```

---

### P2-5: Documentation Quality Improvement

**Impact:** Standardize docs across repos
**Effort:** 16 hours (30 min per repo)

**Checklist per repo:**
- ✅ README has badges (build, coverage, license, scorecard)
- ✅ README has Getting Started section
- ✅ README has Contributing link
- ✅ README has License section
- ✅ API documentation exists (for libraries)
- ✅ Architecture decision records (for core platforms)

**Template Badges:**

```markdown
# {PROJECT_NAME}

[![CI](https://github.com/alaweimm90/{repo}/actions/workflows/ci.yml/badge.svg)](https://github.com/alaweimm90/{repo}/actions)
[![Coverage](https://codecov.io/gh/alaweimm90/{repo}/branch/main/graph/badge.svg)](https://codecov.io/gh/alaweimm90/{repo})
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/alaweimm90/{repo}/badge)](https://securityscorecards.dev/viewer/?uri=github.com/alaweimm90/{repo})
```

---

### P2-6: Split or Merge Repositories

**Impact:** Rationalize repo structure
**Effort:** 20 hours (case by case)

**Candidates for Merging:**

| Group | Repos | Reason | New Name |
|-------|-------|--------|----------|
| MeatheadPhysicist components | cli, frontend, src, visualizations | Workspace with sub-projects | Keep workspace, document pattern |
| alaweimm90-tools | 17 repos | Too fragmented | Keep workspace, add monorepo tooling |
| Prompty projects | prompty, prompty-service | Single purpose | Merge into prompty |

**Candidates for Splitting:**

| Repo | Reason | Split Into |
|------|--------|------------|
| AlaweinOS | Large workspace | Keep as monorepo, add clearer boundaries |
| MEZAN | Monorepo | Keep as-is, document internal structure |

**Recommendation:**
- Keep current structure
- Add `.meta/repo.yaml` with `monorepo: true` flag
- Document workspace patterns in governance
- Use tools like Nx or Turborepo for monorepo management

---

### P2-7: Add GitHub Issue/PR Templates

**Impact:** Standardize contributions
**Effort:** 4 hours (template creation + deployment)

**Templates to Add:**

`.github/ISSUE_TEMPLATE/bug_report.md`:
```markdown
---
name: Bug Report
about: Report a bug
labels: bug
---

## Description
A clear description of the bug.

## Steps to Reproduce
1.
2.
3.

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS:
- Version:
- Browser (if applicable):

## Additional Context
Any other relevant information.
```

`.github/ISSUE_TEMPLATE/feature_request.md`, `.github/pull_request_template.md`

Deploy to all repos via script.

---

### P2-8: Establish Monthly Audit Cadence

**Impact:** Maintain compliance over time
**Effort:** 4 hours/month

**Process:**

1. **Monthly Audit Script:**

```bash
#!/bin/bash
# .metaHub/scripts/monthly-audit.sh

echo "=== Monthly Golden Path Audit ==="
echo "Date: $(date)"
echo ""

# Check .meta/repo.yaml compliance
echo "=== .meta/repo.yaml Coverage ==="
total=$(find organizations/ -type d -maxdepth 3 | wc -l)
with_meta=$(find organizations/ -name "repo.yaml" -path "*/.meta/*" | wc -l)
echo "$with_meta / $total repos have .meta/repo.yaml"

# Check CODEOWNERS coverage
echo ""
echo "=== CODEOWNERS Coverage ==="
with_codeowners=$(find organizations/ -name "CODEOWNERS" -path "*/.github/*" | wc -l)
echo "$with_codeowners / $total repos have CODEOWNERS"

# Check CI coverage
echo ""
echo "=== CI Coverage ==="
with_ci=$(find organizations/ -type d -name "workflows" -path "*/.github/*" | wc -l)
echo "$with_ci / $total repos have CI workflows"

# Run inventory.json generation
echo ""
echo "=== Generating Updated Inventory ==="
# Re-run audit process

# Commit monthly report
echo ""
echo "=== Committing Monthly Report ==="
git add inventory.json gaps.md
git commit -m "chore: monthly Golden Path audit - $(date +%Y-%m)"
```

2. **Schedule:**
   - 1st of every month: Run audit script
   - Review gaps.md changes
   - Triage new gaps
   - Update actions.md priorities

---

## Implementation Roadmap

### Week 1 (P0 Focus)

| Day | Action | Hours | Deliverable |
|-----|--------|-------|-------------|
| Mon | P0-1: mag-logic tests | 8 | Test suite ≥80% |
| Tue | P0-1: spin-circ tests | 8 | Test suite ≥80% |
| Wed | P0-2: HELIOS tests | 8 | Test suite ≥80% |
| Thu | P0-2: TalAI tests | 8 | Test suite ≥80% |
| Fri | P0-3: Archive/exempt repos | 4 | 3 repos handled |
|     | P0-4: Coverage tracking | 4 | 3 libs with coverage |

**Total Week 1:** 40 hours

### Week 2-3 (P0 Completion + P1 Start)

| Days | Action | Hours | Deliverable |
|------|--------|-------|-------------|
| 6-7  | P0-5: Bootstrap CI | 12 | 3 tools with CI |
| 8-10 | P1-1: .meta/repo.yaml (all) | 8 | 34 repos compliant |
| 11-12 | P1-2: CODEOWNERS (all) | 6 | 34 repos compliant |
| 13-14 | P1-3: LICENSE files | 2 | 9 repos compliant |
| 15 | P1-4: SECURITY.md files (start) | 4 | 4 repos done |

**Total Weeks 2-3:** 32 hours

### Week 4 (P1 Continuation)

| Days | Action | Hours | Deliverable |
|------|--------|-------|-------------|
| 16-17 | P1-4: SECURITY.md (finish) | 4 | 16 repos compliant |
| 18-20 | P1-5: Pilot reusable workflows | 12 | repz + live-it-iconic migrated |
| 21 | P1-6: Tool coverage | 4 | 2 tools with coverage |
| 22 | P1-7: CONTRIBUTING.md (start) | 4 | 3 repos done |

**Total Week 4:** 24 hours

### Month 2 (P1 Completion + P2 Start)

| Week | Action | Hours | Deliverable |
|------|--------|-------|-------------|
| 5 | P1-7: CONTRIBUTING.md (finish) | 4 | 15 repos compliant |
|   | P2-1: Batch 1 workflow migration | 4 | 5 scientific libs migrated |
| 6 | P2-1: Batch 2 workflow migration | 4 | 5 tools migrated |
|   | P2-2: Renovate org-wide | 4 | Automated updates live |
| 7 | P2-1: Batch 3 workflow migration | 4 | 5 core platforms migrated |
|   | P2-3: Backstage sync | 6 | Automated catalog |
| 8 | P2-1: Batch 4 workflow migration | 4 | Remaining repos migrated |
|   | P2-4: Pre-commit hooks | 2 | All repos protected |

**Total Month 2:** 32 hours

### Month 3 (P2 Completion)

| Week | Action | Hours | Deliverable |
|------|--------|-------|-------------|
| 9-10 | P2-5: Documentation quality | 16 | Standardized docs |
| 11 | P2-6: Repo rationalization | 8 | Merges/splits complete |
| 12 | P2-7: Issue/PR templates | 4 | Contribution templates |
|    | P2-8: Monthly audit setup | 4 | Ongoing compliance |

**Total Month 3:** 32 hours

---

## Success Metrics

### Immediate (Week 1)

- ✅ 2 libraries with ≥80% test coverage (mag-logic, spin-circ)
- ✅ 2 tools with ≥80% test coverage (HELIOS, TalAI)
- ✅ 3 repos archived/exempted/fixed
- ✅ 3 libraries with coverage tracking

### Short-term (Month 1)

- ✅ 100% repos with `.meta/repo.yaml`
- ✅ 100% repos with `CODEOWNERS`
- ✅ 100% repos with `LICENSE`
- ✅ 100% repos with `SECURITY.md`
- ✅ 2 pilot repos migrated to reusable workflows

### Medium-term (Month 2-3)

- ✅ 100% CI repos calling governance workflows
- ✅ Renovate managing dependencies across portfolio
- ✅ Backstage catalog auto-syncing
- ✅ Pre-commit hooks preventing issues
- ✅ Standardized documentation

### Long-term (Ongoing)

- ✅ Monthly compliance rate >95%
- ✅ Average OpenSSF Scorecard score >8.0
- ✅ Zero repos without required files
- ✅ Automated governance enforcement

---

## Risk Mitigation

### Risk: Breaking Existing CI

**Likelihood:** Medium
**Impact:** High
**Mitigation:**
- Test reusable workflows in pilot repos first
- Keep old workflows in archive initially
- Gradual rollout with verification at each step
- Rollback plan documented

### Risk: Test Coverage Can't Reach ≥80%

**Likelihood:** Medium (for some legacy code)
**Impact:** Medium
**Mitigation:**
- Document exemptions in `.meta/repo.yaml`
- Create remediation plan for legacy code
- Focus on new code coverage
- Consider refactoring if critical

### Risk: Developer Pushback on Governance

**Likelihood:** Low (solo developer)
**Impact:** Low
**Mitigation:**
- Automate as much as possible
- Make compliance easy (templates, scripts)
- Show benefits (security, quality, time savings)

### Risk: Maintenance Burden Too High

**Likelihood:** Medium
**Impact:** High
**Mitigation:**
- Automate governance (Renovate, Backstage sync, monthly audits)
- Use reusable workflows to centralize changes
- Implement pre-commit hooks for early detection
- Document everything for future reference

---

## Appendix A: Batch Scripts

All batch scripts referenced above are available in:
- `.metaHub/scripts/bootstrap/`

To run complete bootstrap:

```bash
cd .metaHub/scripts/bootstrap
./00-master-bootstrap.sh
```

This will execute all P0-P2 actions in sequence with verification at each step.

---

## Appendix B: Validation Checklist

After completing all actions, verify compliance:

```bash
#!/bin/bash
# .metaHub/scripts/validate-compliance.sh

echo "=== Golden Path Compliance Validation ==="

# Required files check
echo ""
echo "=== Required Files ==="
for file in ".meta/repo.yaml" "LICENSE" "SECURITY.md" ".github/CODEOWNERS"; do
  count=$(find organizations/ -path "*/$file" | wc -l)
  echo "$file: $count repos"
done

# CI coverage
echo ""
echo "=== CI Coverage ==="
count=$(find organizations/ -type f -path "*/.github/workflows/*.yml" | wc -l)
echo "Total workflow files: $count"

# Test coverage
echo ""
echo "=== Test Directories ==="
count=$(find organizations/ -type d -name "tests" -o -name "test" | wc -l)
echo "Repos with tests: $count"

# Library/tool compliance
echo ""
echo "=== Library/Tool Test Coverage ==="
# Check each library has tests and coverage
for lib in organizations/alaweimm90-science/*; do
  if [ -f "$lib/pyproject.toml" ]; then
    if [ -d "$lib/tests" ]; then
      echo "✅ $(basename $lib): has tests"
    else
      echo "❌ $(basename $lib): MISSING TESTS"
    fi
  fi
done

echo ""
echo "=== Validation Complete ==="
```

---

## Summary

**Total Effort:** 120-150 hours over 3 months
**P0 Effort:** 60-80 hours (Week 1-2)
**P1 Effort:** 40-50 hours (Week 3-4)
**P2 Effort:** 20-30 hours (Month 2-3)

**Key Milestones:**
- Week 1: All critical test gaps closed
- Month 1: All required files present
- Month 2: All CI standardized
- Month 3: Ongoing compliance automated

**Expected Outcome:**
- 100% Golden Path compliance (from 0%)
- Automated governance enforcement
- Reduced maintenance burden
- Improved security posture
- Portfolio-wide consistency
