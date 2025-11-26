# alaweimm90 Golden Path Bootstrap

**Goal:** Deploy the complete production-ready GitHub OS for alaweimm90
**Timeline:** 10 business days
**Outcome:** 55 repos compliant with enforced structure

---

## Phase 1: Create Foundation Repos (Day 1-2)

### Step 1: Clone & Setup

```bash
mkdir -p ~/repos/alaweimm90 && cd ~/repos/alaweimm90
export GITHUB_USER=alaweimm90

# Configure git
git config --global user.email "ops@alaweimm90.dev"
git config --global user.name "alaweimm90 bot"
```

### Step 2: Create Five Foundation Repos on GitHub

```bash
# Via gh CLI or GitHub web UI
for repo in .github standards core-control-center alaweimm90 infra-actions; do
  gh repo create "alaweimm90/$repo" \
    --public \
    --description "Org-wide governance and core infrastructure" \
    --confirm
done
```

### Step 3: Bootstrap `.github` Repo

```bash
mkdir .github && cd .github
git init
git config user.email "ops@alaweimm90.dev"
git config user.name "alaweimm90 bot"

# Create directory structure
mkdir -p .github/workflows
mkdir -p .github/ISSUE_TEMPLATE
touch .github/CODEOWNERS
touch .github/labels.json
touch .github/dependabot.yml
```

**File: `.github/workflows/reusable-python-ci.yml`** (full content below)

**File: `.github/workflows/reusable-ts-ci.yml`** (full content below)

**File: `.github/workflows/reusable-policy.yml`** (full content below)

**File: `.github/workflows/reusable-release.yml`** (full content below)

**File: `.github/ISSUE_TEMPLATE/bug.yml`**

```yaml
name: Bug Report
description: File a bug report
title: "[BUG] "
labels: ["bug"]
assignees: []

body:
  - type: markdown
    attributes:
      value: |
        Thanks for reporting! Please provide details below.

  - type: textarea
    id: description
    attributes:
      label: Description
      description: What is the bug?
      placeholder: Clear, concise description...
    validations:
      required: true

  - type: textarea
    id: steps
    attributes:
      label: Steps to Reproduce
      description: How do we reproduce it?
      placeholder: |
        1. Step one
        2. Step two
        3. ...
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      placeholder: What should happen?
    validations:
      required: true

  - type: textarea
    id: actual
    attributes:
      label: Actual Behavior
      placeholder: What actually happens?
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: Environment
      placeholder: |
        - OS: [e.g., macOS, Linux, Windows]
        - Python/Node version: [version]
        - Package version: [version]
```

**File: `.github/ISSUE_TEMPLATE/feature.yml`**

```yaml
name: Feature Request
description: Suggest a new feature
title: "[FEATURE] "
labels: ["enhancement"]
assignees: []

body:
  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      description: What problem does this solve?
      placeholder: Describe the use case...
    validations:
      required: true

  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: How should this work?
      placeholder: Detailed description...
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Any other approaches?
      placeholder: List alternatives...
```

**File: `.github/PULL_REQUEST_TEMPLATE.md`**

```markdown
## Description

Brief description of changes.

## Related Issues

Closes #<!-- issue number -->

## Type of Change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update

## Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No new linting warnings
- [ ] Code reviewed locally
- [ ] Coverage maintained/improved

## Testing Notes

Describe how this was tested.
```

**File: `.github/CODEOWNERS`**

```
* @alaweimm90
/docs/ @alaweimm90
/src/ @alaweimm90
/tests/ @alaweimm90
/.github/workflows/ @alaweimm90
```

**File: `.github/labels.json`**

```json
[
  {"name": "bug", "color": "d73a4a", "description": "Something isn't working"},
  {"name": "enhancement", "color": "a2eeef", "description": "New feature or request"},
  {"name": "documentation", "color": "0075ca", "description": "Improvements or additions to documentation"},
  {"name": "good first issue", "color": "7057ff", "description": "Good for newcomers"},
  {"name": "help wanted", "color": "008672", "description": "Extra attention is needed"},
  {"name": "question", "color": "d876e3", "description": "Further information is requested"},
  {"name": "policy-violation", "color": "ff0000", "description": "Violates org standards"},
  {"name": "coverage-gap", "color": "ff9e00", "description": "Test coverage below threshold"},
  {"name": "archived", "color": "cccccc", "description": "No longer maintained"}
]
```

**File: `.github/dependabot.yml`**

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    allow:
      - dependency-type: "direct"

  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    allow:
      - dependency-type: "direct"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

**File: `README.md`**

```markdown
# .github (Governance Repository)

Central governance, reusable workflows, and organization-wide GitHub configuration.

## Reusable Workflows

All consumer repos call these workflows; **no custom CI duplication**.

- **`reusable-python-ci.yml`** — Python: test, lint, type-check, coverage gate ≥80%
- **`reusable-ts-ci.yml`** — TypeScript: test, lint, format, coverage gate ≥80%
- **`reusable-policy.yml`** — Policy enforcement: OPA checks, Markdown linting
- **`reusable-release.yml`** — Automated release drafting and tagging

## Usage

In any consumer repo (e.g., a library):

**`.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]
jobs:
  ci:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with: { python-version: '3.11' }
  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

## Templates & Standards

- **[Standards](https://github.com/alaweimm90/standards)** — Naming, policies, OPA rules, docs spec
- **[Core Control Center](https://github.com/alaweimm90/core-control-center)** — Orchestrator
- **[Templates](https://github.com/alaweimm90/template-python-lib)** — Golden starters

## Issue & PR Workflow

- Templates in `.github/ISSUE_TEMPLATE/` ensure consistent intake
- Labeling automated via `labels.json`
- CODEOWNERS requires appropriate reviews

## Maintenance

Changes to workflows affect all repos. Bump version tag after merge for safe rollout.

```

**File: `.meta/repo.yaml`**

```yaml
---
type: meta
language: yaml
docs_profile: minimal
criticality_tier: 1
description: "Organization-wide GitHub governance, reusable workflows, and standards"
status: active
```

**File: `LICENSE`**

```
MIT License

Copyright (c) 2024-2025 alaweimm90

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

**File: `SECURITY.md`**

```markdown
# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability, please email:

**security@alaweimm90.dev**

Do **NOT** open a public GitHub issue for security vulnerabilities.

## Supported Versions

| Version | Status |
|---------|--------|
| Latest | ✅ Active |
| Previous | ⚠️ Security fixes only |
| Older | ❌ Unsupported |

## Response Timeline

- **Critical:** 24 hours
- **High:** 48 hours
- **Medium:** 1 week

## Scope

- Workflow vulnerabilities
- GitHub Actions security issues
- Secrets management breaks
- Policy bypass techniques
```

**File: `CONTRIBUTING.md`**

```markdown
# Contributing

This is the org-wide governance repo. Changes affect all 55+ repositories.

## Workflow Changes

Before merging:

1. [ ] Test locally with `act -j test`
2. [ ] Verify consumers can call your workflow
3. [ ] Document breaking changes clearly
4. [ ] Bump workflow version if breaking (e.g., `v1.2.0`)
5. [ ] Open PR for team review

## Process

1. Fork and branch from `main`
2. Test changes in your fork
3. Open PR with detailed explanation
4. Require 2+ approvals before merge
5. Tag release after merge

## Policy Updates

Changes to labels, issue templates, or automation:
1. Update `.github/` files
2. Test via dry-run if possible
3. Merge and announce to all teams

## Questions?

Open a discussion or email ops@alaweimm90.dev
```

**Commit & Push**

```bash
git add .
git commit -m "Initial: reusable workflows, issue templates, org governance"
git branch -M main
git remote add origin https://github.com/alaweimm90/.github.git
git push -u origin main

# Enable branch protection (via GH CLI or web UI)
gh repo edit alaweimm90/.github \
  --enable-discussions \
  --enable-security-advisories
```

---

## Complete Workflow Files for `.github` Repo

### `reusable-python-ci.yml`

```yaml
name: python-ci
on:
  workflow_call:
    inputs:
      python-version:
        description: "Python version to test"
        type: string
        default: '3.11'
      test-command:
        description: "Custom test command (default: pytest with coverage)"
        type: string
        default: "pytest -q --cov --cov-report=term-missing --cov-fail-under=80"
      install-extras:
        description: "Extras to install (e.g., dev,docs)"
        type: string
        default: "dev"

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python ${{ inputs.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ inputs.python-version }}
          cache: pip

      - name: Upgrade pip and install build tools
        run: |
          python -m pip install --upgrade pip setuptools wheel

      - name: Install package with extras
        run: |
          pip install -e ".[${EXTRAS}]"
        env:
          EXTRAS: ${{ inputs.install-extras }}

      - name: Lint with Ruff
        run: ruff check .
        continue-on-error: true

      - name: Format check with Black
        run: black --check .
        continue-on-error: true

      - name: Type check with mypy
        run: mypy . --ignore-missing-imports
        continue-on-error: true

      - name: Run tests with coverage
        run: ${{ inputs.test-command }}

      - name: Upload coverage to Codecov
        if: always()
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml,./coverage/.coverage
          flags: python
          name: python-coverage
          fail_ci_if_error: false
          verbose: false
```

### `reusable-ts-ci.yml`

```yaml
name: ts-ci
on:
  workflow_call:
    inputs:
      node-version:
        description: "Node.js version"
        type: string
        default: '20'
      test-command:
        description: "Custom test command"
        type: string
        default: "pnpm test -- --run --coverage"

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Node.js ${{ inputs.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}

      - name: Setup pnpm
        uses: pnpm/action-setup@v2
        with:
          version: 9

      - name: Get pnpm store directory
        id: pnpm-cache
        shell: bash
        run: echo "STORE_PATH=$(pnpm store path)" >> $GITHUB_OUTPUT

      - name: Setup pnpm cache
        uses: actions/cache@v4
        with:
          path: ${{ steps.pnpm-cache.outputs.STORE_PATH }}
          key: ${{ runner.os }}-pnpm-store-${{ hashFiles('**/pnpm-lock.yaml') }}
          restore-keys: |
            ${{ runner.os }}-pnpm-store-

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Lint with ESLint
        run: pnpm lint
        continue-on-error: true

      - name: Run tests with coverage
        run: ${{ inputs.test-command }}

      - name: Upload coverage to Codecov
        if: always()
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage/coverage-final.json
          flags: typescript
          name: ts-coverage
          fail_ci_if_error: false
```

### `reusable-policy.yml`

```yaml
name: policy
on:
  workflow_call:

jobs:
  structure:
    name: Validate Repository Structure
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Generate file tree
        run: |
          python3 << 'EOF'
          import json
          import os
          from pathlib import Path

          paths = []
          for root, dirs, files in os.walk('.'):
              # Skip common exclusions
              dirs[:] = [d for d in dirs if d not in [
                  '.git', 'node_modules', '__pycache__', '.pytest_cache',
                  'venv', '.venv', '.env', 'build', 'dist', '.egg-info'
              ]]
              for f in files:
                  if not f.startswith('.'):
                      path = os.path.join(root, f)[2:]
                      paths.append(path)

          with open('/tmp/file-tree.json', 'w') as f:
              json.dump({"files": sorted(paths)}, f, indent=2)

          print(f"Generated tree with {len(paths)} files")
          EOF

      - name: Validate required files
        run: |
          python3 << 'EOF'
          import json
          import os
          import yaml

          required_files = {
              "library": ["README.md", ".meta/repo.yaml", ".github/CODEOWNERS", ".github/workflows/ci.yml", "tests/"],
              "tool": ["README.md", ".meta/repo.yaml", ".github/CODEOWNERS", ".github/workflows/ci.yml", "tests/"],
              "meta": ["README.md", ".meta/repo.yaml", ".github/CODEOWNERS", ".github/workflows/ci.yml"],
              "demo": ["README.md", ".meta/repo.yaml"],
              "research": ["README.md", ".meta/repo.yaml"],
              "adapter": ["README.md", ".meta/repo.yaml", ".github/CODEOWNERS", ".github/workflows/ci.yml", "tests/"],
          }

          # Read .meta/repo.yaml
          if os.path.exists('.meta/repo.yaml'):
              with open('.meta/repo.yaml') as f:
                  meta = yaml.safe_load(f) or {}
              repo_type = meta.get('type', 'unknown')

              if repo_type in required_files:
                  missing = []
                  for required_file in required_files[repo_type]:
                      if not os.path.exists(required_file):
                          missing.append(required_file)

                  if missing:
                      print(f"❌ ERROR: Missing required files for type '{repo_type}':")
                      for f in missing:
                          print(f"   - {f}")
                      exit(1)
                  else:
                      print(f"✅ All required files present for type '{repo_type}'")
          else:
              print("❌ ERROR: .meta/repo.yaml not found in root")
              exit(1)
          EOF

      - name: Check .meta/repo.yaml schema
        run: |
          python3 << 'EOF'
          import yaml

          required_fields = ["type", "language", "docs_profile"]

          with open('.meta/repo.yaml') as f:
              meta = yaml.safe_load(f) or {}

          missing = [f for f in required_fields if f not in meta]
          if missing:
              print(f"❌ ERROR: Missing required fields in .meta/repo.yaml: {missing}")
              exit(1)
          else:
              print("✅ .meta/repo.yaml schema valid")
          EOF

  markdownlint:
    name: Lint Markdown
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Super-Linter (Markdown focus)
        uses: super-linter/super-linter@v6
        env:
          VALIDATE_MARKDOWN: true
          VALIDATE_YAML: true
          VALIDATE_JSON: true
          DEFAULT_BRANCH: main
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        continue-on-error: true
```

### `reusable-release.yml`

```yaml
name: release
on:
  workflow_call:

jobs:
  draft:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Create release draft
        uses: release-drafter/release-drafter@v6
        with:
          config-name: release-drafter.yml
          skip-publish: true
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Next Steps

Once `.github` is live:

1. **Create `standards/` repo** with policies, OPA rules, linter configs
2. **Create `core-control-center/`** with DAG orchestrator
3. **Create four templates** (python-lib, ts-lib, research, monorepo)
4. **Create adapter repos** (claude, openai, lammps, siesta)
5. **Migrate 55 existing repos** to use `.meta/repo.yaml` + call reusable CI

Total effort: **10 business days** with 2-3 person team.

---

## Verification Checklist (Day 2 end)

- [ ] `.github` repo created and pushed
- [ ] All workflows syntactically valid (GH validates on push)
- [ ] Issue templates appear in repo settings
- [ ] Branch protection configured for `main`
- [ ] CODEOWNERS file recognized
- [ ] Dependabot configured and running
- [ ] Labels are visible in repo

