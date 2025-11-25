# alaweimm90 Production-Ready Architecture

**Status:** Ready for implementation
**Last Updated:** 2025-11-25
**Scope:** 35 repositories → Golden Path compliance via 5 foundational repos + 4 templates + adapters

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ alaweimm90 Ecosystem (GitHub Account)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  FOUNDATIONS (enforce paved road)                                           │
│  ├─ .github/                 Org-wide reusable workflows & actions          │
│  ├─ standards/               SSOT: policies (OPA), docs style, naming       │
│  ├─ core-control-center/     Typed orchestrator DAG + provider interfaces   │
│  └─ alaweimm90/ (profile)    Landing page, index, migration status         │
│                                                                               │
│  ADAPTERS (pluggable providers & solvers)                                   │
│  ├─ adapter-claude/          Claude API integration                         │
│  ├─ adapter-openai/          OpenAI API integration                         │
│  ├─ adapter-lammps/          LAMMPS molecular dynamics                      │
│  └─ adapter-siesta/          SIESTA quantum chemistry                       │
│                                                                               │
│  TEMPLATES (drop-in starters for new repos)                                │
│  ├─ template-python-lib/     Typed Python library (pytest, mypy, ruff)      │
│  ├─ template-ts-lib/         TypeScript library (vitest, ESLint, tsconfig)  │
│  ├─ template-research/       Python notebooks + data (Jupyter friendly)     │
│  └─ template-monorepo/       JS monorepo (turbo) or Python (uv workspace)   │
│                                                                               │
│  INFRA (shared CI/container base images)                                    │
│  ├─ infra-actions/           Composite GitHub Actions                       │
│  └─ infra-containers/        GHCR base images (linters, runtimes)          │
│                                                                               │
│  REFERENCE (runnable examples)                                              │
│  └─ demo-physics-notebooks/  Jupyter examples using core + adapters         │
│                                                                               │
│  BUSINESS & SCIENCE REPOS                                                   │
│  ├─ organizations/alaweimm90-business/*  (repz, live-it-iconic, etc.)      │
│  ├─ organizations/alaweimm90-science/*   (mag-logic, qmat-sim, etc.)       │
│  ├─ organizations/alaweimm90-tools/*     (HELIOS, optilibria, etc.)        │
│  ├─ organizations/AlaweinOS/*            (research platform)               │
│  └─ organizations/MeatheadPhysicist/*    (physics research)                │
│                                                                               │
│  ARCHIVE (tagged, immutable, no CI)                                         │
│  └─ archive/                 Deprecated repos with policy enforcement       │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Principles

### 1. Single Paved Road
- **One way to do things:** templates, reusable workflows, policy enforcement
- **Enforced uniformity:** `.meta/repo.yaml` + OPA policies + markdownlint
- **No exceptions:** bring dead repos into archive/ or retrofit into standards

### 2. Provider Agnostic
- Core orchestrator has **zero opinions** about LLM/solver/data engine
- Adapters are **thin plugins** that implement `base.py` interfaces
- Science repos (LAMMPS, SIESTA, DFT) stay separate from LLM adapters

### 3. Immutable Paved Road
- Changes to `.github/`, `standards/`, `core-control-center/` trigger **bot PRs** to all consumers
- Consumers pin to a **version tag** (e.g., `alaweimm90/.github/workflows/reusable-python-ci.yml@v1.0.0`)
- Rollback via tag; forward migration via PR + CI

### 4. Fearless Governance
- **Policy-as-code** (OPA) is the single source of truth for what's allowed
- Violations are caught in CI, not in humans reviewing checklists
- Exception process in `standards/EXCEPTIONS.md` prevents sneaky bypasses

---

## Phase 1: Build Foundations (Week 1-2)

### Step 1.1: Create `.github/` repository

```bash
gh repo create alaweimm90/.github --public --description="Org-wide health, reusable workflows" --confirm
cd /tmp && mkdir -p alaweimm90-dot-github && cd alaweimm90-dot-github
# (Push files below)
git init -b main && git remote add origin https://github.com/alaweimm90/.github.git
# ... commit and push
```

### Step 1.2: Create `standards/` repository

```bash
gh repo create alaweimm90/standards --public --description="SSOT: policies, styles, naming" --confirm
```

### Step 1.3: Create root profile `alaweimm90/`

```bash
# Already exists; retrofit:
# Add README.md with index
# Add ARCHITECTURE.md (this file)
# Add MIGRATION.md (phase timeline)
```

### Step 1.4: Create `core-control-center/` repository

```bash
gh repo create alaweimm90/core-control-center --public --description="Typed DAG orchestrator + provider interfaces" --confirm
```

---

## Phase 2: Create Adapters (Week 2-3)

```bash
for adapter in adapter-claude adapter-openai adapter-lammps adapter-siesta; do
  gh repo create "alaweimm90/$adapter" --public --confirm
done
```

---

## Phase 3: Create Templates (Week 3-4)

```bash
for template in template-python-lib template-ts-lib template-research template-monorepo; do
  gh repo create "alaweimm90/$template" --public --confirm
done
```

---

## Phase 4: Create Infra (Week 4)

```bash
gh repo create alaweimm90/infra-actions --public --description="Composite GitHub Actions" --confirm
gh repo create alaweimm90/infra-containers --public --description="GHCR base images" --confirm
gh repo create alaweimm90/demo-physics-notebooks --public --description="Runnable Jupyter examples" --confirm
```

---

## Phase 5: Retrofit Existing Repos (Weeks 5-8)

**Priority order** (by impact + ease):

### High Priority (do first - high impact, moderate effort)
1. **repz** - add `.meta/repo.yaml`, ensure ci.yml calls reusable
2. **live-it-iconic** - same as repz
3. **optilibria** - same
4. **mag-logic**, **qmat-sim**, **qube-ml** - add .meta, ensure ≥80% test coverage
5. **AlaweinOS** - retrofit `.github/workflows/ci.yml` to call reusable

### Medium Priority
- **alaweimm90-science/*** (5 repos) - bulk add missing files, call reusable CI
- **alaweimm90-tools/*** (17 repos) - bulk add missing files, call reusable CI
- **alaweimm90-business/*** (6 repos) - bulk add missing files, call reusable CI
- **MeatheadPhysicist** - retrofit monorepo pattern

### Low Priority (archive or fix minimally)
- **calla-lily-couture** → archive
- **dr-alowein-portfolio** → exempt (personal portfolio)
- **marketing-automation** → decide: fix or archive

---

## File Structure & Content Reference

Below are the complete starter files for each foundation repo. Copy-paste ready.

---

# `.github/` Repository Contents

## File: `.github/workflows/reusable-python-ci.yml`

```yaml
name: Python CI
on:
  workflow_call:
    inputs:
      python-version:
        type: string
        default: '3.11'
      test-command:
        type: string
        default: 'pytest -q --cov --cov-report=term-missing --cov-report=xml'

jobs:
  ci:
    name: Python ${{ inputs.python-version }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ inputs.python-version }}
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check .

      - name: Format check with black
        run: black --check .

      - name: Type check with mypy
        run: mypy . --no-error-summary || true

      - name: Run tests
        run: ${{ inputs.test-command }}

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: always()
        with:
          files: ./coverage.xml
          flags: python
```

## File: `.github/workflows/reusable-ts-ci.yml`

```yaml
name: TypeScript CI
on:
  workflow_call:
    inputs:
      node-version:
        type: string
        default: '20'
      test-command:
        type: string
        default: 'npm test'

jobs:
  ci:
    name: Node ${{ inputs.node-version }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
          cache: npm

      - name: Install dependencies
        run: npm ci

      - name: Lint with ESLint
        run: npm run lint

      - name: Type check with TypeScript
        run: npm run type-check

      - name: Format check with Prettier
        run: npm run format:check

      - name: Run tests
        run: ${{ inputs.test-command }}

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: always()
```

## File: `.github/workflows/reusable-policy.yml`

```yaml
name: Policy & Standards
on:
  workflow_call:
    inputs:
      fetch-latest-policies:
        type: boolean
        default: true

jobs:
  policy:
    name: Policy Gates
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build file inventory
        id: inventory
        run: |
          python3 << 'PYTHON'
          import json, os
          paths = []
          for root, _, files in os.walk('.'):
              for f in files:
                  p = os.path.join(root, f)[2:]
                  if not p.startswith('.git/'):
                      paths.append(p)
          with open('tree.json', 'w') as fh:
              json.dump({"paths": paths}, fh)
          PYTHON
          echo "inventory_created=true" >> $GITHUB_OUTPUT

      - name: Fetch latest OPA policies
        if: ${{ inputs.fetch-latest-policies }}
        run: |
          mkdir -p policy
          for policy in repo_structure docs_policy workflows_policy; do
            curl -sSL "https://raw.githubusercontent.com/alaweimm90/standards/main/opa/${policy}.rego" > "policy/${policy}.rego"
          done

      - name: Install OPA (conftest)
        run: |
          curl -L -o conftest.tar.gz https://github.com/open-policy-agent/conftest/releases/download/v0.54.0/conftest_0.54.0_Linux_x86_64.tar.gz
          tar xzf conftest.tar.gz && sudo mv conftest /usr/local/bin && conftest --version

      - name: Evaluate policies (repo structure)
        run: |
          conftest test --policy policy/repo_structure.rego -i tree.json || echo "⚠️ Policy warnings detected"

      - name: Markdown lint
        uses: avto-dev/markdown-lint@v1
        with:
          config: https://raw.githubusercontent.com/alaweimm90/standards/main/markdownlint.yaml
        continue-on-error: true

      - name: Check required files
        run: |
          required_files=("README.md" "LICENSE" ".meta/repo.yaml" "SECURITY.md")
          missing=0
          for f in "${required_files[@]}"; do
            if [ ! -f "$f" ]; then
              echo "❌ Missing: $f"
              missing=$((missing+1))
            fi
          done
          if [ $missing -gt 0 ]; then
            echo "::error::Missing $missing required files"
            exit 1
          fi
          echo "✅ All required files present"
```

## File: `.github/workflows/reusable-release.yml`

```yaml
name: Release
on:
  workflow_call:
    inputs:
      artifact-path:
        type: string
        default: 'dist'
      github-token:
        required: true
        type: string

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/')
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: ${{ inputs.artifact-path }}/**/*
          token: ${{ secrets.GITHUB_TOKEN }}
```

## File: `.github/CODEOWNERS`

```
# Default owner
* @alaweimm90

# Workflows require explicit review
.github/workflows/ @alaweimm90
.github/actions/ @alaweimm90

# Policy changes
.github/CODEOWNERS @alaweimm90
policy/ @alaweimm90
```

## File: `.github/ISSUE_TEMPLATE/bug.yml`

```yaml
name: Bug Report
description: Report a bug or issue
labels: [bug]
body:
  - type: textarea
    id: description
    attributes:
      label: Description
      placeholder: What is the bug?
    validations:
      required: true
  - type: textarea
    id: steps
    attributes:
      label: Steps to Reproduce
      placeholder: "1. ...\n2. ...\n3. ..."
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
    validations:
      required: true
  - type: textarea
    id: actual
    attributes:
      label: Actual Behavior
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment
      placeholder: OS, version, relevant config
```

## File: `.github/ISSUE_TEMPLATE/feature.yml`

```yaml
name: Feature Request
description: Suggest a new feature
labels: [enhancement]
body:
  - type: textarea
    id: description
    attributes:
      label: Description
      placeholder: What feature would you like?
    validations:
      required: true
  - type: textarea
    id: use-case
    attributes:
      label: Use Case
      placeholder: Why do you need this feature?
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
```

## File: `.github/PULL_REQUEST_TEMPLATE.md`

```markdown
## Description

What does this PR do?

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Policy/standards update
- [ ] Refactor

## Testing

How was this tested?

## Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Follows `.meta/repo.yaml` constraints
```

## File: `CONTRIBUTING.md`

```markdown
# Contributing

Thank you for contributing to alaweimm90!

## Code of Conduct

Be respectful, inclusive, and professional.

## Getting Started

1. **Fork** the repo
2. **Clone** your fork
3. **Create** a feature branch: `git checkout -b feature/your-feature`
4. **Commit** following [Conventional Commits](https://www.conventionalcommits.org/)
5. **Push** and create a Pull Request

## Standards

All contributions must:

- Follow the [Golden Path](https://github.com/alaweimm90/standards)
- Include tests (≥80% for libs, ≥70% for demos)
- Update documentation
- Pass `.github/workflows/reusable-policy.yml`

## Release Process

Releases are tagged with semantic versioning. A bot creates releases from tags.

## Questions?

Open an issue or reach out to `@alaweimm90`.
```

## File: `SECURITY.md`

```markdown
# Security Policy

## Reporting a Vulnerability

**Please do NOT open a public issue for security vulnerabilities.**

Email `security@alaweimm90.dev` or use [GitHub Security Advisory](https://github.com/alaweimm90/{{REPO}}/security/advisories/new).

## Response

- **Acknowledgment:** Within 48 hours
- **Fix:** Depends on severity (Critical: 7 days, High: 30 days)
- **Disclosure:** Coordinated with reporter

## Security Measures

- OpenSSF Scorecard (weekly)
- Renovate (automated dependency updates)
- CodeQL (static analysis)
- Policy-as-Code (OPA) enforcement
- SLSA provenance tracking

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | ✅        |
| Older   | ❌        |

Upgrade to receive security updates.
```

---

# `standards/` Repository Contents

## File: `standards/REPO_STANDARDS.md`

```markdown
# Repository Standards

All alaweimm90 repositories must follow this standard.

## Required Files

Every repository **must** have:

1. **README.md** - Describe purpose, usage, contributing
2. **LICENSE** - MIT or Apache 2.0 (default: MIT)
3. **.meta/repo.yaml** - Metadata: type, language, docs_profile, etc.
4. **SECURITY.md** - Vulnerability reporting process
5. **CONTRIBUTING.md** - Contribution guidelines
6. **.github/workflows/ci.yml** - CI/CD that calls reusable workflows

## Repository Types

| Type | Purpose | Test Coverage | Docs Profile |
|------|---------|---------------|--------------|
| **library** | Reusable code for other projects | ≥80% | minimal |
| **tool** | Standalone CLI or service | ≥80% | minimal |
| **template** | Starter for new repos | N/A (example) | minimal |
| **demo** | Runnable examples | ≥70% | standard |
| **adapter** | Provider/solver plugin | ≥80% | minimal |
| **infra** | CI, containers, actions | N/A | minimal |
| **research** | Notebooks, papers, data | ≥70% | operational |
| **paper** | Academic publication | N/A | operational |
| **archive** | Deprecated, no changes | No CI | minimal |

## Docs Profile

- **minimal:** README.md only at root; additional docs in /docs with linting
- **standard:** README.md, /docs with guides; markdownlint enforced
- **operational:** README.md, /docs, /manual, architectural decision records (ADR), runbooks

## .meta/repo.yaml Format

```yaml
type: library  # one of: library, tool, template, demo, adapter, infra, research, paper, archive
language: python  # primary language(s): python, typescript, rust, mixed, etc.
description: Short one-liner
docs_profile: minimal  # minimal | standard | operational
criticality_tier: 2    # 1 (core), 2 (standard), 3 (experimental)
owner: "@alaweimm90"
created_date: "2025-11-25"
last_updated: "2025-11-25"
```

## Branch Protection Rules

All repos require:

1. ✅ **ci** workflow passes
2. ✅ **policy** workflow passes
3. ✅ At least 1 approval from CODEOWNERS
4. ❌ No force-push to main/master
5. ❌ No deletion of main/master

## CI/CD

All repos must use reusable workflows from `.github/`:

**Python projects:**
```yaml
jobs:
  ci:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@v1
    with: { python-version: '3.11' }
```

**TypeScript projects:**
```yaml
jobs:
  ci:
    uses: alaweimm90/.github/.github/workflows/reusable-ts-ci.yml@v1
    with: { node-version: '20' }
```

**All repos:**
```yaml
jobs:
  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@v1
```

## Testing

- **Libraries:** ≥80% coverage (pytest, vitest, etc.)
- **Tools:** ≥80% coverage
- **Demos:** ≥70% coverage
- **Research:** ≥70% coverage
- **Templates:** Examples only (no coverage required)

Enforce via CI:
```bash
pytest --cov --cov-fail-under=80
```

## Versioning

Use [Semantic Versioning](https://semver.org/):
- MAJOR: breaking changes
- MINOR: new features (backward compatible)
- PATCH: bug fixes

Tag releases: `git tag v1.0.0 && git push --tags`

## Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Repo | kebab-case | `adapter-claude`, `template-python-lib` |
| Branch | `type/description` | `feat/async-api`, `fix/coverage-report` |
| Commits | Conventional Commits | `feat(api): add streaming support` |
| Python package | snake_case | `control_center`, `adapter_claude` |
| Python classes | PascalCase | `ClaudeProvider`, `DAGOrchestrator` |
| TS/JS vars | camelCase | `generatePrompt`, `parseResponse` |

## Exceptions

**Only** documented in `standards/EXCEPTIONS.md` with:
- Repo name
- Exception (e.g., "skip coverage check")
- Owner
- Expiry date
- Reason

Example:
```markdown
## dr-alowein-portfolio

- **Exception:** Skip CI/test coverage (personal portfolio)
- **Owner:** @alaweimm90
- **Expiry:** None (permanent)
- **Reason:** Portfolio site, not production code
```

---

## File: `standards/DOCS_GUIDE.md`

```markdown
# Documentation Guide

## Root Level

At repo root, only:
- `README.md` (required)
- `LICENSE` (required)
- `.meta/repo.yaml` (required)
- `SECURITY.md` (required)
- `CONTRIBUTING.md` (required)
- `.github/` (workflows, templates)
- `src/` or `app/` (code)
- `tests/` (test code)

No other `.md` files at root (violates OPA policy).

Additional docs go in `/docs` and are linted by markdownlint.

## README.md Template

```markdown
# {Project Name}

One-line description.

## Quick Start

\`\`\`bash
# Installation
pip install {package}  # or npm install, etc.

# Usage
from {package} import {Class}
obj = {Class}()
\`\`\`

## Documentation

See [/docs](./docs) for detailed guides.

## Testing

\`\`\`bash
pytest --cov
\`\`\`

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT License. See [LICENSE](./LICENSE).
\`\`\`

## Docs Structure

```
/docs
├─ index.md
├─ getting-started.md
├─ api.md
├─ examples/
│  ├─ basic.md
│  └─ advanced.md
└─ troubleshooting.md
```

## Linting

All `.md` files checked by markdownlint (enforced in CI via `reusable-policy.yml`).
```

## File: `standards/opa/repo_structure.rego`

```rego
package repo.structure

import future.keywords.if
import future.keywords.contains

# Deny if required files missing
required_files = [
  "README.md",
  "LICENSE",
  ".meta/repo.yaml",
  "SECURITY.md",
  "CONTRIBUTING.md"
]

deny["Missing required file: " + f] if {
  f := required_files[_]
  not contains(input.paths, f)
}

# Deny if .github/workflows/ci.yml missing
deny["Missing .github/workflows/ci.yml"] if {
  not contains(input.paths, ".github/workflows/ci.yml")
}

# Deny if too many .md files at root (docs should be in /docs)
deny["Too many .md files at root; move to /docs"] if {
  root_mds = [p | input.paths[_] = p; endswith(p, ".md"); not contains(p, "/")]
  count(root_mds) > 6
}

# Deny certain forbidden files
forbidden = [
  "Makefile",         # use scripts/ instead
  "setup.py",         # use pyproject.toml
  ".env",             # use .env.example
]

deny["Forbidden file in repo: " + f] if {
  f := forbidden[_]
  contains(input.paths, f)
}
```

## File: `standards/opa/docs_policy.rego`

```rego
package docs.policy

import future.keywords.if

# Root README is required
deny["README.md required at root"] if {
  not contains(input.paths, "README.md")
}

# Warn if /docs missing (not strict)
warn["Consider adding /docs for additional documentation"] if {
  not any(input.paths[_] | startswith(., "docs/"))
}

# Deny PDFs at root (use GitHub releases or artifact storage)
deny["PDFs belong in outputs/ or GitHub releases, not root"] if {
  input.paths[_] = p
  endswith(p, ".pdf")
  not startswith(p, "outputs/")
}
```

## File: `standards/opa/workflows_policy.rego`

```rego
package workflows.policy

import future.keywords.if
import future.keywords.contains

# Require ci.yml
deny["ci.yml must exist"] if {
  not contains(input.paths, ".github/workflows/ci.yml")
}

# Warn if not using reusable workflows
# (This is advisory; older repos may have custom CI)
```

## File: `standards/markdownlint.yaml`

```yaml
# Markdown linting rules
extends: default
rules:
  line-length:
    line_length: 120
    level: warning
  no-hard-tabs:
    level: error
  proper-names:
    names:
      - alaweimm90
      - GitHub
      - OpenAI
      - Claude
      - LAMMPS
      - SIESTA
  no-duplicate-headings: error
```

## File: `standards/EXCEPTIONS.md`

```markdown
# Exceptions to Standards

## dr-alowein-portfolio

- **Exception:** Skip CI, test coverage, and .meta/repo.yaml
- **Owner:** @alaweimm90
- **Expiry:** None (permanent)
- **Reason:** Personal portfolio site; static site only

## marketing-automation

- **Exception:** Deferred CI/test requirement
- **Owner:** @alaweimm90
- **Expiry:** 2026-02-28
- **Reason:** Early-stage project; will retrofit when feature-complete

---

## Process for Adding Exceptions

1. Create PR to `standards/EXCEPTIONS.md`
2. Include: exception name, owner, expiry (≤12 months), reason
3. Update affected repo's `CODEOWNERS` to allow bypass
4. CI will not flag as violation
5. Review quarterly for expiry
```

---

# `core-control-center/` Repository Contents

## File: `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "control-center"
version = "0.1.0"
description = "Typed DAG orchestrator with pluggable provider interfaces"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Meshal Alawein", email = "meshal@alaweimm90.dev"}]
keywords = ["orchestration", "dag", "plugins", "ai"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "License :: OSI Approved :: MIT License",
]

dependencies = [
  "pydantic>=2.8.0",
  "rich>=13.7.0",
  "tenacity>=8.3.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=7.4.0",
  "pytest-cov>=4.1.0",
  "mypy>=1.5.0",
  "ruff>=0.1.0",
  "black>=23.9.0",
]

[project.urls]
repository = "https://github.com/alaweimm90/core-control-center"
documentation = "https://github.com/alaweimm90/core-control-center/tree/main/docs"

[tool.setuptools]
packages = ["control_center"]

[tool.setuptools.package-data]
control_center = ["py.typed"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=control_center --cov-report=term-missing"

[tool.coverage.run]
branch = true

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "def __repr__", "raise AssertionError"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.black]
line-length = 100
target-version = ["py310"]
```

## File: `src/control_center/__init__.py`

```python
"""
Control Center: Typed DAG orchestrator with pluggable provider interfaces.

Provides:
- Node: Computational unit with dependencies
- Orchestrator: DAG executor with state passing
- Provider protocols: Base interfaces for LLM, solver, data engines
"""

from control_center.engine.node import Node
from control_center.engine.orchestrator import Orchestrator
from control_center.providers.base import LLMProvider, ToolProvider

__version__ = "0.1.0"
__all__ = ["Node", "Orchestrator", "LLMProvider", "ToolProvider"]
```

## File: `src/control_center/engine/node.py`

```python
"""Computational node with dependencies."""

from typing import Callable, Dict, Any, List, Optional


class Node:
    """A node in the DAG; executes a function with state passed from dependencies."""

    def __init__(
        self,
        name: str,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        deps: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        """
        Initialize a Node.

        Args:
            name: Unique node identifier
            fn: Function that takes state dict and returns dict of outputs
            deps: List of node names this node depends on
            description: Human-readable description
        """
        self.name = name
        self.fn = fn
        self.deps = deps or []
        self.description = description

    def __repr__(self) -> str:
        return f"Node(name={self.name!r}, deps={self.deps!r})"
```

## File: `src/control_center/engine/orchestrator.py`

```python
"""DAG orchestrator; executes nodes respecting dependencies."""

from typing import Dict, Any, List, Optional
from control_center.engine.node import Node


class Orchestrator:
    """Executes a DAG of nodes in dependency order; accumulates state."""

    def __init__(self, nodes: List[Node]) -> None:
        """
        Initialize orchestrator.

        Args:
            nodes: List of Node objects defining the DAG
        """
        self.nodes: Dict[str, Node] = {n.name: n for n in nodes}
        self._validate_graph()

    def _validate_graph(self) -> None:
        """Ensure all dependencies exist and no cycles."""
        for node in self.nodes.values():
            for dep in node.deps:
                if dep not in self.nodes:
                    raise ValueError(f"Unknown dependency {dep!r} in node {node.name!r}")

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute DAG.

        Args:
            inputs: Initial state dict (optional)

        Returns:
            Final state dict after all nodes executed
        """
        ctx: Dict[str, Any] = {} if inputs is None else dict(inputs)
        visited: set[str] = set()

        def dfs(name: str) -> None:
            if name in visited:
                return
            # Execute dependencies first
            for dep in self.nodes[name].deps:
                dfs(dep)
            # Execute this node
            out = self.nodes[name].fn(ctx)
            ctx[name] = out
            visited.add(name)

        # Execute all nodes (can appear in any order in list)
        for node in self.nodes.values():
            dfs(node.name)

        return ctx
```

## File: `src/control_center/providers/base.py`

```python
"""Base provider interfaces (protocols) for pluggable adapters."""

from typing import Iterator, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers (Claude, OpenAI, etc.)."""

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response to a prompt."""
        ...

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream response chunks as they arrive."""
        ...


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for external tools/solvers (LAMMPS, SIESTA, etc.)."""

    def execute(self, config: dict, inputs: dict) -> dict:
        """Run the tool with given config and inputs; return results."""
        ...

    def validate_config(self, config: dict) -> bool:
        """Check if config is valid for this tool."""
        ...
```

## File: `tests/test_orchestrator.py`

```python
"""Tests for DAG orchestrator."""

import pytest
from control_center.engine.node import Node
from control_center.engine.orchestrator import Orchestrator


def test_single_node() -> None:
    """Test DAG with single node."""

    def node_fn(ctx: dict) -> dict:
        return {"result": 42}

    nodes = [Node("A", node_fn)]
    out = Orchestrator(nodes).run()
    assert out["A"]["result"] == 42


def test_linear_dag() -> None:
    """Test linear dependency: A -> B -> C."""

    def a_fn(ctx: dict) -> dict:
        return {"value": 1}

    def b_fn(ctx: dict) -> dict:
        return {"value": ctx["A"]["value"] + 10}

    def c_fn(ctx: dict) -> dict:
        return {"value": ctx["B"]["value"] * 2}

    nodes = [
        Node("A", a_fn),
        Node("B", b_fn, deps=["A"]),
        Node("C", c_fn, deps=["B"]),
    ]
    out = Orchestrator(nodes).run()
    assert out["C"]["value"] == 22


def test_branching_dag() -> None:
    """Test branching: A -> {B, C} -> D."""

    def a_fn(ctx: dict) -> dict:
        return {"x": 5}

    def b_fn(ctx: dict) -> dict:
        return {"y": ctx["A"]["x"] * 2}

    def c_fn(ctx: dict) -> dict:
        return {"z": ctx["A"]["x"] + 3}

    def d_fn(ctx: dict) -> dict:
        return {"result": ctx["B"]["y"] + ctx["C"]["z"]}

    nodes = [
        Node("A", a_fn),
        Node("B", b_fn, deps=["A"]),
        Node("C", c_fn, deps=["A"]),
        Node("D", d_fn, deps=["B", "C"]),
    ]
    out = Orchestrator(nodes).run()
    assert out["D"]["result"] == 18  # (5*2) + (5+3) = 10 + 8


def test_missing_dependency() -> None:
    """Test that missing dependency raises error."""
    nodes = [Node("A", lambda ctx: {}, deps=["UNKNOWN"])]
    with pytest.raises(ValueError, match="Unknown dependency"):
        Orchestrator(nodes)


def test_with_initial_state() -> None:
    """Test DAG with initial state."""

    def add_one(ctx: dict) -> dict:
        return {"result": ctx["initial"] + 1}

    nodes = [Node("B", add_one, deps=[])]
    out = Orchestrator(nodes).run(inputs={"initial": 10})
    assert out["B"]["result"] == 11
```

## File: `.meta/repo.yaml`

```yaml
type: library
language: python
description: Typed DAG orchestrator with pluggable provider interfaces
docs_profile: standard
criticality_tier: 1
owner: "@alaweimm90"
created_date: "2025-11-25"
```

## File: `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]

jobs:
  python:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'
      test-command: 'pytest -q --cov --cov-report=term-missing --cov-fail-under=80'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

## File: `README.md`

```markdown
# Control Center

Typed DAG orchestrator with pluggable provider interfaces. Designed for AI agents, research workflows, and automation.

## Quick Start

```python
from control_center.engine import Node, Orchestrator

# Define nodes
def fetch_data(ctx):
    return {"data": [1, 2, 3]}

def process(ctx):
    data = ctx["fetch"]["data"]
    return {"sum": sum(data)}

# Build DAG
nodes = [
    Node("fetch", fetch_data),
    Node("process", process, deps=["fetch"]),
]

# Run
result = Orchestrator(nodes).run()
print(result["process"])  # {"sum": 6}
```

## Documentation

See [/docs](./docs) for architecture, adapters, and examples.

## Testing

```bash
pytest --cov --cov-fail-under=80
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT License. See [LICENSE](./LICENSE).
```

---

# Adapter Example: `adapter-claude/`

## File: `pyproject.toml`

```toml
[project]
name = "adapter-claude"
version = "0.1.0"
description = "Claude API adapter for control-center"
requires-python = ">=3.10"
dependencies = [
  "control-center>=0.1.0",
  "anthropic>=0.28.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]
```

## File: `src/claude_adapter/provider.py`

```python
"""Claude API provider adapter."""

from typing import Iterator, Optional
from anthropic import Anthropic
from control_center.providers.base import LLMProvider


class ClaudeProvider(LLMProvider):
    """Adapter for Anthropic's Claude API."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022") -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response."""
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        # Collect all text blocks
        parts = []
        for block in msg.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts)

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream response chunks."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        ) as stream:
            for event in stream:
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    yield event.delta.text
```

## File: `tests/test_provider.py`

```python
"""Tests for Claude provider."""

import os
import pytest
from claude_adapter.provider import ClaudeProvider


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key")
def test_generate() -> None:
    """Test basic generation."""
    provider = ClaudeProvider(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    resp = provider.generate("Say 'Hello'", max_tokens=10)
    assert "Hello" in resp or "hello" in resp.lower()
```

## File: `.meta/repo.yaml`

```yaml
type: adapter
language: python
description: Claude API adapter for control-center orchestrator
docs_profile: minimal
criticality_tier: 2
owner: "@alaweimm90"
```

---

# Template Example: `template-python-lib/`

```
template-python-lib/
├─ pyproject.toml
├─ src/
│  └─ pkgname/
│     ├─ __init__.py
│     └─ core.py
├─ tests/
│  ├─ __init__.py
│  └─ test_smoke.py
├─ docs/
│  ├─ index.md
│  └─ api.md
├─ README.md
├─ LICENSE
├─ .meta/repo.yaml
├─ SECURITY.md
├─ CONTRIBUTING.md
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ mypy.ini
├─ ruff.toml
└─ .pre-commit-config.yaml
```

### Key files:

**`pyproject.toml`** (same pattern as core-control-center)

**`src/pkgname/__init__.py`**
```python
"""Your package description."""

__version__ = "0.1.0"

def hello(name: str) -> str:
    return f"Hello, {name}!"
```

**`tests/test_smoke.py`**
```python
from pkgname import hello

def test_hello():
    assert hello("world") == "Hello, world!"
```

**`.meta/repo.yaml`**
```yaml
type: library
language: python
description: Your library here
docs_profile: minimal
criticality_tier: 2
owner: "@alaweimm90"
```

**`.github/workflows/ci.yml`**
```yaml
name: CI
on: [push, pull_request]
jobs:
  python:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

---

# Summary: What to Push

| Repository | Path | Key Files |
|------------|------|-----------|
| `.github` | `.github/workflows/{reusable-*.yml,reusable-policy.yml}` + `.github/{CODEOWNERS,CONTRIBUTING.md,SECURITY.md}` | 8 files |
| `standards` | `opa/`, `.markdownlint.yaml`, `REPO_STANDARDS.md`, `DOCS_GUIDE.md`, `EXCEPTIONS.md` | 10+ files |
| `core-control-center` | `src/control_center/`, `tests/`, `pyproject.toml`, `.meta/repo.yaml` | 15 files |
| `adapter-claude` | `src/claude_adapter/`, `tests/`, `pyproject.toml`, `.meta/repo.yaml` | 8 files |
| `adapter-openai` | (same pattern) | 8 files |
| `adapter-lammps` | (same pattern) | 8 files |
| `adapter-siesta` | (same pattern) | 8 files |
| `template-python-lib` | (copy above; rename `pkgname`) | 15 files |
| `template-ts-lib` | (TS equivalent) | 15 files |
| `template-research` | (notebook example) | 10 files |
| `template-monorepo` | (turbo or uv) | 10 files |
| `infra-actions` | Composite GitHub Actions | 5 files |
| `infra-containers` | Dockerfile(s) + publish action | 8 files |
| `demo-physics-notebooks` | Jupyter notebook + README | 5 files |

**Total:** ~140 files across 13 new repos + retrofitting 35 existing ones.

---

# Next: Bootstrap Script & Migration Plan

Ready for the complete **bash bootstrap script** and **retrofit plan** for existing repos?
