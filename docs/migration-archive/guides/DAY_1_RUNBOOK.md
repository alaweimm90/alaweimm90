# Day 1 Runbook: alaweimm90 Golden Path Deployment

**Goal:** Stand up `.github`, `standards`, and `core-control-center` with working CI and policy enforcement.
**Time:** 4-6 hours
**Team:** 1-2 people
**Outcome:** Three foundation repos live, workflows validated, branch protection enabled.

---

## Pre-Flight (30 min)

### Prerequisites
```bash
# Verify tools
gh --version          # GitHub CLI 2.0+
git --version         # 2.30+
python3 --version     # 3.10+
pip install pyyaml    # For any YAML parsing

# Configure git
git config --global user.email "ops@alaweimm90.dev"
git config --global user.name "alaweimm90 bot"

# Test GitHub access
gh auth status
```

### Decisions
- [ ] All three repos created on GitHub? (or using `gh repo create`)
- [ ] Team member assigned to each repo?
- [ ] Slack/Discord channel for real-time sync?

---

## Step 1: Create `.github` Repo (1 hour)

### 1a. GitHub Setup
```bash
gh repo create alaweimm90/.github \
  --public \
  --description "Org-wide governance, reusable workflows, issue templates" \
  --confirm

# Verify
gh repo view alaweimm90/.github
```

### 1b. Local Clone & Init
```bash
cd ~/repos/alaweimm90
git clone https://github.com/alaweimm90/.github.git
cd .github

git config user.email "ops@alaweimm90.dev"
git config user.name "alaweimm90 bot"
```

### 1c. Create Directory Structure
```bash
mkdir -p .github/workflows
mkdir -p .github/ISSUE_TEMPLATE
touch .github/CODEOWNERS
touch .github/labels.json
touch .github/dependabot.yml
```

### 1d. Copy Files from BOOTSTRAP.md

**File: `.github/workflows/reusable-python-ci.yml`**
- Copy entire contents from BOOTSTRAP.md section "reusable-python-ci.yml"
- Verify: `name: python-ci`, `on: workflow_call`, has pytest + ruff + black + mypy

**File: `.github/workflows/reusable-ts-ci.yml`**
- Copy entire contents from BOOTSTRAP.md
- Verify: `name: ts-ci`, pnpm, vitest, ESLint

**File: `.github/workflows/reusable-policy.yml`**
- Copy entire contents from BOOTSTRAP.md
- Verify: `name: policy`, generates file tree, runs Conftest, markdownlint

**File: `.github/workflows/reusable-release.yml`**
- Copy entire contents from BOOTSTRAP.md
- Verify: `name: release`, release-drafter

**File: `.github/ISSUE_TEMPLATE/bug.yml`**
```yaml
name: Bug Report
description: File a bug report
title: "[BUG] "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: Thanks for reporting!
  - type: textarea
    id: description
    attributes:
      label: Description
      placeholder: Clear description...
    validations:
      required: true
  - type: textarea
    id: steps
    attributes:
      label: Steps to Reproduce
      placeholder: "1. ...\n2. ..."
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
```

**File: `.github/ISSUE_TEMPLATE/feature.yml`**
```yaml
name: Feature Request
description: Suggest a feature
title: "[FEATURE] "
labels: ["enhancement"]
body:
  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      placeholder: What problem does this solve?
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
    validations:
      required: true
```

**File: `.github/PULL_REQUEST_TEMPLATE.md`**
```markdown
## Description
<!-- Brief description -->

## Related Issues
Closes #<!-- issue number -->

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Docs

## Checklist
- [ ] Tests added
- [ ] Docs updated
- [ ] No warnings
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
  {"name": "documentation", "color": "0075ca", "description": "Docs improvements"},
  {"name": "good first issue", "color": "7057ff", "description": "Good for newcomers"},
  {"name": "help wanted", "color": "008672", "description": "Extra attention needed"},
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
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 1e. Root-Level Files

**File: `README.md`**
```markdown
# .github (Org Governance)

Reusable workflows, templates, and policies for all alaweimm90 repos.

## Reusable Workflows

All consumer repos call these workflows—**no custom CI duplication**.

- **reusable-python-ci.yml** — Python: test, lint, type-check, coverage gate
- **reusable-ts-ci.yml** — TypeScript: test, lint, coverage gate
- **reusable-policy.yml** — Enforce standards: OPA, Markdown linting
- **reusable-release.yml** — Automated release drafting

## Usage (in any consumer repo)

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  python:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with: { python-version: '3.11' }
  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

## Standards & Governance

See [standards](https://github.com/alaweimm90/standards) for the SSOT (policies, OPA rules, linter configs, docs rules, naming taxonomy).

## Maintenance

Changes here affect all 55+ repos. Require 2+ approvals before merge.
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
```

**File: `SECURITY.md`**
```markdown
# Security Policy

## Reporting Vulnerabilities

Email: **security@alaweimm90.dev**

Do NOT open public issues for security vulnerabilities.

## Response SLA

- Critical: 24 hours
- High: 48 hours
- Medium: 1 week

## Scope

Vulnerabilities in workflows, Actions, secrets management, or policy bypass.
```

**File: `CONTRIBUTING.md`**
```markdown
# Contributing

This repo affects all 55+ repos in the portfolio. Test carefully.

## Workflow Changes

1. [ ] Test locally with `act -j test` (if installed)
2. [ ] Verify syntax in GitHub UI before push
3. [ ] Document breaking changes clearly
4. [ ] Bump version tag after merge if breaking

## Process

1. Fork → branch → test locally
2. Open PR with detailed explanation
3. Require 2+ approvals before merge
4. Tag release (e.g., v1.2.0) after merge

## Questions?

Post to org discussions or email ops@alaweimm90.dev
```

**File: `.meta/repo.yaml`**
```yaml
---
type: meta
language: yaml
docs_profile: minimal
criticality_tier: 1
description: "Organization-wide GitHub governance and reusable workflows"
status: active
owner: alaweimm90
```

### 1f. Commit & Push
```bash
git add .
git commit -m "Initial: reusable workflows, issue templates, governance

- Add reusable-python-ci.yml (test, lint, type-check, coverage)
- Add reusable-ts-ci.yml (test, lint, coverage)
- Add reusable-policy.yml (OPA, Markdown linting)
- Add reusable-release.yml (release drafting)
- Add issue templates (bug, feature)
- Add PR template and CODEOWNERS
- Add labels, Dependabot config, governance docs

This is the single source of truth for org-wide CI/policy.
All consumer repos call these reusable workflows.

Requires 2+ approvals to merge; testing required.

Co-authored-by: alaweimm90 <ops@alaweimm90.dev>"

git branch -M main
git remote set-url origin https://github.com/alaweimm90/.github.git
git push -u origin main
```

### 1g. Enable Branch Protection
```bash
# Via GitHub CLI (requires owner access) OR web UI:
# Settings → Branches → Add rule
# Pattern: main
# Require status checks: (will be empty until workflows run, add later)
# Require code owner reviews: yes
# Require up-to-date branches: yes
# Dismiss stale reviews: yes
```

### 1h. Verification
```bash
# Check workflows are recognized
gh workflow list -R alaweimm90/.github
# Expected: python-ci, ts-ci, policy, release (all ACTIVE)

# Check repo was created correctly
gh repo view alaweimm90/.github --json description,visibility
```

---

## Step 2: Create `standards` Repo (1 hour)

### 2a. GitHub Setup
```bash
gh repo create alaweimm90/standards \
  --public \
  --description "Single source of truth: policies, OPA rules, linter configs, naming, docs standards" \
  --confirm
```

### 2b. Clone & Structure
```bash
cd ~/repos/alaweimm90
git clone https://github.com/alaweimm90/standards.git
cd standards

mkdir -p docs opa linters templates
```

### 2c. Minimal SSOT Files

**File: `docs/AI-SPECS.md`**
```markdown
# AI/Org Specifications

## Principles

- **Vendor-neutral:** No cloud lock-in. Adapters are optional.
- **Typed & tested:** ≥80% coverage (libs/tools), ≥70% (demos).
- **Deterministic:** Outputs → `outputs/YYYYMMDD-HHMMSS/`
- **Provider-agnostic:** Core logic; swap Claude ↔ OpenAI ↔ local.
- **Modular:** One concern per module; clear interfaces.

## Implementation Standards

### Python
- Framework: FastAPI (servers), Click (CLI), Pydantic (validation)
- Testing: pytest + pytest-cov
- Linting: ruff, black, mypy (strict)

### TypeScript
- Framework: React (UI), tsup (bundling)
- Testing: Vitest + Vitest coverage
- Linting: ESLint, Prettier (strict)

### Release Model
- Semantic versioning (major.minor.patch)
- Automated release via release-drafter on tag
- Registries: PyPI (Python), npm (Node.js)
```

**File: `docs/NAMING.md`**
```markdown
# Repo Naming Taxonomy

## Enforced Prefixes

| Prefix | Purpose | Example |
|--------|---------|---------|
| `core-` | Orchestration/engines | core-control-center |
| `lib-` | Reusable libraries | lib-quantum, lib-ml |
| `adapter-` | External service wrappers | adapter-claude |
| `tool-` | CLI tools | tool-migration |
| `template-` | Repo starters | template-python-lib |
| `demo-` | Examples/notebooks | demo-quantum-101 |
| `infra-` | CI/containers/IaC | infra-actions |
| `paper-` | Reproducible research | paper-optimization |

## Format Rules

- **Case:** lowercase-kebab-case (no spaces, underscores, mixed case)
- **Length:** 3–50 characters after prefix
- **Uniqueness:** No duplicates across all orgs
```

**File: `docs/DOCS_GUIDE.md`**
```markdown
# Documentation Standards

## Root-Level Files (Allowed)

Every repo MUST have:
1. README.md (overview, quick-start, badges)
2. LICENSE (legal permissions)
3. .meta/repo.yaml (governance metadata)
4. .github/CODEOWNERS (ownership)
5. .github/workflows/ci.yml (calls reusable)

Optional but recommended:
6. CONTRIBUTING.md (contribution guidelines)
7. SECURITY.md (vulnerability reporting)

## Long-Form Documentation

Place in `/docs` folder:
- API.md
- ARCHITECTURE.md
- INSTALLATION.md
- CHANGELOG.md
- FAQ.md

Root directory must stay clean. Enforce via policy (OPA).
```

**File: `.meta/repo.yaml`**
```yaml
---
type: meta
language: yaml
docs_profile: minimal
criticality_tier: 1
description: "Single source of truth: policies, standards, OPA rules, linter configs"
status: active
owner: alaweimm90
```

**File: `README.md`**
```markdown
# standards (SSOT)

Single Source of Truth for org-wide policies, standards, and configurations.

## Contents

- **docs/** — Policy documentation (AI-SPECS, NAMING, DOCS_GUIDE, GOVERNANCE)
- **opa/** — Conftest/OPA rules (repo_structure.rego, coverage.rego, naming.rego)
- **linters/** — Shared configs (ruff, black, eslint, markdownlint, prettier)
- **templates/** — Standard templates (.meta/repo.yaml, README, CONTRIBUTING, SECURITY)

## Usage

1. Consume linter configs in your repos (reference via URL or copy)
2. All repos must pass OPA checks (enforced in reusable-policy.yml from .github)
3. Use templates when creating new repos (via template repos or manual copy)

## Maintenance

Changes here affect ALL repos. Require 2+ approvals. Version tags enable safe rollout.
```

**File: `LICENSE`**
(Same as .github)

### 2d. Commit & Push
```bash
git add .
git commit -m "Initial: SSOT policies and standards

- Add AI-SPECS (org principles: vendor-neutral, typed, tested)
- Add NAMING (enforced prefix taxonomy)
- Add DOCS_GUIDE (root file rules, long-form in /docs)
- Scaffold OPA policies and linter configs

See .github for reusable workflows that enforce these standards.

Co-authored-by: alaweimm90 <ops@alaweimm90.dev>"

git branch -M main
git remote set-url origin https://github.com/alaweimm90/standards.git
git push -u origin main
```

---

## Step 3: Create `core-control-center` Repo (1 hour)

### 3a. GitHub Setup
```bash
gh repo create alaweimm90/core-control-center \
  --public \
  --description "Vendor-neutral DAG orchestration engine for agents, tools, and providers" \
  --confirm
```

### 3b. Clone & Structure
```bash
cd ~/repos/alaweimm90
git clone https://github.com/alaweimm90/core-control-center.git
cd core-control-center

mkdir -p src/control_center/{engine,providers,agents,tools} tests
```

### 3c. Python Package Files

**File: `pyproject.toml`**
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "control-center"
version = "0.1.0"
description = "Vendor-neutral DAG orchestration for agents/tools/providers"
requires-python = ">=3.10"
dependencies = [
  "pydantic>=2.8.0",
  "rich>=13.7.0",
  "tenacity>=8.3.0",
]
[project.optional-dependencies]
dev = ["pytest", "pytest-cov>=5", "mypy", "ruff", "black"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --cov=control_center --cov-report=term-missing --cov-fail-under=80"

[tool.coverage.run]
source = ["control_center"]
omit = ["*/tests/*", "*/venv/*"]
```

**File: `src/control_center/__init__.py`**
```python
__version__ = "0.1.0"
```

**File: `src/control_center/providers/base.py`**
```python
from typing import Protocol, Iterator, Optional

class LLMProvider(Protocol):
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024) -> str: ...
    def stream(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024) -> Iterator[str]: ...
```

**File: `src/control_center/tools/base.py`**
```python
from typing import Protocol, Dict, Any

class Tool(Protocol):
    name: str
    def run(self, **kwargs) -> Dict[str, Any]: ...
```

**File: `src/control_center/agents/base.py`**
```python
from typing import Protocol, Dict, Any

class Agent(Protocol):
    name: str
    def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]: ...
```

**File: `src/control_center/engine/node.py`**
```python
from typing import Callable, Dict, Any, List

class Node:
    def __init__(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]], deps: List[str] | None = None):
        self.name = name
        self.fn = fn
        self.deps = deps or []
```

**File: `src/control_center/engine/orchestrator.py`**
```python
from __future__ import annotations
from typing import Dict, Any, List
from .node import Node

class Orchestrator:
    def __init__(self, nodes: List[Node]):
        self.nodes = {n.name: n for n in nodes}

    def run(self, inputs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {} if inputs is None else dict(inputs)
        visited: set[str] = set()

        def dfs(name: str) -> None:
            if name in visited:
                return
            for d in self.nodes[name].deps:
                dfs(d)
            out = self.nodes[name].fn(ctx)
            ctx[name] = out
            visited.add(name)

        for n in self.nodes.values():
            dfs(n.name)

        return ctx
```

**File: `tests/test_orchestrator.py`**
```python
from control_center.engine.node import Node
from control_center.engine.orchestrator import Orchestrator

def test_dag_order_and_state():
    def a(ctx): return {"a": 1}
    def b(ctx): return {"b": ctx["A"]["a"] + 1}
    nodes = [Node("A", a), Node("B", b, deps=["A"])]
    out = Orchestrator(nodes).run()
    assert out["B"]["b"] == 2
```

### 3d. Root Files

**File: `README.md`**
```markdown
# core-control-center

Vendor-neutral DAG orchestration engine. No provider lock-in—adapters are optional.

## Features

- **DAG execution:** Topologically sort and run node functions with automatic dependency resolution
- **Provider protocol:** Define LLMProvider, Agent, Tool interfaces; swap implementations
- **Orchestrator:** Pass context (state) through DAG, collect outputs

## Quick Start

```python
from control_center.engine.node import Node
from control_center.engine.orchestrator import Orchestrator

def task_a(ctx):
    return {"result": "hello"}

def task_b(ctx):
    return {"result": ctx["A"]["result"] + " world"}

nodes = [
    Node("A", task_a),
    Node("B", task_b, deps=["A"])
]

orch = Orchestrator(nodes)
output = orch.run()
print(output["B"]["result"])  # hello world
```

## Adapters

See `adapter-claude`, `adapter-openai`, `adapter-lammps` for examples.
```

**File: `.meta/repo.yaml`**
```yaml
---
type: core
language: python
docs_profile: minimal
criticality_tier: 1
description: "Vendor-neutral DAG orchestration engine"
status: active
owner: alaweimm90
```

**File: `.github/workflows/ci.yml`**
```yaml
name: ci
on: [push, pull_request]
jobs:
  python:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with: { python-version: '3.11' }
  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

**File: `.github/workflows/policy.yml`**
```yaml
name: policy
on: [pull_request]
jobs:
  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

**File: `.github/CODEOWNERS`**
```
* @alaweimm90
/src/ @alaweimm90
/tests/ @alaweimm90
```

**File: `LICENSE`**
(Same as .github)

### 3e. Commit & Push
```bash
git add .
git commit -m "Initial: vendor-neutral orchestration engine

- Add DAG orchestrator (topologically sorts and executes nodes)
- Add provider/agent/tool base protocols
- Add example orchestrator test

This is the core engine for agent coordination.
Adapters (Claude, OpenAI, LAMMPS, etc.) implement optional interfaces.
No provider lock-in.

Co-authored-by: alaweimm90 <ops@alaweimm90.dev>"

git branch -M main
git remote set-url origin https://github.com/alaweimm90/core-control-center.git
git push -u origin main
```

### 3f. Verify CI Runs
```bash
# Wait 30s for GitHub to process
sleep 30

# Check workflow status
gh run list -R alaweimm90/core-control-center --limit 1

# Expected: python-ci and policy both pass
# If not, check logs:
gh run view <run-id> -R alaweimm90/core-control-center --log
```

---

## Final Verification (30 min)

### Checklist: Day 1 Success
```
[ ] .github repo created and pushed
[ ] All 4 reusable workflows present and syntactically valid
[ ] Issue templates appear in repo
[ ] Branch protection enabled (manual or GH CLI)
[ ] Dependabot configured

[ ] standards repo created and pushed
[ ] AI-SPECS, NAMING, DOCS_GUIDE present
[ ] OPA policies scaffolded (even if empty)
[ ] Linter configs present (even if empty)

[ ] core-control-center repo created and pushed
[ ] Orchestrator test passes locally (`pip install -e .[dev] && pytest`)
[ ] CI ran successfully (both python-ci and policy jobs pass)
[ ] .meta/repo.yaml present

[ ] All three repos have:
  - README.md
  - LICENSE
  - .meta/repo.yaml
  - .github/CODEOWNERS
  - .github/workflows/ci.yml
  - .github/workflows/policy.yml (where applicable)

[ ] Team trained: everyone knows where to find docs
```

### Commands to Verify
```bash
# Verify all three repos exist
gh repo list alaweimm90 --limit 10 | grep -E '\.github|standards|core-control-center'

# Verify workflows are active
for repo in .github standards core-control-center; do
  echo "=== $repo ==="
  gh workflow list -R "alaweimm90/$repo"
done

# Verify CI runs completed
gh run list -R alaweimm90/core-control-center --limit 3
gh run list -R alaweimm90/standards --limit 3
```

---

## If Something Fails

### Workflow YAML Syntax Error
```bash
# Validate locally
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"

# Or use GitHub Actions validator:
# https://github.com/actions/starter-workflows
```

### Coverage Gate Failure in core-control-center
```bash
# Run tests locally
cd core-control-center
pip install -e ".[dev]"
pytest --cov=control_center --cov-report=term-missing

# If coverage < 80%, add tests to tests/
```

### Policy Workflow Fails (OPA)
- Expected: OPA rules are empty stubs; should pass vacuously
- If it fails, check policy job output for YAML parsing errors
- Workaround: Disable policy workflow temporarily, enable after standards repo is fully seeded

### Dependabot Doesn't Show Up
- Normal: takes up to 24 hours to index
- Can manually trigger via Settings → Code security & analysis

---

## Hand-Off to Days 2–10

**Day 2:** Repeat for `alaweimm90` (profile) and validate templates can be cloned.

**Days 3–5:** Create 4 templates + 4 adapters (follow same pattern).

**Days 6–10:** Run `MIGRATION_SCRIPT.py`, commit to all 55 repos, validate CI, archive stale repos.

---

## Post-Day-1 Sync

```markdown
### Day 1 Complete ✅

Three foundation repos are live:

- [`.github`](https://github.com/alaweimm90/.github) — Reusable workflows, governance, issue templates
- [`standards`](https://github.com/alaweimm90/standards) — SSOT policies, naming, docs rules
- [`core-control-center`](https://github.com/alaweimm90/core-control-center) — Vendor-neutral orchestrator

All three have:
- Working CI (Python test, lint, type-check, coverage; policy enforcement)
- Branch protection enabled
- Complete governance documentation

Next: Days 2–5 (templates & adapters) → Days 6–10 (migration of 55 repos).

Questions? See README in each repo or email ops@alaweimm90.dev
```

---

**You're done with Day 1. Rest, celebrate, and prep for Day 2.**

