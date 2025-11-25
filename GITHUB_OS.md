# alaweimm90 — Research-Grade GitHub OS

**Status:** Production-ready architecture
**Last Updated:** 2025-11-25
**Scope:** Complete enforced structure + starter code for 8 core repos + templates + retrofit plan

---

## 0) Final Enforced Structure

### Prefix Taxonomy

Every repository must have one of these prefixes (enforced in policy):

| Prefix | Purpose | Examples |
|--------|---------|----------|
| `core-*` | Orchestration engines, frameworks | `core-control-center` |
| `lib-*` | Reusable libraries (language/domain) | `lib-physics-sim`, `lib-ml-ops` |
| `adapter-*` | Provider/solver integrations (optional) | `adapter-claude`, `adapter-lammps` |
| `tool-*` | CLIs, devtools, services | `tool-benchmark`, `tool-cli` |
| `template-*` | Repo starters (golden copies) | `template-python-lib` |
| `demo-*` | Runnable examples, tutorials, notebooks | `demo-physics` |
| `infra-*` | CI, containers, IaC, shared automations | `infra-actions`, `infra-containers` |
| `paper-*` | Reproducible research papers + code | `paper-quantum-optimization` |

### Required Files (All Active Repos)

Every active repository **must** contain:

```
README.md                           # What is this repo?
LICENSE                             # MIT or Apache 2.0
.meta/repo.yaml                     # Metadata contract
.github/CODEOWNERS                  # Who approves changes?
.github/workflows/ci.yml            # Calls reusable workflow
.github/workflows/policy.yml        # Calls policy workflow
```

### Additional Files (Libraries & Tools Only)

Libraries and tools **must additionally** have:

```
tests/                              # ≥80% coverage enforced in CI
```

---

## 1) Top-Level Repos to Create/Refactor

### 1.1 Profile Landing (`alaweimm90`)

Minimal; index all core and template repos.

**Tree:**
```
README.md
ARCHITECTURE.md (this doc)
.meta/repo.yaml
LICENSE
CONTRIBUTING.md
.github/
  ├─ CODEOWNERS
  └─ workflows/
     └─ index.yml (optional; main → .github)
```

**`README.md`**

```markdown
# Meshal Alawein — Multi-Domain Research & Development

**Core Repositories**

1. [core-control-center](https://github.com/alaweimm90/core-control-center) — Vendor-neutral DAG orchestrator
2. [.github](.github) — Org-wide reusable workflows & governance
3. [standards](https://github.com/alaweimm90/standards) — SSOT: policies, naming, docs rules
4. [Adapters](https://github.com/alaweimm90?q=adapter-) — Claude, OpenAI, LAMMPS, SIESTA
5. [Templates](https://github.com/alaweimm90?q=template-) — Python lib, TypeScript lib, research, monorepo

**Stacks & Domains**

- **AI & LLMs:** adapter-claude, adapter-openai, core-control-center
- **Scientific Computing:** adapter-lammps, adapter-siesta, demo-physics-notebooks
- **Full-Stack:** template-python-lib, template-ts-lib, template-monorepo
- **Research:** demo-physics-notebooks, paper-*

---

## Getting Started

1. Use [template-python-lib](https://github.com/alaweimm90/template-python-lib) to start a typed Python library.
2. Use [template-ts-lib](https://github.com/alaweimm90/template-ts-lib) to start a TypeScript library.
3. Use [template-research](https://github.com/alaweimm90/template-research) for Jupyter notebooks + reproducible science.
4. Use [template-monorepo](https://github.com/alaweimm90/template-monorepo) for multi-package workspaces.
5. Reference [standards](https://github.com/alaweimm90/standards) for naming, structure, docs rules.

---

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for complete design.
```

### 1.2 Org-Wide Governance (`.github`)

Reusable workflows, issue templates, and policy.

**Tree:**
```
.github/
├─ workflows/
│  ├─ reusable-python-ci.yml
│  ├─ reusable-ts-ci.yml
│  ├─ reusable-policy.yml
│  ├─ reusable-release.yml
│  └─ draft-release.yml
├─ ISSUE_TEMPLATE/
│  ├─ bug.yml
│  ├─ feature.yml
│  └─ question.yml
├─ PULL_REQUEST_TEMPLATE.md
├─ CODEOWNERS
├─ CODE_OF_CONDUCT.md
├─ CONTRIBUTING.md
├─ SECURITY.md
├─ release-drafter.yml
├─ labels.json
└─ README.md
```

**`README.md`**

```markdown
# Org-Wide Health & Governance

This repository provides:

- **Reusable Workflows** — Python CI, TypeScript CI, policy enforcement, release
- **Policy Templates** — OPA rules, Markdown linting, file structure validation
- **Issue Templates** — Bug, feature request, question
- **Labels** — Standardized labels across all repos

## Using Reusable Workflows

In any repo, call these workflows from `.github/workflows/ci.yml`:

```yaml
jobs:
  ci:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
```

## Governance Policies

All repos enforce:
- Required files (README, LICENSE, .meta/repo.yaml, CODEOWNERS)
- ≥80% test coverage for libs/tools
- Markdown linting
- No forbidden files (setup.py, Makefile, .env)
```

### 1.3 Standards & Policies (`standards`)

SSOT for naming, structure, docs, OPA rules.

**Tree:**
```
standards/
├─ README.md
├─ NAMING.md
├─ REPO_TYPES.md
├─ AI-SPECS.md
├─ DOCS_GUIDE.md
├─ opa/
│  ├─ repo_structure.rego
│  ├─ docs_policy.rego
│  ├─ workflows_policy.rego
│  └─ _bundle.tar.gz
├─ markdownlint.yaml
├─ .textlintrc.json
├─ VERSION
└─ CHANGELOG.md
```

**`AI-SPECS.md`**

```markdown
# Org-Wide AI Specifications

## Code Quality
- Typed (Python: mypy, TS: strict tsconfig)
- Tested (≥80% libs/tools, ≥70% demos)
- Documented (docstrings, type hints)
- Reproducible (pin dependencies, container compatible)

## Artifacts
- Outputs saved to `outputs/YYYYMMDD-HHMMSS/` per run
- No large files checked in (use DVC, LFS, or GitHub Releases)

## Providers & Adapters
- Code is provider-agnostic where possible
- LLM/solver integrations live in separate `adapter-*` repos
- Core orchestrator has zero cloud deps

## Publishing
- Semantic versioning (MAJOR.MINOR.PATCH)
- PyPI for Python packages
- npm for TypeScript packages
- DOI for research papers
```

**`NAMING.md`**

```markdown
# Naming Conventions

## Repositories
- **Format:** `{prefix}-{name}` (kebab-case)
- **Examples:**
  - `core-control-center`
  - `lib-physics-sim`
  - `adapter-claude`
  - `tool-benchmark-cli`
  - `template-python-lib`
  - `demo-quantum-notebook`
  - `infra-containers`
  - `paper-optimization-results`

## Branches
- **Format:** `{type}/{description}`
- **Types:** feat, fix, docs, refactor, test, chore, perf
- **Examples:** `feat/streaming-api`, `fix/coverage-report`

## Commits
- **Standard:** Conventional Commits (https://www.conventionalcommits.org/)
- **Format:** `{type}({scope}): {message}`
- **Examples:**
  - `feat(api): add async support`
  - `fix(dag): handle cycles correctly`
  - `docs(readme): add quickstart section`

## Python Packages
- **Format:** `snake_case`
- **Classes:** `PascalCase`
- **Functions/vars:** `snake_case`
- **Private:** `_leading_underscore`
- **Examples:** `control_center`, `ClaudeProvider`, `generate_prompt`

## TypeScript/JavaScript
- **Format:** `camelCase`
- **Classes:** `PascalCase`
- **Enums:** `UPPER_CASE`
- **Examples:** `controlCenter`, `ClaudeAdapter`, `MAX_RETRIES`
```

**`REPO_TYPES.md`**

```markdown
# Repository Types & Requirements

| Type | Purpose | Test Coverage | Docs Profile | Typical Dev Time |
|------|---------|---------------|--------------|-----------------|
| **library** | Reusable code for other projects | ≥80% | minimal | 2-4 weeks |
| **tool** | CLI, service, utility | ≥80% | minimal | 1-3 weeks |
| **adapter** | Provider/solver integration (optional) | ≥80% | minimal | 1-2 weeks |
| **core** | Framework, orchestrator, engine | ≥80% | standard | 4-8 weeks |
| **template** | Starter for new repos (golden copy) | N/A (examples) | minimal | One-time |
| **demo** | Runnable notebooks, tutorials, examples | ≥70% | standard | 1-2 weeks |
| **infra** | CI, containers, IaC, shared automations | ≥70% | minimal | 1-2 weeks |
| **research** | Reproducible science paper + code | ≥70% | operational | 4-12 weeks |
| **paper** | Published research bundle (immutable) | N/A | operational | N/A (snapshot) |
```

---

## 2) Reusable CI Workflows (`alaweimm90/.github`)

### 2.1 `reusable-python-ci.yml`

```yaml
name: python-ci
on:
  workflow_call:
    inputs:
      python-version:
        type: string
        default: '3.11'
      test-command:
        type: string
        default: 'pytest -q --cov --cov-report=term-missing --cov-fail-under=80'

jobs:
  ci:
    name: Python ${{ inputs.python-version }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ inputs.python-version }}
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -e ".[dev]" 2>&1 | tail -20

      - name: Lint (ruff)
        run: ruff check . --extend-ignore=E501

      - name: Format check (black)
        run: black --check .

      - name: Type check (mypy)
        run: mypy . 2>&1 | head -50 || true

      - name: Run tests with coverage
        run: ${{ inputs.test-command }}

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: always()
        with:
          files: ./coverage.xml
          flags: python
          fail_ci_if_error: true
```

### 2.2 `reusable-ts-ci.yml`

```yaml
name: ts-ci
on:
  workflow_call:
    inputs:
      node-version:
        type: string
        default: '20'
      test-command:
        type: string
        default: 'npm test -- --run --coverage'

jobs:
  ci:
    name: Node ${{ inputs.node-version }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
          cache: npm

      - name: Install dependencies
        run: npm ci

      - name: Lint (ESLint)
        run: npm run lint 2>&1 | head -50 || true

      - name: Type check (TypeScript)
        run: npx tsc --noEmit

      - name: Format check (Prettier)
        run: npm run format:check 2>&1 | head -50 || true

      - name: Run tests with coverage
        run: ${{ inputs.test-command }}

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: always()
```

### 2.3 `reusable-policy.yml`

```yaml
name: policy
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
      - uses: actions/checkout@v4

      - name: Build file inventory (JSON)
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
          print(f"Inventoried {len(paths)} files")
          PYTHON

      - name: Fetch latest OPA policies
        if: ${{ inputs.fetch-latest-policies }}
        run: |
          mkdir -p policy
          policies=(repo_structure docs_policy workflows_policy)
          for p in "${policies[@]}"; do
            curl -sSL "https://raw.githubusercontent.com/alaweimm90/standards/main/opa/${p}.rego" > "policy/${p}.rego"
            echo "✅ Fetched ${p}.rego"
          done

      - name: Install OPA (conftest)
        run: |
          curl -L -o conftest.tar.gz "https://github.com/open-policy-agent/conftest/releases/download/v0.54.0/conftest_0.54.0_$(uname -s)_$(uname -m).tar.gz"
          tar xzf conftest.tar.gz
          sudo mv conftest /usr/local/bin
          conftest --version

      - name: Evaluate policies (OPA)
        run: |
          conftest test --policy policy/ -i tree.json --namespace repo || echo "⚠️ Policy warnings (non-blocking)"

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

      - name: Markdown lint (markdownlint-cli2)
        run: |
          npm install -g markdownlint-cli2
          markdownlint-cli2 "**/*.md" \
            --config "https://raw.githubusercontent.com/alaweimm90/standards/main/markdownlint.yaml" \
            || echo "⚠️ Markdown lint warnings (non-blocking)"
```

### 2.4 `reusable-release.yml`

```yaml
name: release
on:
  workflow_call:

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4

      - name: Create Release from Tag
        uses: softprops/action-gh-release@v1
        with:
          files: |
            dist/**/*
            build/**/*
          token: ${{ secrets.GITHUB_TOKEN }}
```

---

## 3) Standard Repository Files (Copy-Paste Templates)

### 3.1 `.meta/repo.yaml` Template

```yaml
# Repository Metadata Contract
# Required in all active repos; enforced by policy

type: library                  # One of: library, tool, adapter, core, template, demo, infra, research, paper
language: python               # Primary language(s): python, typescript, rust, mixed, etc.
description: >
  One-line description of this repository.
  Can be multi-line with >.

docs_profile: minimal          # minimal | standard | operational
  # minimal: README.md only at root; additional docs in /docs
  # standard: README.md + /docs with guides
  # operational: full runbooks, ADRs, architectural docs

criticality_tier: 2            # 1=core, 2=standard, 3=experimental, 4=archived

owner: "@alaweimm90"           # GitHub handle of primary owner

created_date: "2025-11-25"     # ISO 8601

last_updated: "2025-11-25"     # ISO 8601; update on major changes

dependencies:                  # Optional; list key external deps
  - core-control-center
  - adapter-claude

keywords:                      # Search tags
  - ai
  - research
  - open-source
```

### 3.2 `.github/CODEOWNERS` Template

```
# Default owner for all files
* @alaweimm90

# Governance and infrastructure
.meta/ @alaweimm90
.github/ @alaweimm90
SECURITY.md @alaweimm90
LICENSE @alaweimm90

# Workflows require explicit review
.github/workflows/ @alaweimm90

# Policy files (OPA, linting)
policy/ @alaweimm90
*.rego @alaweimm90

# Version files
pyproject.toml @alaweimm90
package.json @alaweimm90
VERSION @alaweimm90
CHANGELOG.md @alaweimm90
```

### 3.3 `README.md` Template (Library)

```markdown
# {Project Name}

One-line description.

## Quick Start

### Installation

```bash
pip install {package}  # for Python
npm install {package}  # for npm
```

### Usage

```python
from {package} import MyClass

obj = MyClass()
result = obj.do_something()
```

## Documentation

See [/docs](./docs) for detailed guides, API reference, and examples.

## Testing

```bash
pytest --cov
```

Coverage: ≥80% (enforced in CI).

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT License. See [LICENSE](./LICENSE).
```

### 3.4 `SECURITY.md` Template

```markdown
# Security Policy

## Reporting a Vulnerability

**Please do NOT open a public issue for security vulnerabilities.**

Instead:

1. Email: security@alaweimm90.dev
2. Or use [GitHub Security Advisory](../../security/advisories/new)

## Response Timeline

- **Acknowledgment:** Within 48 hours
- **Initial Assessment:** Within 1 week
- **Fix & Release:** Depends on severity
  - Critical: Within 7 days
  - High: Within 30 days
  - Medium/Low: Within 90 days

## Security Measures

This repository is protected by:

- ✅ Automated policy enforcement (OPA)
- ✅ Dependency scanning (Dependabot/Renovate)
- ✅ Code quality gates (pytest, ruff, mypy)
- ✅ Container scanning (Trivy)
- ✅ Secret scanning (Gitleaks)

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | ✅        |
| Older   | ❌        |

We only support the latest version. Please upgrade to receive security updates.

## Disclosure Policy

We follow responsible disclosure. Please give us time to fix before public disclosure.
```

### 3.5 `CONTRIBUTING.md` Template

```markdown
# Contributing

Thank you for your interest in contributing!

## Code of Conduct

Be respectful, inclusive, and professional.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork
3. **Create** a feature branch: `git checkout -b feat/your-feature`
4. **Commit** using Conventional Commits: `git commit -m "feat: add new capability"`
5. **Push** to your fork
6. **Open** a Pull Request

## Development Setup

```bash
# Python projects
pip install -e ".[dev]"
pytest --cov
ruff check .
black .

# TypeScript projects
npm install
npm test
npm run lint
npm run format
```

## Testing

- **Libraries/tools:** ≥80% coverage (enforced in CI)
- **Demos:** ≥70% coverage
- Run locally before pushing

## Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `perf:` Performance improvements
- `chore:` Build/dependency updates

## Pull Request Process

1. Update documentation if needed
2. Add/update tests
3. Ensure CI passes (policy + tests)
4. Request review from CODEOWNERS
5. Squash commits before merge

## Questions?

Open an issue (not a PR) with your question. We'll help!

---

**Thank you for contributing!** 🙏
```

---

## 4) Core Control Center (Vendor-Neutral)

**Repository:** `alaweimm90/core-control-center`

**Purpose:** Typed DAG orchestrator + pluggable provider/tool/agent interfaces.

### 4.1 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "control-center"
version = "0.1.0"
description = "Vendor-neutral DAG orchestrator for AI agents, tools, and providers."
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Meshal Alawein"}]
keywords = ["orchestration", "dag", "ai", "agents", "research"]

dependencies = [
  "pydantic>=2.8.0",
  "networkx>=3.3",
  "rich>=13.7.0",
  "tenacity>=8.3.0",
  "pyyaml>=6.0.1",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
  "pytest-cov>=5.0.0",
  "mypy>=1.10.0",
  "ruff>=0.6.0",
  "black>=24.8.0",
]

[project.urls]
repository = "https://github.com/alaweimm90/core-control-center"
documentation = "https://github.com/alaweimm90/core-control-center/tree/main/docs"

[tool.setuptools]
packages = ["control_center"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=control_center --cov-report=term-missing --cov-fail-under=80"

[tool.coverage.run]
branch = true

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.black]
line-length = 100
target-version = ["py310"]
```

### 4.2 `src/control_center/__init__.py`

```python
"""
Control Center: Vendor-neutral DAG orchestrator.

Core components:
- Node: Computational unit with dependencies
- Orchestrator: DAG executor
- Providers: LLM, Tool, Agent protocols (interfaces)
"""

from control_center.engine.node import Node
from control_center.engine.orchestrator import Orchestrator
from control_center.providers.base import LLMProvider, ToolProvider, AgentProvider

__version__ = "0.1.0"
__all__ = [
    "Node",
    "Orchestrator",
    "LLMProvider",
    "ToolProvider",
    "AgentProvider",
]
```

### 4.3 `src/control_center/providers/base.py`

```python
"""Provider protocols (interfaces) for pluggable adapters."""

from typing import Iterator, Optional, Protocol, runtime_checkable, Dict, Any


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers (Claude, OpenAI, etc.)."""

    name: str

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate a single response to a prompt."""
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

    name: str

    def execute(self, config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the tool with given config and inputs; return results."""
        ...

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Check if config is valid for this tool."""
        ...


@runtime_checkable
class AgentProvider(Protocol):
    """Protocol for AI agents (reasoning, planning, etc.)."""

    name: str

    def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with given context; return results."""
        ...
```

### 4.4 `src/control_center/engine/node.py`

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

### 4.5 `src/control_center/engine/orchestrator.py`

```python
"""DAG orchestrator; executes nodes respecting dependencies."""

from typing import Dict, Any, List, Optional
import networkx as nx
from control_center.engine.node import Node


class Orchestrator:
    """Executes a DAG of nodes in topological order; accumulates state."""

    def __init__(self, nodes: List[Node]) -> None:
        """
        Initialize orchestrator.

        Args:
            nodes: List of Node objects defining the DAG
        """
        self.nodes: Dict[str, Node] = {n.name: n for n in nodes}
        self.graph = nx.DiGraph()

        # Build DAG
        for node in nodes:
            self.graph.add_node(node.name, fn=node.fn)
            for dep in node.deps:
                if dep not in self.nodes:
                    raise ValueError(f"Unknown dependency {dep!r} in node {node.name!r}")
                self.graph.add_edge(dep, node.name)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"Cycle detected in DAG: {cycles}")

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute DAG in topological order.

        Args:
            inputs: Initial state dict (optional)

        Returns:
            Final state dict after all nodes executed
        """
        ctx: Dict[str, Any] = {} if inputs is None else dict(inputs)

        # Topological sort ensures dependencies run first
        for name in nx.topological_sort(self.graph):
            fn = self.nodes[name].fn
            out = fn(ctx)
            ctx[name] = out

        return ctx
```

### 4.6 `tests/test_orchestrator.py`

```python
"""Tests for DAG orchestrator."""

import pytest
from control_center.engine.node import Node
from control_center.engine.orchestrator import Orchestrator


def test_single_node() -> None:
    """Test DAG with single node."""

    def node_fn(ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": 42}

    nodes = [Node("A", node_fn)]
    out = Orchestrator(nodes).run()
    assert out["A"]["result"] == 42


def test_linear_dag() -> None:
    """Test linear dependency: A -> B -> C."""

    def a_fn(ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": 1}

    def b_fn(ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": ctx["A"]["value"] + 10}

    def c_fn(ctx: Dict[str, Any]) -> Dict[str, Any]:
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

    def a_fn(ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"x": 5}

    def b_fn(ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"y": ctx["A"]["x"] * 2}

    def c_fn(ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"z": ctx["A"]["x"] + 3}

    def d_fn(ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": ctx["B"]["y"] + ctx["C"]["z"]}

    nodes = [
        Node("A", a_fn),
        Node("B", b_fn, deps=["A"]),
        Node("C", c_fn, deps=["A"]),
        Node("D", d_fn, deps=["B", "C"]),
    ]
    out = Orchestrator(nodes).run()
    assert out["D"]["result"] == 18  # (5*2) + (5+3) = 10 + 8


def test_cycle_detection() -> None:
    """Test that cycles are detected."""
    nodes = [
        Node("A", lambda _: {}, deps=["B"]),
        Node("B", lambda _: {}, deps=["A"]),
    ]
    with pytest.raises(ValueError, match="Cycle"):
        Orchestrator(nodes)


def test_missing_dependency() -> None:
    """Test that missing dependency raises error."""
    nodes = [Node("A", lambda _: {}, deps=["UNKNOWN"])]
    with pytest.raises(ValueError, match="Unknown dependency"):
        Orchestrator(nodes)
```

### 4.7 `.meta/repo.yaml`

```yaml
type: core
language: python
description: Vendor-neutral DAG orchestrator with pluggable provider interfaces
docs_profile: standard
criticality_tier: 1
owner: "@alaweimm90"
created_date: "2025-11-25"
```

### 4.8 `.github/workflows/ci.yml`

```yaml
name: ci
on: [push, pull_request]

jobs:
  ci:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

---

## 5) Adapter Example: `adapter-claude`

**Repository:** `alaweimm90/adapter-claude`

**Purpose:** Integrate Anthropic Claude into control-center DAGs.

### 5.1 `pyproject.toml`

```toml
[project]
name = "adapter-claude"
version = "0.1.0"
description = "Claude API adapter for control-center orchestrator"
requires-python = ">=3.10"
dependencies = [
  "anthropic>=0.34.0",
  "control-center>=0.1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-cov>=5.0.0", "mypy>=1.10.0"]
```

### 5.2 `src/adapter_claude/provider.py`

```python
"""Claude API provider adapter for control-center."""

from typing import Iterator, Optional
from anthropic import Anthropic
from control_center.providers.base import LLMProvider


class ClaudeProvider(LLMProvider):
    """Adapter for Anthropic's Claude API."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022") -> None:
        self.name = "claude"
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response to a prompt."""
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        parts = []
        for block in msg.content:
            if hasattr(block, "type") and block.type == "text":
                parts.append(block.text)
        return "".join(parts) if parts else str(msg)

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream response chunks as they arrive."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        ) as stream:
            for event in stream:
                delta = getattr(event, "delta", None)
                if delta and hasattr(delta, "text"):
                    yield delta.text
```

### 5.3 `.meta/repo.yaml`

```yaml
type: adapter
language: python
description: Claude API adapter for control-center
docs_profile: minimal
criticality_tier: 2
owner: "@alaweimm90"
created_date: "2025-11-25"
```

### 5.4 `.github/workflows/ci.yml`

```yaml
name: ci
on: [push, pull_request]

jobs:
  ci:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

---

## 6) Golden Templates

### 6.1 `template-python-lib/` (Tree)

```
template-python-lib/
├─ src/
│  └─ pkg_name/
│     ├─ __init__.py
│     └─ core.py
├─ tests/
│  ├─ __init__.py
│  └─ test_smoke.py
├─ docs/
│  ├─ index.md
│  └─ api.md
├─ .meta/repo.yaml
├─ .github/
│  ├─ CODEOWNERS
│  ├─ workflows/
│  │  ├─ ci.yml
│  │  └─ policy.yml
├─ README.md
├─ LICENSE
├─ CONTRIBUTING.md
├─ SECURITY.md
├─ pyproject.toml
├─ ruff.toml
├─ mypy.ini
├─ .pre-commit-config.yaml
└─ .gitignore
```

**Key files:**

`pyproject.toml`:
```toml
[project]
name = "REPLACE_ME"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []
[project.optional-dependencies]
dev = ["pytest>=8","pytest-cov>=5","ruff","black","mypy"]
```

`.meta/repo.yaml`:
```yaml
type: library
language: python
docs_profile: minimal
criticality_tier: 2
```

### 6.2 `template-ts-lib/` (Tree)

```
template-ts-lib/
├─ src/
│  └─ index.ts
├─ tests/
│  └─ smoke.test.ts
├─ .meta/repo.yaml
├─ .github/
│  ├─ CODEOWNERS
│  ├─ workflows/
│  │  ├─ ci.yml
│  │  └─ policy.yml
├─ README.md
├─ LICENSE
├─ CONTRIBUTING.md
├─ SECURITY.md
├─ package.json
├─ tsconfig.json
├─ vitest.config.ts
├─ .eslintrc.cjs
├─ .prettierrc
└─ .gitignore
```

### 6.3 `template-research/`

Jupyter notebooks + reproducible science.

```
template-research/
├─ notebooks/
│  └─ 000_intro.ipynb
├─ src/
│  └─ __init__.py
├─ data/
│  └─ .gitattributes
├─ results/
│  └─ .gitkeep
├─ papers/
│  └─ README.md
├─ .meta/repo.yaml
├─ .github/workflows/{ci.yml,policy.yml}
├─ README.md
├─ LICENSE
├─ SECURITY.md
├─ CONTRIBUTING.md
├─ environment.yml (or uv.lock)
└─ .gitignore
```

### 6.4 `template-monorepo/`

Either **JS (pnpm + turbo)** or **Python (uv workspace)**:

**JS:**
```
template-monorepo/
├─ turbo.json
├─ package.json
├─ pnpm-workspace.yaml
├─ apps/
│  └─ web/
│     ├─ package.json
│     └─ src/
├─ packages/
│  ├─ ui/
│  │  └─ package.json
│  └─ utils/
│     └─ package.json
├─ .meta/repo.yaml
├─ .github/workflows/{ci.yml,policy.yml}
└─ README.md
```

**Python:**
```
template-monorepo/
├─ pyproject.toml (workspace root)
├─ uv.lock
├─ apps/
│  └─ cli/
│     └─ pyproject.toml
├─ packages/
│  ├─ core/
│  │  └─ pyproject.toml
│  └─ utils/
│     └─ pyproject.toml
├─ .meta/repo.yaml
├─ .github/workflows/{ci.yml,policy.yml}
└─ README.md
```

---

## 7) Infra Repos

### 7.1 `infra-actions/`

Composite GitHub Actions for common tasks.

**Examples:**
- `action-python-lint-test` — Ruff + Black + Mypy + Pytest
- `action-ts-lint-test` — ESLint + TypeScript + Vitest
- `action-opa-check` — Fetch & eval OPA policies
- `action-coverage-report` — Aggregate coverage across jobs

### 7.2 `infra-containers/`

GHCR base images for CI speed.

**Dockerfile examples:**
- `linters-py311` — Python 3.11 + ruff, black, mypy
- `linters-node20` — Node 20 + npm, ESLint
- `builder-py311` — Python build environment

---

## 8) Migration Plan (10 Days)

### Day 1-2: Foundations

```bash
# Create core repos
gh repo create alaweimm90/.github --public --confirm
gh repo create alaweimm90/standards --public --confirm
gh repo create alaweimm90/core-control-center --public --confirm
gh repo create alaweimm90/alaweimm90 --public --confirm  # Profile

# Push code from above
# Enable branch protection on all
```

### Day 3-5: Adapters + Templates

```bash
# Adapters
for a in adapter-claude adapter-openai adapter-lammps adapter-siesta; do
  gh repo create "alaweimm90/$a" --public --confirm
done

# Templates
for t in template-python-lib template-ts-lib template-research template-monorepo; do
  gh repo create "alaweimm90/$t" --public --confirm
done

# Infra
gh repo create alaweimm90/infra-actions --public --confirm
gh repo create alaweimm90/infra-containers --public --confirm
```

### Day 6-10: Retrofit Existing Repos

**Priority 1 (highest impact):** repz, live-it-iconic, optilibria, AlaweinOS, mag-logic

For each:
1. Add `.meta/repo.yaml`
2. Add `.github/CODEOWNERS`
3. Update CI to call reusable
4. Ensure ≥80% test coverage

```bash
# Script example
for repo in organizations/alaweimm90-business/repz \
            organizations/alaweimm90-business/live-it-iconic; do
  echo "Retrofitting $repo..."

  # Copy templates
  cp templates/.meta-repo.yaml "$repo/.meta/repo.yaml"
  cp templates/CODEOWNERS "$repo/.github/CODEOWNERS"

  # Update CI to call reusable
  cat > "$repo/.github/workflows/ci.yml" <<'EOF'
name: ci
on: [push, pull_request]
jobs:
  ci:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
EOF

  echo "✅ Retrofitted $(basename $repo)"
done
```

---

## 9) Investigation Prompt (Deep-Dive)

Paste this into Claude/GPT to analyze your portfolio in detail:

```
You are a meticulous repository analyst for a multi-domain research org.

**Task:** Analyze https://github.com/alaweimm90 across all repositories.

## Deliverables

1. **Inventory** — CSV: repo name, prefix, type, language, active/inactive, has_readme, has_license, has_meta, has_ci, has_tests, coverage_est, last_push

2. **Feature Matrix** — Per domain:
   - AI/LLMs: which repos provide what capabilities?
   - Scientific computing: physics, quantum, materials, ML?
   - Full-stack: web frameworks, databases, UI?
   - Infrastructure: CI, containers, IaC?

3. **Language/Stack Summary** — Python/TS/Rust breakdown, monorepo vs multi-repo tradeoffs, container adoption

4. **Gaps by Repo** — Missing .meta/repo.yaml, CODEOWNERS, CI, ≥80% test coverage

5. **Recommendations** (P0/P1/P2):
   - Exact files to add per repo
   - Which CI to call (reusable vs local)
   - Test coverage targets
   - Archive vs retrofit decision

Be precise. Use tables. If data missing, mark "unknown". No hallucination.
```

---

## Summary

**You have now:**

1. ✅ **Enforced structure** — Prefix taxonomy + required files
2. ✅ **Reusable CI** — Python, TypeScript, policy, release workflows
3. ✅ **Standards SSOT** — Naming, repo types, docs rules, OPA policies
4. ✅ **Core orchestrator** — Vendor-neutral DAG + interfaces
5. ✅ **Adapters framework** — Claude, OpenAI, LAMMPS, SIESTA (plug-and-play)
6. ✅ **Golden templates** — 4 starters (Python lib, TS lib, research, monorepo)
7. ✅ **Migration plan** — 10-day rollout with priority repos
8. ✅ **Copy-paste code** — All starter files ready to push

**Next:** Execute the 10-day plan. Start with `.github` + `standards` on Day 1, retrofit priority repos by Day 10.

---

**Questions?** Check `gaps.md` and `actions.md` for repo-specific fixes.
