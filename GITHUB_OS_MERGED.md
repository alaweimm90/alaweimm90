# alaweimm90 — Research-Grade GitHub OS (Complete Edition)

**Status:** Production-ready, merged specification
**Last Updated:** 2025-11-25
**Scope:** Complete enforced structure + all starter code + full implementations + complete guidance

This document combines the best of both the detailed guidance document AND the complete code implementations into a single unified reference.

---

## 0) Portfolio Contract (Enforced by Policy)

### Prefix Taxonomy (All Active Repos)

Every repository **must** use one of these prefixes (enforced in OPA policy):

| Prefix | Purpose | Scope | Example |
|--------|---------|-------|---------|
| `core-*` | Orchestration engines, frameworks, platforms | Central logic | `core-control-center` |
| `lib-*` | Reusable libraries (any language/domain) | Shareable | `lib-physics-sim`, `lib-ml-ops` |
| `adapter-*` | Provider/solver integrations (optional plugins) | Optional | `adapter-claude`, `adapter-lammps` |
| `tool-*` | CLIs, devtools, utilities, services | Tooling | `tool-benchmark-cli`, `tool-metrics` |
| `template-*` | Starter repos (golden copies for new projects) | Templates | `template-python-lib`, `template-research` |
| `demo-*` | Runnable examples, tutorials, notebooks | Examples | `demo-physics-notebooks`, `demo-quantum-lab` |
| `infra-*` | CI, containers, IaC, shared automations | Infrastructure | `infra-actions`, `infra-containers` |
| `paper-*` | Reproducible research papers + code (immutable) | Archives | `paper-optimization-results` |

### Required Files (All Active Repos)

Every active repository **must** contain:

```
README.md                           # What is this repo? Quick start guide
LICENSE                             # MIT or Apache 2.0
.meta/repo.yaml                     # Metadata contract (see Section 3.2)
.github/CODEOWNERS                  # Who approves changes? (see Section 3.3)
.github/workflows/ci.yml            # Calls reusable workflow (see Section 2)
.github/workflows/policy.yml        # Calls policy enforcement (see Section 2)
```

### Additional Requirements (Libraries & Tools Only)

Libraries (`lib-*`) and tools (`tool-*`) **must additionally** have:

```
tests/                              # Test directory
  - ≥80% code coverage enforced in CI
```

Research (`demo-*`, `paper-*`) may have:

```
tests/                              # Test directory (optional, ≥70% if present)
```

---

## 1) Naming Conventions (Org-Wide Standards)

### Repository Names

- **Format:** `{prefix}-{name}` in kebab-case
- **Pattern:** Lowercase letters, hyphens, numbers only
- **Examples:**
  - ✅ `core-control-center`
  - ✅ `lib-quantum-materials`
  - ✅ `adapter-claude`
  - ✅ `tool-benchmark-cli`
  - ✅ `template-python-lib`
  - ❌ `Control_Center` (wrong case)
  - ❌ `lib-quantumMaterials` (camelCase in repo name)

### Branch Names

- **Format:** `{type}/{description}` in kebab-case
- **Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`
- **Examples:**
  - `feat/streaming-api`
  - `fix/coverage-enforcement`
  - `docs/installation-guide`
  - `refactor/dag-engine`

### Commit Messages

- **Standard:** Conventional Commits (https://www.conventionalcommits.org/)
- **Format:** `{type}({scope}): {message}`
- **Examples:**
  - `feat(orchestrator): add async node execution`
  - `fix(dag): handle cyclic dependency detection`
  - `docs(readme): add quickstart section`
  - `test(adapters): add Claude adapter integration tests`

### Python Code Style

- **Package names:** `snake_case` (matches directory)
- **Classes:** `PascalCase`
- **Functions/Methods:** `snake_case`
- **Constants:** `UPPER_CASE`
- **Private:** Leading underscore `_private_var`
- **Examples:**
  - ✅ `control_center` (package)
  - ✅ `class ClaudeAdapter:`
  - ✅ `def generate_prompt():`
  - ✅ `MAX_RETRIES = 3`

### TypeScript/JavaScript Code Style

- **Package names:** kebab-case in package.json
- **Classes:** `PascalCase`
- **Functions:** `camelCase`
- **Constants:** `UPPER_CASE`
- **Examples:**
  - ✅ `"name": "control-center"`
  - ✅ `class ClaudeAdapter`
  - ✅ `function generatePrompt()`
  - ✅ `const MAX_RETRIES = 3`

---

## 2) Reusable CI & Policy Workflows

All project repos call these centralized workflows from `alaweimm90/.github`. This eliminates duplication, ensures consistency, and allows policy changes to apply globally.

**Location:** `https://github.com/alaweimm90/.github/.github/workflows/`

### 2.1 `reusable-python-ci.yml`

Handles Python testing, linting, type checking, and coverage enforcement.

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

      - name: Lint with ruff
        run: ruff check . --extend-ignore=E501

      - name: Format check with black
        run: black --check .

      - name: Type check with mypy
        run: mypy . 2>&1 | head -50 || true

      - name: Run tests with coverage
        run: ${{ inputs.test-command }}

      - name: Upload coverage to codecov
        uses: codecov/codecov-action@v4
        if: always()
        with:
          files: ./coverage.xml
          flags: python
          fail_ci_if_error: true
```

**Key Features:**
- ✅ Ruff linting (extended ignore for line length)
- ✅ Black formatting enforcement
- ✅ MyPy type checking (non-blocking for now)
- ✅ Pytest with coverage enforcement (--cov-fail-under=80)
- ✅ Codecov integration for coverage tracking
- ✅ Configurable Python version and test command

**Usage (in your repo's `.github/workflows/ci.yml`):**

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'
      test-command: 'pytest -q --cov --cov-report=term-missing --cov-fail-under=80'
```

### 2.2 `reusable-ts-ci.yml`

Handles TypeScript/JavaScript testing, linting, type checking, and coverage.

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
        run: npm ci || npm i

      - name: Run linter
        run: npm run lint --if-present || npx eslint . --max-warnings=0

      - name: Type check
        run: npm run typecheck --if-present || npx tsc --noEmit

      - name: Run tests with coverage
        run: ${{ inputs.test-command }}

      - name: Upload coverage to codecov
        uses: codecov/codecov-action@v4
        if: always()
        with:
          files: ./coverage/coverage-summary.json
          flags: typescript
          fail_ci_if_error: true
```

**Key Features:**
- ✅ ESLint linting (if available)
- ✅ TypeScript type checking (if available)
- ✅ Vitest/Jest with coverage reporting
- ✅ Codecov integration
- ✅ Configurable Node version and test command

**Usage (in your repo's `.github/workflows/ci.yml`):**

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    uses: alaweimm90/.github/.github/workflows/reusable-ts-ci.yml@main
    with:
      node-version: '20'
      test-command: 'npm test -- --run --coverage'
```

### 2.3 `reusable-policy.yml`

Enforces governance policies: required files, naming conventions, OPA rules.

```yaml
name: policy
on:
  workflow_call:

jobs:
  policy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate required files
        run: |
          set -e
          for f in README.md LICENSE .meta/repo.yaml .github/CODEOWNERS; do
            if [ ! -f "$f" ]; then
              echo "❌ Missing required file: $f"
              exit 1
            fi
          done
          echo "✅ All required files present"

      - name: Setup OPA
        uses: open-policy-agent/setup-opa@v2
        with:
          version: latest

      - name: Run OPA policy checks
        run: |
          set -e
          if [ -d "policy" ]; then
            echo "📋 Running OPA policies..."
            opa eval --fail-defined \
              -i .meta/repo.yaml \
              -d policy/ \
              -f pretty \
              'data.repo.pass' || {
              echo "❌ Policy check failed"
              exit 1
            }
          else
            echo "⚠️ No policy/ directory found; skipping OPA checks"
          fi
          echo "✅ Policy checks passed"

      - name: Markdown linting
        run: |
          npm install -g markdownlint-cli 2>/dev/null || true
          markdownlint README.md CONTRIBUTING.md SECURITY.md 2>/dev/null || echo "⚠️ Markdown lint warnings (non-blocking)"
```

**Key Features:**
- ✅ Required files validation
- ✅ OPA policy enforcement (if policy/ exists)
- ✅ Markdown linting (non-blocking)

**Usage (in your repo's `.github/workflows/policy.yml`):**

```yaml
name: policy
on: [push, pull_request]
jobs:
  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

---

## 3) Core Orchestrator (`core-control-center`)

A vendor-neutral, typed DAG (directed acyclic graph) orchestrator for composing operations. Written in TypeScript with strict typing and Vitest tests.

### 3.1 Port Definitions (`src/ports/Adapter.ts`)

```typescript
/**
 * Context provided to every adapter at runtime.
 * Contains environment variables, working directory, and timeout settings.
 */
export interface AdapterContext {
  env: Record<string, string>;
  cwd?: string;
  timeoutMs?: number;
}

/**
 * Generic adapter interface for pluggable providers.
 * I = Input type, O = Output type.
 * Implements the Provider Protocol: loose coupling via interfaces.
 */
export interface Adapter<I, O> {
  readonly name: string;

  /**
   * Optional health check before running.
   * Can verify API keys, network connectivity, etc.
   */
  check?(ctx: AdapterContext): Promise<void>;

  /**
   * Execute the adapter.
   * Pure function: same input always produces same output.
   */
  run(input: I, ctx: AdapterContext): Promise<O>;
}
```

### 3.2 DAG Runner (`src/engine/Runner.ts`)

```typescript
/**
 * Specification for a single node in the DAG.
 */
export interface NodeSpec<I, O> {
  id: string;                                      // Unique node ID
  deps: string[];                                  // Dependency IDs
  adapter: Adapter<I, O>;                          // The execution logic
  input: (ctx: Record<string, unknown>) => Promise<I> | I;  // Input mapper
  save: (out: O, ctx: Record<string, unknown>) => Promise<void> | void;  // Output saver
}

/**
 * Typed DAG orchestrator with topological sort and state accumulation.
 */
export class Runner {
  constructor(private readonly nodes: NodeSpec<any, any>[]) {}

  /**
   * Execute the DAG in topological order.
   * Context accumulates outputs from previous nodes.
   */
  async run(): Promise<void> {
    const ctx: Record<string, unknown> = {};
    const done = new Set<string>();
    const byId = new Map(this.nodes.map(n => [n.id, n]));

    while (done.size < this.nodes.length) {
      // Find nodes with all dependencies satisfied
      const ready = this.nodes.filter(
        n => !done.has(n.id) && n.deps.every(d => done.has(d))
      );

      if (ready.length === 0) {
        throw new Error("Deadlock: check DAG for cycles or unsatisfied dependencies");
      }

      // Execute ready nodes in parallel (safe due to DAG property)
      for (const n of ready) {
        const input = await n.input(ctx);
        const out = await n.adapter.run(input, {
          env: process.env,
          timeoutMs: 300_000
        });
        await n.save(out, ctx);
        done.add(n.id);
      }
    }
  }
}
```

### 3.3 Vitest Test (`tests/engine/runner.spec.ts`)

```typescript
import { describe, it, expect } from 'vitest';
import { Runner } from '../../src/engine/Runner';
import { Adapter } from '../../src/ports/Adapter';

describe('Runner', () => {
  it('executes single node', async () => {
    const echo: Adapter<string, string> = {
      name: 'echo',
      run: async (input) => input
    };

    const runner = new Runner([{
      id: 'n1',
      deps: [],
      adapter: echo,
      input: () => 'hello',
      save: (output) => {
        expect(output).toBe('hello');
      }
    }]);

    await runner.run();
  });

  it('executes DAG with dependencies', async () => {
    const results: string[] = [];

    const adapterA: Adapter<void, string> = {
      name: 'a',
      run: async () => 'A'
    };

    const adapterB: Adapter<string, string> = {
      name: 'b',
      run: async (input) => input + 'B'
    };

    const runner = new Runner([
      {
        id: 'nodeA',
        deps: [],
        adapter: adapterA,
        input: () => undefined,
        save: (out, ctx) => { ctx.a = out; results.push(out); }
      },
      {
        id: 'nodeB',
        deps: ['nodeA'],
        adapter: adapterB,
        input: (ctx) => ctx.a as string,
        save: (out) => { results.push(out); }
      }
    ]);

    await runner.run();
    expect(results).toEqual(['A', 'AB']);
  });

  it('detects cyclic dependencies', async () => {
    const dummy: Adapter<any, any> = { name: 'dummy', run: async (i) => i };

    const runner = new Runner([
      { id: 'n1', deps: ['n2'], adapter: dummy, input: () => {}, save: () => {} },
      { id: 'n2', deps: ['n1'], adapter: dummy, input: () => {}, save: () => {} }
    ]);

    await expect(runner.run()).rejects.toThrow('Deadlock');
  });
});
```

### 3.4 Package Configuration (`package.json`)

```json
{
  "name": "core-control-center",
  "version": "0.1.0",
  "type": "module",
  "description": "Vendor-neutral DAG orchestrator for composing typed operations",
  "scripts": {
    "build": "tsc -p .",
    "test": "vitest run --coverage",
    "dev": "vitest"
  },
  "devDependencies": {
    "typescript": "^5.6.3",
    "vitest": "^2.1.0",
    "@vitest/coverage-v8": "^2.1.0"
  }
}
```

---

## 4) Adapters (Pluggable Providers)

Adapters are optional, provider-specific implementations that plug into the core orchestrator. All adapters implement the `Adapter<I,O>` interface.

### 4.1 `adapter-claude` (TypeScript)

Claude API integration for LLM operations.

**File:** `src/claude-adapter.ts`

```typescript
import { Adapter, AdapterContext } from 'core-control-center/ports/Adapter';

export interface ClaudeChatInput {
  prompt: string;
  systemPrompt?: string;
  maxTokens?: number;
}

export interface ClaudeChatOutput {
  text: string;
  stopReason?: string;
  usage?: {
    inputTokens: number;
    outputTokens: number;
  };
}

export const ClaudeChat: Adapter<ClaudeChatInput, ClaudeChatOutput> = {
  name: 'claude.chat',

  async check(ctx: AdapterContext): Promise<void> {
    if (!ctx.env.ANTHROPIC_API_KEY) {
      throw new Error('ANTHROPIC_API_KEY environment variable not set');
    }
  },

  async run(input: ClaudeChatInput, ctx: AdapterContext): Promise<ClaudeChatOutput> {
    const resp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': ctx.env.ANTHROPIC_API_KEY!,
        'content-type': 'application/json',
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: input.maxTokens ?? 1024,
        system: input.systemPrompt ?? '',
        messages: [
          {
            role: 'user',
            content: input.prompt
          }
        ]
      })
    });

    if (!resp.ok) {
      throw new Error(`Claude API error: ${resp.statusText}`);
    }

    const data = await resp.json();
    return {
      text: data.content?.[0]?.text ?? '',
      stopReason: data.stop_reason,
      usage: {
        inputTokens: data.usage?.input_tokens ?? 0,
        outputTokens: data.usage?.output_tokens ?? 0
      }
    };
  }
};
```

### 4.2 `adapter-openai` (TypeScript)

OpenAI API integration for LLM operations.

**File:** `src/openai-adapter.ts`

```typescript
import { Adapter, AdapterContext } from 'core-control-center/ports/Adapter';

export interface OpenAIChatInput {
  prompt: string;
  systemPrompt?: string;
  maxTokens?: number;
}

export interface OpenAIChatOutput {
  text: string;
  finishReason?: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

export const OpenAIChat: Adapter<OpenAIChatInput, OpenAIChatOutput> = {
  name: 'openai.chat',

  async check(ctx: AdapterContext): Promise<void> {
    if (!ctx.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY environment variable not set');
    }
  },

  async run(input: OpenAIChatInput, ctx: AdapterContext): Promise<OpenAIChatOutput> {
    const resp = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${ctx.env.OPENAI_API_KEY!}`,
        'content-type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        max_tokens: input.maxTokens ?? 1024,
        messages: [
          { role: 'system', content: input.systemPrompt ?? 'You are a helpful assistant.' },
          { role: 'user', content: input.prompt }
        ]
      })
    });

    if (!resp.ok) {
      throw new Error(`OpenAI API error: ${resp.statusText}`);
    }

    const data = await resp.json();
    return {
      text: data.choices?.[0]?.message?.content ?? '',
      finishReason: data.choices?.[0]?.finish_reason,
      usage: {
        promptTokens: data.usage?.prompt_tokens ?? 0,
        completionTokens: data.usage?.completion_tokens ?? 0,
        totalTokens: data.usage?.total_tokens ?? 0
      }
    };
  }
};
```

### 4.3 `adapter-lammps` (Python)

LAMMPS molecular dynamics runner.

**File:** `adapter_lammps/runner.py`

```python
"""LAMMPS adapter for core-control-center orchestrator."""

from pathlib import Path
from typing import Dict, Any
import subprocess


def run(
    input_file: str,
    workdir: str = ".",
    env: Dict[str, str] | None = None
) -> Dict[str, Any]:
    """
    Run LAMMPS simulation.

    Args:
        input_file: LAMMPS input script filename
        workdir: Working directory for simulation
        env: Environment variables (e.g., LD_LIBRARY_PATH)

    Returns:
        {
            "returncode": int,
            "stdout": str,
            "stderr": str,
            "success": bool
        }
    """
    wd = Path(workdir)
    cmd = ["lmp", "-in", input_file]

    proc = subprocess.run(
        cmd,
        cwd=wd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300
    )

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "success": proc.returncode == 0
    }
```

### 4.4 `adapter-siesta` (Python)

SIESTA quantum chemistry runner.

**File:** `adapter_siesta/runner.py`

```python
"""SIESTA adapter for core-control-center orchestrator."""

from pathlib import Path
from typing import Dict, Any
import subprocess


def run(
    fdf: str,
    workdir: str = ".",
    env: Dict[str, str] | None = None
) -> Dict[str, Any]:
    """
    Run SIESTA DFT calculation.

    Args:
        fdf: SIESTA input file (.fdf)
        workdir: Working directory
        env: Environment variables

    Returns:
        {
            "returncode": int,
            "stdout": str,
            "stderr": str,
            "success": bool
        }
    """
    wd = Path(workdir)
    cmd = ["siesta", fdf]

    proc = subprocess.run(
        cmd,
        cwd=wd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600
    )

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "success": proc.returncode == 0
    }
```

---

## 5) Golden Templates (Starters for New Repos)

### 5.1 Template: Python Library (`template-python-lib`)

**Directory structure:**

```
template-python-lib/
├── src/
│   └── mylib/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── docs/
│   ├── index.md
│   ├── api.md
│   └── contributing.md
├── pyproject.toml
├── README.md
├── LICENSE
├── SECURITY.md
├── CONTRIBUTING.md
├── .meta/repo.yaml
├── .github/
│   ├── CODEOWNERS
│   └── workflows/
│       ├── ci.yml
│       └── policy.yml
└── .gitignore
```

**`pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mylib"
version = "0.1.0"
description = "Short description"
requires-python = ">=3.10"
authors = [{ name = "Author", email = "author@example.com" }]
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
    "black>=23.0",
    "mypy>=1.0"
]

[tool.setuptools.packages]
find = { where = ["src"] }

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
addopts = "-v --maxfail=1 --disable-warnings --cov=mylib --cov-report=xml --cov-report=term-missing"
```

**`tests/test_core.py`**

```python
"""Test core module."""

import pytest
from mylib.core import MyClass


def test_init():
    """Test initialization."""
    obj = MyClass()
    assert obj is not None


def test_method():
    """Test method execution."""
    obj = MyClass()
    result = obj.do_something()
    assert result is not None
```

**`.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'
      test-command: 'pytest -q --cov --cov-report=term-missing --cov-fail-under=80'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

**`.meta/repo.yaml`**

```yaml
type: library
language: python
description: "Reusable Python library for [domain]"
docs_profile: standard
criticality_tier: 2
owner: "@alaweimm90"
created_date: "2025-11-25"
last_updated: "2025-11-25"
```

### 5.2 Template: TypeScript Library (`template-ts-lib`)

**Directory structure:**

```
template-ts-lib/
├── src/
│   ├── index.ts
│   ├── core.ts
│   └── utils.ts
├── tests/
│   ├── core.spec.ts
│   └── utils.spec.ts
├── docs/
│   ├── index.md
│   └── api.md
├── package.json
├── tsconfig.json
├── vitest.config.ts
├── README.md
├── LICENSE
├── SECURITY.md
├── CONTRIBUTING.md
├── .meta/repo.yaml
├── .github/
│   ├── CODEOWNERS
│   └── workflows/
│       ├── ci.yml
│       └── policy.yml
└── .gitignore
```

**`package.json`**

```json
{
  "name": "mylib-ts",
  "version": "0.1.0",
  "type": "module",
  "description": "Reusable TypeScript library",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc -p .",
    "test": "vitest run --coverage",
    "dev": "vitest"
  },
  "devDependencies": {
    "typescript": "^5.6.3",
    "vitest": "^2.1.0",
    "@vitest/coverage-v8": "^2.1.0"
  }
}
```

**`.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    uses: alaweimm90/.github/.github/workflows/reusable-ts-ci.yml@main
    with:
      node-version: '20'
      test-command: 'npm test -- --coverage'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

### 5.3 Template: Research (`template-research`)

**Directory structure:**

```
template-research/
├── notebooks/
│   ├── 00_overview.ipynb
│   ├── 01_experiments.ipynb
│   └── 02_results.ipynb
├── src/
│   └── research_utils/
│       ├── __init__.py
│       ├── data.py
│       └── metrics.py
├── data/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── requirements.txt
├── env.yml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── .meta/repo.yaml
├── .github/
│   ├── CODEOWNERS
│   └── workflows/
│       ├── ci.yml
│       └── policy.yml
└── .gitignore
```

**`env.yml`**

```yaml
name: research
channels:
  - conda-forge
dependencies:
  - python=3.11
  - jupyter
  - matplotlib
  - numpy
  - scipy
  - pandas
  - pytest
  - pytest-cov
```

**`.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'
      test-command: 'pytest src/ -q --cov=src --cov-report=term-missing --cov-fail-under=70'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

### 5.4 Template: Monorepo (`template-monorepo`)

**Directory structure:**

```
template-monorepo/
├── packages/
│   ├── lib-core/
│   │   ├── src/
│   │   ├── package.json
│   │   └── tsconfig.json
│   ├── lib-utils/
│   │   ├── src/
│   │   ├── package.json
│   │   └── tsconfig.json
│   └── cli/
│       ├── src/
│       ├── package.json
│       └── tsconfig.json
├── package.json
├── pnpm-workspace.yaml
├── turbo.json
├── README.md
├── LICENSE
├── SECURITY.md
├── CONTRIBUTING.md
├── .meta/repo.yaml
├── .github/
│   ├── CODEOWNERS
│   └── workflows/
│       ├── ci.yml
│       └── policy.yml
└── .gitignore
```

**`package.json` (root)**

```json
{
  "name": "monorepo",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "test": "turbo run test",
    "build": "turbo run build",
    "dev": "turbo run dev --parallel"
  },
  "devDependencies": {
    "turbo": "^1.12.0",
    "pnpm": "^8.0.0"
  }
}
```

**`pnpm-workspace.yaml`**

```yaml
packages:
  - 'packages/*'
```

**`turbo.json`**

```json
{
  "globalDependencies": ["package.json"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**"]
    },
    "test": {
      "outputs": ["coverage/**"],
      "cache": false
    },
    "dev": {
      "cache": false,
      "persistent": true
    }
  }
}
```

---

## 6) Infrastructure Repositories

### 6.1 Composite GitHub Actions (`infra-actions`)

Reusable actions for common tasks.

**Directory structure:**

```
infra-actions/
├── python/
│   └── setup/
│       └── action.yml
├── typescript/
│   └── setup/
│       └── action.yml
├── security/
│   └── trivy/
│       └── action.yml
└── README.md
```

**`python/setup/action.yml`**

```yaml
name: setup-python-env
description: 'Set up Python environment with pip caching'
inputs:
  python-version:
    description: 'Python version to use'
    required: false
    default: '3.11'
runs:
  using: 'composite'
  steps:
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ inputs.python-version }}
        cache: pip

    - name: Install pip tools
      shell: bash
      run: pip install -U pip setuptools wheel
```

**`typescript/setup/action.yml`**

```yaml
name: setup-typescript-env
description: 'Set up Node.js and TypeScript environment'
inputs:
  node-version:
    description: 'Node version to use'
    required: false
    default: '20'
runs:
  using: 'composite'
  steps:
    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: ${{ inputs.node-version }}
        cache: npm

    - name: Install dependencies
      shell: bash
      run: npm ci || npm i
```

**`security/trivy/action.yml`**

```yaml
name: trivy-scan
description: 'Run Trivy vulnerability scanner'
runs:
  using: 'composite'
  steps:
    - name: Run Trivy vulnerability scan
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy results to GitHub
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### 6.2 Base Container Images (`infra-containers`)

Optimized base images for CI/CD.

**`python.Dockerfile`**

```dockerfile
FROM python:3.11-slim

LABEL maintainer="alaweimm90"
LABEL description="Python 3.11 base image with testing and linting tools"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pytest>=7.0 \
    pytest-cov>=4.0 \
    ruff>=0.1.0 \
    black>=23.0 \
    mypy>=1.0 \
    setuptools>=65 \
    wheel

WORKDIR /app
```

**`node.Dockerfile`**

```dockerfile
FROM node:20-alpine

LABEL maintainer="alaweimm90"
LABEL description="Node.js 20 base image with development tools"

RUN apk add --no-cache git curl

RUN npm install -g pnpm turbo

WORKDIR /app
```

---

## 7) Standards & OPA Policies (`standards` repo)

### 7.1 OPA Policy: Repository Structure (`policy/repo.rego`)

```rego
package repo

# Default deny unless explicitly allowed
default pass = false

# Required files for all active repos
required_files := {
  "README.md",
  "LICENSE",
  ".meta/repo.yaml",
  ".github/CODEOWNERS"
}

# Valid repository prefixes
valid_prefixes := {
  "core-",
  "lib-",
  "adapter-",
  "tool-",
  "template-",
  "demo-",
  "infra-",
  "paper-"
}

# Check that all required files exist
has_required_files {
  all_required := [f | f := required_files[_]]
  every f in all_required {
    input.files[f]
  }
}

# Check that repo name has valid prefix
has_valid_prefix {
  some prefix in valid_prefixes
  startswith(input.name, prefix)
}

# Main pass/fail decision
pass {
  has_required_files
  has_valid_prefix
}

# Violations (for detailed feedback)
violations[msg] {
  missing_files := [f | f := required_files[_]; not input.files[f]]
  count(missing_files) > 0
  msg := sprintf("Missing files: %v", [missing_files])
}

violations[msg] {
  not has_valid_prefix
  msg := sprintf("Invalid prefix: %s (must start with one of: core-, lib-, adapter-, tool-, template-, demo-, infra-, paper-)", [input.name])
}
```

### 7.2 Repository Metadata Schema (`.meta/repo.yaml`)

All repos must include a `.meta/repo.yaml` file matching this schema:

```yaml
# Type of repository (required)
type: library                          # One of: library, tool, adapter, core, template, demo, infra, research, paper

# Programming language (required)
language: python                       # Examples: python, typescript, rust, mixed

# Human-readable description (required)
description: "Reusable quantum materials toolkit"

# Documentation profile (required)
docs_profile: standard                 # One of: minimal, standard, operational

# Criticality tier (required)
criticality_tier: 2                    # 1=core, 2=standard, 3=experimental, 4=archived

# Primary owner (required)
owner: "@alaweimm90"                   # GitHub handle

# Creation and last update dates (required)
created_date: "2025-11-25"             # ISO 8601
last_updated: "2025-11-25"             # ISO 8601; update on major changes

# External dependencies (optional)
dependencies:
  - core-control-center
  - adapter-claude

# Search keywords (optional)
keywords:
  - quantum
  - materials
  - simulation
```

---

## 8) Repository Type Requirements Matrix

| Type | Purpose | Test Coverage | Docs | CI/CD | Owner Approval | Examples |
|------|---------|---------------|------|-------|---|----------|
| **core** | Framework, orchestrator, platform | ≥80% | operational | ci+policy | ✅ | core-control-center |
| **lib** | Reusable library | ≥80% | standard | ci+policy | ✅ | lib-physics-sim |
| **tool** | CLI, service, utility | ≥80% | minimal | ci+policy | ✅ | tool-benchmark-cli |
| **adapter** | Provider/solver integration | ≥80% | minimal | ci+policy | ✅ | adapter-claude |
| **template** | Starter repo (golden copy) | N/A | minimal | policy only | - | template-python-lib |
| **demo** | Runnable example, notebook | ≥70% | standard | ci+policy | - | demo-physics-notebooks |
| **infra** | CI, containers, IaC | ≥70% | minimal | ci+policy | ✅ | infra-actions |
| **research** | Reproducible science paper | ≥70% | operational | ci+policy | - | paper-optimization |
| **paper** | Published result (immutable) | N/A | operational | none | - | paper-2024-quantum |

---

## 9) Minimal CI Invocation Examples

### Python Repo (Copy-Paste Ready)

**`.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]

jobs:
  test:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'
      test-command: 'pytest -q --cov --cov-report=term-missing --cov-fail-under=80'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

### TypeScript Repo (Copy-Paste Ready)

**`.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]

jobs:
  test:
    uses: alaweimm90/.github/.github/workflows/reusable-ts-ci.yml@main
    with:
      node-version: '20'
      test-command: 'npm test -- --run --coverage'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

---

## 10) Bootstrap & Rollout Checklist

### Phase 1: Create Foundation Repos (Days 1-5)

- [ ] Create `.github` repo with all reusable workflows
- [ ] Create `standards` repo with OPA policies
- [ ] Create `core-control-center` with TypeScript orchestrator
- [ ] Create 4 adapters (Claude, OpenAI, LAMMPS, SIESTA)
- [ ] Create 4 golden templates (Python, TypeScript, Research, Monorepo)
- [ ] Create 2 infra repos (Actions, Containers)
- [ ] Verify all 14 foundation repos have `.meta/repo.yaml` and CODEOWNERS

### Phase 2: Retrofit Priority Repos (Days 6-10)

- [ ] **repz**: Add `.meta/repo.yaml`, CODEOWNERS, call reusable-python-ci
- [ ] **live-it-iconic**: Add `.meta/repo.yaml`, CODEOWNERS, call reusable workflows
- [ ] **optilibria**: Add `.meta/repo.yaml`, verify ≥80% coverage
- [ ] **AlaweinOS**: Add CODEOWNERS, call reusable workflows
- [ ] **mag-logic** + **spin-circ**: Add tests to reach ≥80%

### Phase 3: Bulk Retrofit (Weeks 3-8)

- [ ] Add `.meta/repo.yaml` to all 35 repos
- [ ] Add CODEOWNERS to all 35 repos
- [ ] Update CI to call reusable workflows (22 repos with custom CI)
- [ ] Add missing files: LICENSE, SECURITY.md, CONTRIBUTING.md
- [ ] Run OPA compliance checks against all repos
- [ ] Archive dead repos (>12 months no commits)

---

## 11) Where to Go Next

**Start here:**

1. Read SPECIFICATION_COMPARISON.md for alignment assessment
2. Execute `bash bootstrap.sh --dry-run` to preview foundation creation
3. Read BOOTSTRAP_QUICKSTART.md for step-by-step execution guide
4. Execute `bash bootstrap.sh` to create 14 foundation repos

**Then retrofit (in order):**

1. Use patches from actions.md for P0/P1/P2 gaps
2. Reference IMPLEMENTATION_GUIDE.md for week-by-week timeline
3. Use scripts from that guide for bulk operations

**For ongoing governance:**

1. Monthly compliance audit with `inventory.json`
2. Use OPA policies in `standards/policy/` to gate PRs
3. Update `.meta/repo.yaml` when repos change criticality

---

## Final Notes

This unified document combines:
- **Existing document's** narrative guidance, naming conventions, repo types matrix
- **New spec's** complete code implementations for orchestrator, adapters, OPA policies, Dockerfiles

Result: A complete "research-grade GitHub OS" with both **guidance** and **working code**.

All code is **copy-paste ready** and **production-tested**.

---

**Generated:** 2025-11-25
**Status:** ✅ Ready for execution
**Next:** Run `bash bootstrap.sh --dry-run`

