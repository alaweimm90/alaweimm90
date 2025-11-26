# Enforcement Layers Guide: metaHub → alaweimm90 → organizations

**Status:** ✅ Complete enforcement stack
**Objective:** Keep portfolio compliant and clutter-free (layer-by-layer)
**Execution:** Sequential (no parallelization; each layer depends on previous)

---

## Overview: Three Enforcement Layers

```
Layer 1: .metaHub
├─ Local governance gate (pre-commit + OPA + catalog)
├─ Scripts: govern.sh, build_tree.py
└─ Gates: OPA policies, linters, project validation

Layer 2: alaweimm90 (org profile)
├─ Router to governance (no code, no sprawl)
├─ Reusable workflows (called by all 55 repos)
└─ CODEOWNERS (ownership matrix)

Layer 3: organizations
├─ Project catalog (doc-only until promotion)
├─ CI enforcement (.github/workflows/organizations-policy.yml)
└─ Automated indexing (.projects.json, PROJECTS_INDEX.md)
```

---

## Layer 1: `.metaHub` — Local Governance Gate

### What It Does

**Before any commit can be pushed**, one script validates everything locally:

```bash
./scripts/govern.sh
```

This runs (in order):
1. **Build file tree** — Deterministic snapshot of all files (OPA input)
2. **OPA policy validation** — Enforce repo structure + project manifests
3. **Linting** — Markdown, Python, JavaScript, YAML
4. **Project cataloging** — Discover projects, seed manifests, validate
5. **Print summary** — Show what passed/failed

### Setup (One-Time)

```bash
# Install pre-commit
pip install pre-commit

# Install the hook
pre-commit install

# Test it works
./scripts/govern.sh
```

### Files Created

| File | Purpose |
|------|---------|
| `scripts/build_tree.py` | Capture deterministic file tree for OPA |
| `scripts/govern.sh` | Main governance gate (orchestrator) |
| `.pre-commit-config.yaml` | Hook definitions (OPA, linters, etc.) |

### Usage

**Run locally before committing:**
```bash
./scripts/govern.sh
# or
pre-commit run --all-files
```

**Automatic on `git commit`:**
```bash
git add .
git commit -m "..."
# Pre-commit hooks run automatically
```

**In CI (on every PR):**
```yaml
# Will be integrated into .github/workflows/organizations-policy.yml
pre-commit run --all-files
```

### What It Validates

✅ **OPA Policies:**
- All repos have required files (.meta/repo.yaml, LICENSE, SECURITY.md, CODEOWNERS)
- All projects have valid .project.yaml manifests
- All project names are unique
- All enums are valid

✅ **Linters:**
- Markdown syntax (markdownlint)
- Python code (ruff)
- JavaScript/TypeScript (prettier)

✅ **Project Catalog:**
- All projects discovered
- All manifests seeded (if missing)
- All enums validated

---

## Layer 2: `alaweimm90` — Org Profile Router

### Structure

```
alaweimm90/
├─ README.md                    # Navigation router
├─ PROJECTS_SYSTEM_INDEX.md     # Generated (auto-maintained)
├─ .meta/repo.yaml
├─ CODEOWNERS
├─ .github/
│  └─ workflows/
│     ├─ reusable-python-ci.yml
│     ├─ reusable-ts-ci.yml
│     └─ reusable-policy.yml
└─ (no code; routing only)
```

### What It Does

**Provides reusable workflows that all 55 repos call** (eliminates CI duplication):

**`reusable-python-ci.yml`:**
```yaml
name: Python CI (Reusable)
on:
  workflow_call:
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with: {python-version: '3.11'}
      - run: pip install -e .[dev]
      - run: pytest tests/ --cov --cov-report=xml
      - run: ruff check src/
      - run: mypy src/
```

**`reusable-ts-ci.yml`:**
```yaml
name: TypeScript CI (Reusable)
on:
  workflow_call:
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: {node-version: '20'}
      - run: pnpm install --frozen-lockfile
      - run: pnpm test -- --run
      - run: pnpm lint
```

**`reusable-policy.yml`:**
```yaml
name: Policy & Standards (Reusable)
on:
  workflow_call:
jobs:
  policy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          required=("README.md" "LICENSE" ".meta/repo.yaml")
          for f in "${required[@]}"; do
            [[ -f "$f" ]] || (echo "❌ Missing $f"; exit 1)
          done
```

### Consumer Repo Usage

Every repo (all 55) has a `.github/workflows/ci.yml` that calls one reusable workflow:

**Python library:**
```yaml
name: ci
on: [push, pull_request]
jobs:
  call-reusable:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
```

**TypeScript library:**
```yaml
name: ci
on: [push, pull_request]
jobs:
  call-reusable:
    uses: alaweimm90/.github/.github/workflows/reusable-ts-ci.yml@main
```

**Result:** All repos automatically:
- ✅ Test with coverage gates
- ✅ Lint code
- ✅ Type-check
- ✅ Validate governance files
- ❌ NO custom CI sprawl

### Updating Reusable Workflows

To update CI for all 55 repos at once:

1. Edit `alaweimm90/.github/workflows/reusable-*.yml`
2. Push to `alaweimm90/`
3. All repos pick up the change on next push (they reference `@main`)

---

## Layer 3: `organizations/` — Project Catalog & Enforcement

### Structure

```
organizations/
├─ science/
│  └─ qmat-sim/
│     └─ .project.yaml
├─ tools/
│  └─ spin-dynamics/
│     └─ .project.yaml
├─ business/
│  └─ repz/
│     └─ .project.yaml
│
├─ PROJECTS_INDEX.md           # Auto-generated table
├─ .projects.json              # Auto-generated inventory
├─ README.md                   # How to add projects
├─ .meta/repo.yaml
├─ CODEOWNERS
└─ .github/
   └─ workflows/
      └─ organizations-policy.yml  # Auto-validates projects
```

### What It Does

**CI workflow validates projects on every PR:**

```yaml
name: organizations-policy
on:
  pull_request:
    paths: ["organizations/**"]
jobs:
  scan-validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python metaHub/cli/catalog.py --seed
      - run: python metaHub/cli/meta.py scan-projects
      # Fails if any project invalid
```

**Auto-commits updated indexes on main:**
```yaml
      - name: Commit index updates
        if: github.ref == 'refs/heads/main'
        run: |
          git config user.name github-actions
          git add organizations/PROJECTS_INDEX.md organizations/.projects.json
          git commit -m "chore(projects): update index" || true
          git push
```

### Usage

**Add a project:**
```bash
mkdir -p organizations/science/my-project
python metaHub/cli/catalog.py --seed
# Manifest auto-created; fill in details
# Push → CI validates automatically
```

**Promote a project to a repo:**
```bash
# Update .project.yaml with promotion config
# Then:
python metaHub/cli/meta.py promote-project science/my-project
# Scaffolds new repo, fully compliant
```

**View all projects:**
```bash
cat organizations/PROJECTS_INDEX.md      # Human-readable
cat organizations/.projects.json | jq .  # Machine-readable
```

---

## End-to-End Workflow: Adding & Promoting a Project

### Day 1: Create Project

```bash
# 1. Create folder
mkdir -p organizations/tools/my-lib

# 2. Add code/docs
cp -r ~/my-work/* organizations/tools/my-lib/

# 3. Catalog & seed manifest
python metaHub/cli/catalog.py --seed

# 4. Edit manifest
vim organizations/tools/my-lib/.project.yaml
# Fill: title, description, type, language, priority

# 5. Validate locally
./scripts/govern.sh  # or pre-commit run --all-files

# 6. Commit
git add organizations/
git commit -m "chore(projects): add my-lib"
git push
# CI auto-validates

# Result: Project cataloged, indexed
```

### Day 30: Promote Project

```bash
# 1. Update manifest (fill promotion block)
vim organizations/tools/my-lib/.project.yaml
# promotion:
#   target_repo_name: "my-lib"
#   template: "python-lib"
#   visibility: public
#   reason: "Stable, production-ready"

# 2. Promote
python metaHub/cli/meta.py promote-project tools/my-lib
# Scaffolds ../my-lib/ with full governance

# 3. Review
cd ../my-lib
git log --oneline
cat .meta/repo.yaml
cat README.md

# 4. Push
gh repo create --source=. --remote=origin --public
git push -u origin main

# Result: New repo with:
# ✅ .meta/repo.yaml (seeded)
# ✅ LICENSE, SECURITY.md (from templates)
# ✅ .github/workflows/ci.yml (calls reusable)
# ✅ CODEOWNERS
# ✅ README (seeded with project description)
# ✅ Full test suite ready
```

---

## Layer 2B: Enforce Governance (Idempotent Codemod)

### What `enforce.py` Does

Ensures every repo has required governance files (safe to re-run):

```bash
# Apply to single repo
python metaHub/cli/enforce.py my-repo

# Apply to all repos in inventory.json
python metaHub/cli/enforce.py --all-inventory

# Dry-run first (see what would change)
python metaHub/cli/enforce.py --all-inventory --dry-run
```

**Automatically:**
1. Creates `.meta/repo.yaml` (infers type/language if missing)
2. Syncs `LICENSE`, `SECURITY.md`, `CODEOWNERS` from templates
3. Rewrites `.github/workflows/ci.yml` to call reusable based on language
4. Syncs `.pre-commit-config.yaml` from templates
5. Prints summary table

**Example output:**
```
✅ ENFORCE: Governance File Sync
Repos to process: 55

✓ qmat-sim
    + .meta/repo.yaml
    + LICENSE
    + SECURITY.md
    + .github/CODEOWNERS
    + .github/workflows/ci.yml

✓ spin-dynamics
    ~ already compliant

~ repz: skipped (directory not found)

Summary: 23/55 repos updated
```

### When to Run

- **Initial setup:** `enforce.py --all-inventory` (brings all repos into compliance)
- **Quarterly:** Re-run to catch any drift
- **After policy change:** Update templates in metaHub/, re-run enforce

---

## Execution Timeline: Three Weeks

### Week 1: Layer 1 Setup

**Day 1-2: Local governance gate**
```bash
pip install pre-commit conftest
pre-commit install
./scripts/govern.sh
# Make sure all gates pass locally
```

**Day 3-5: Test on repos**
```bash
# Edit a repo, commit, verify pre-commit hooks run
git add metaHub/policies/...
git commit -m "test: pre-commit hooks"
# Should block if validation fails; pass if OK
```

### Week 2: Layer 2 Setup

**Day 6-7: Create reusable workflows**
```bash
# In alaweimm90/.github/workflows/:
# - reusable-python-ci.yml
# - reusable-ts-ci.yml
# - reusable-policy.yml
git push
```

**Day 8-10: Update 55 repos to call reusable**
```bash
python metaHub/cli/enforce.py --all-inventory
# All repos now call reusable workflows
git add [all repos]
git commit -m "chore(ci): migrate to reusable workflows"
git push
# CI runs; verify all repos build
```

### Week 3: Layer 3 Enforcement

**Day 11-12: Catalog all projects**
```bash
python metaHub/cli/catalog.py --seed
python metaHub/cli/meta.py scan-projects
# Fix any invalid manifests
git push
```

**Day 13-14: Enable CI + promote samples**
```bash
git add .github/workflows/organizations-policy.yml
git commit -m "chore(ci): enable project validation"
git push

# Promote 1-2 sample projects
python metaHub/cli/meta.py promote-project tools/spin-dynamics
# Verify new repo is compliant
```

**Day 15: Portfolio audit**
```bash
# Run cartography investigation (portfolio view)
# Identify gaps + next promotions
# Document in PORTFOLIO_STATUS.md
```

---

## Maintenance: Weekly Gate

Every developer runs this before pushing:

```bash
./scripts/govern.sh
```

This ensures:
- ✅ No policy drift
- ✅ No governance gaps
- ✅ All projects cataloged & valid
- ✅ Clean merges

---

## Troubleshooting

### "Pre-commit hook blocked my commit"

```bash
# See what failed
./scripts/govern.sh

# Fix the issue (usually manifest schema or governance file)
# Then:
git add [fixed files]
git commit -m "..."
```

### "OPA policy violation"

The error message will say which file violated which rule. Example:

```
ERROR: evaluate: [repo.structure.valid] missing .meta/repo.yaml in: my-repo
```

**Fix:**
```bash
# Create the file (or run enforce.py)
echo "type: library
language: python
docs_profile: standard
criticality_tier: 3" > my-repo/.meta/repo.yaml

git add my-repo/.meta/repo.yaml
git commit -m "..."
```

### "Enforce didn't update my repo"

```bash
# Check if repo exists
ls -la ../my-repo

# Run enforce with --dry-run first
python metaHub/cli/enforce.py my-repo --dry-run

# Then apply
python metaHub/cli/enforce.py my-repo
```

---

## Summary: Three Layers = One Unified System

| Layer | Gate | Input | Output |
|-------|------|-------|--------|
| **1** | Local (pre-commit + OPA) | File tree | ✅/❌ Validation errors |
| **2** | Reusable workflows | Repo code | ✅ CI passed (all repos) |
| **3** | CI + auto-index | Projects | ✅/❌ Project validation + auto-committed index |

**Result:** No drift, no sprawl, complete compliance.

---

## Files & Commands Reference

### Layer 1: Local
```bash
scripts/build_tree.py              # Capture tree
scripts/govern.sh                  # Run all gates
.pre-commit-config.yaml            # Hook definitions

# Commands:
./scripts/govern.sh
pre-commit run --all-files
```

### Layer 2: Reusable CI
```bash
alaweimm90/.github/workflows/reusable-*.yml
metaHub/cli/enforce.py             # Apply to repos

# Commands:
python metaHub/cli/enforce.py --all-inventory
```

### Layer 3: Projects
```bash
organizations/
metaHub/cli/catalog.py             # Discover & seed
metaHub/cli/meta.py                # Validate & promote

# Commands:
python metaHub/cli/catalog.py --seed
python metaHub/cli/meta.py scan-projects
python metaHub/cli/meta.py promote-project <path>
```

---

**Status:** ✅ ENFORCEMENT STACK READY
**Date:** 2025-11-25
**Next Step:** Follow the three-week timeline or use any layer independently

