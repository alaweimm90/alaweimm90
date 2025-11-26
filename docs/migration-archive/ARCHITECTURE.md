# System Architecture

## Three-Layer Enforcement Model

```
Layer 1: metaHub (Local Gate)
├─ scripts/govern.sh → orchestrator
├─ scripts/build_tree.py → file tree snapshot
├─ .pre-commit-config.yaml → hook definitions
└─ Validates before every commit (OPA + linters + projects)

Layer 2: alaweimm90 (Reusable CI)
├─ metaHub/templates/workflows/ → templates
├─ metaHub/cli/enforce.py → idempotent sync
└─ All 55 repos call reusable (zero duplication)

Layer 3: organizations/ (Project Catalog)
├─ .project.yaml (per-project manifest)
├─ .github/workflows/organizations-policy.yml (CI validation)
├─ PROJECTS_INDEX.md (auto-generated)
└─ .projects.json (machine-readable)
```

## Key Components

### Governance Infrastructure (metaHub/)

**CLI Tools** (`metaHub/cli/`):
- `catalog.py` - Discover & seed project manifests
- `meta.py` - Scan, validate & promote projects
- `enforce.py` - Sync governance files across repos

**Policies** (`metaHub/policies/`):
- `organizations_policy.rego` - Project manifest validation
- `repo_structure.rego` - Repository compliance

**Templates** (`metaHub/templates/`):
- `governance/` - LICENSE, SECURITY.md, CODEOWNERS
- `pre-commit/` - Hook configs (Python, generic)
- `workflows/` - Reusable CI templates

**Linter Configs** (`metaHub/linters/`):
- `markdownlint.yaml`
- `ruff.toml`
- `prettier.config.js`
- `eslint.config.js`

**Documentation** (`metaHub/docs/`):
- `enforcement-guide.md` - Three-layer explanation
- `projects-reference.md` - Working with projects
- `cli-tools.md` - Tool usage
- `opa-policies.md` - Policy rules
- `troubleshooting.md` - Common issues

### Execution Layer (scripts/)

- `govern.sh` - Local enforcement gate orchestrator
- `build_tree.py` - Deterministic file tree builder for OPA

### Project Catalog (organizations/)

**Structure**:
- 5 domains: alaweimm90-science, alaweimm90-tools, alaweimm90-business, alaweimm90-os, meathead-physicist
- 80+ projects with `.project.yaml` manifests
- Auto-generated: PROJECTS_INDEX.md, .projects.json

**Manifest Schema** (`.project.yaml`):
```yaml
title: string                    # Project name
description: string             # One-line summary
domain: science|tools|business|os|research
status: active|dormant|archived
type: library|application|research|documentation|infrastructure
language: python|typescript|javascript|rust|na
priority: p1|p2|p3
promotion:                       # Optional
  target_repo_name: string
  template: string
  visibility: public|private
  reason: string
```

### Root Documentation

**3 files only** (zero sprawl):
- `README.md` - Navigation hub
- `ARCHITECTURE.md` - This document (system design)
- `QUICK_START.md` - Getting started (day-1 operations)

**Deep docs** in `metaHub/docs/` (5 files):
- Enforcement guide, projects reference, CLI tools, OPA policies, troubleshooting

**Historical archive** in `.archive/docs-historical/`:
- 28+ timestamped implementation documents
- For reference only; not for daily use

## Design Principles

### 1. Single Source of Truth (SSOT)
- `metaHub/` is canonical for all governance
- No scattered configs (.pre-commit-config.yaml deleted from organizations/)
- One template source, one policy source

### 2. Idempotent Operations
- All commands safe to re-run
- `enforce.py` fixes drift automatically
- No manual state tracking needed

### 3. Zero Duplication
- All 55 repos call reusable CI workflows
- One change = automatic rollout to all repos
- Governance templates centralized

### 4. Automatic Compliance
- Pre-commit hooks block bad commits
- OPA policies enforce structure
- CI validates before merge

### 5. Complete Visibility
- Projects auto-cataloged
- Indexes auto-generated
- Manifests machine-readable

## Workflow Examples

### Adding a Project

```bash
mkdir -p organizations/tools/my-lib
python metaHub/cli/catalog.py --seed      # Create manifest
# Edit manifest (fill title, description, etc.)
./scripts/govern.sh                        # Validate
git push                                   # CI validates
```

### Promoting Project to Repo

```bash
# Update .project.yaml with promotion config
python metaHub/cli/meta.py promote-project tools/my-lib
cd ../my-lib
git push -u origin main
# Result: New repo with all governance files
```

### Syncing Governance to All Repos

```bash
python metaHub/cli/enforce.py --all-inventory --dry-run  # Preview
python metaHub/cli/enforce.py --all-inventory             # Apply
```

### Updating CI for All 55 Repos

```bash
# Edit reusable workflow
vim alaweimm90/.github/workflows/reusable-python-ci.yml
git push
# All 55 repos pick up change automatically (@main reference)
```

## Success Criteria

✅ All 55 repos have `.meta/repo.yaml`
✅ All repos call reusable CI (zero custom CI)
✅ All projects have `.project.yaml` manifests (100% compliance)
✅ No OPA policy violations in pre-commit
✅ Auto-indexed projects (PROJECTS_INDEX.md, .projects.json)
✅ Zero governance drift (enforce.py keeps repos compliant)
✅ New repos auto-compliant via templates

---

**Last Updated:** 2025-11-25
**Status:** ✅ PRODUCTION READY
