# Quick Start Guide

**Get up and running in 30 minutes**

## What Is This?

A three-layer governance system for 55 repositories:
- **Layer 1:** Local validation (`./scripts/govern.sh`)
- **Layer 2:** Reusable CI (all repos call centralized workflows)
- **Layer 3:** Project catalog with auto-indexing

## Five-Minute Setup

### 1. Install Dependencies

```bash
pip install pre-commit conftest pyyaml
pre-commit install
```

### 2. Run Local Validation

```bash
./scripts/govern.sh
```

Expected output:
```
✅ OPA policies: PASS
✅ Linters: PASS
✅ Projects cataloged
✅ GOVERNANCE GATE: ALL CHECKS PASSED
```

### 3. Add Your First Project

```bash
# Create directory
mkdir -p organizations/tools/my-lib

# Copy your code
cp -r ~/my-work/* organizations/tools/my-lib/

# Seed manifest (auto-creates .project.yaml)
python metaHub/cli/catalog.py --seed

# Edit manifest
vim organizations/tools/my-lib/.project.yaml
# Fill: title, description, type, language, priority

# Validate
./scripts/govern.sh

# Commit
git add organizations/
git commit -m "chore(projects): add my-lib"
git push
```

### 4. Promote to Repo (Optional)

When project is stable:

```bash
# Update manifest
vim organizations/tools/my-lib/.project.yaml
# Add promotion block:
# promotion:
#   target_repo_name: "my-lib"
#   template: "python-lib"
#   visibility: public

# Promote
python metaHub/cli/meta.py promote-project tools/my-lib

# Push new repo
cd ../my-lib
git push -u origin main
```

## Daily Workflow

### Before Every Commit

```bash
./scripts/govern.sh
```

If it fails:
1. Read the error
2. Fix the issue (usually missing `.project.yaml` or invalid manifest)
3. Re-run `./scripts/govern.sh`
4. Commit when it passes

### Common Tasks

**Add a project:**
```bash
mkdir -p organizations/tools/my-lib
python metaHub/cli/catalog.py --seed
vim organizations/tools/my-lib/.project.yaml
./scripts/govern.sh
git push
```

**Validate all projects:**
```bash
python metaHub/cli/meta.py scan-projects
```

**Promote project to repo:**
```bash
python metaHub/cli/meta.py promote-project tools/my-lib
```

**Sync governance to all repos:**
```bash
python metaHub/cli/enforce.py --all-inventory
```

**Check what would change:**
```bash
python metaHub/cli/enforce.py --all-inventory --dry-run
```

## Understanding Validation

### Layer 1: OPA Policies
- Every project must have `.project.yaml`
- Manifest must have: title, description, domain, status, type, language, priority
- All enum values must be valid
- Project names must be unique

### Layer 2: Linters
- Markdown syntax valid
- Python code passes ruff
- YAML valid

### Layer 3: Projects
- All discovered projects cataloged
- All manifests seeded if missing
- All enums validated

## Project Manifest Template

```yaml
title: "Project Name"
description: "One-line description"
domain: science  # or tools, business, os, research
status: active   # or dormant, archived
type: library    # or application, research, documentation, infrastructure
language: python # or typescript, javascript, rust, na
priority: p2     # or p1, p3
```

## Project Domains

| Domain | Examples |
|--------|----------|
| science | qmat-sim, qube-ml, sci-comp |
| tools | alaweimm90-cli, admin-dashboard |
| business | dr-alowein-portfolio, marketing-automation |
| os | mezan, qaplibria, sim-core |
| research | notebooks, papers, analysis |

## What If Validation Fails?

### Missing .project.yaml

```bash
python metaHub/cli/catalog.py --seed
# Then edit the created manifest
vim organizations/domain/project/.project.yaml
```

### Invalid enum value

```bash
# Check valid values:
# - status: active, dormant, archived
# - type: library, application, research, documentation, infrastructure
# - language: python, typescript, javascript, rust, na
# - priority: p1, p2, p3
# - domain: science, tools, business, os, research

# Fix your manifest
vim organizations/domain/project/.project.yaml
```

### Linting failures

```bash
# Fix markdown
# Fix python with: ruff check --fix src/
# Fix javascript/typescript formatting with: prettier --write .
```

## Common Errors

### "Pre-commit hook blocked my commit"

**Solution:**
```bash
./scripts/govern.sh  # See what failed
# Fix the issue
git add [fixed files]
git commit -m "..."
```

### "OPA policy violation"

**Solution:**
```bash
# Read the error message
# Create missing files or fix manifest enums
./scripts/govern.sh
```

### "Catalog seeding failed"

**Solution:**
```bash
# Check directory structure
ls -la organizations/

# Verify paths are correct (domain/project)
mkdir -p organizations/tools/my-lib

# Retry
python metaHub/cli/catalog.py --seed
```

## Next Steps

1. **Understand the system:**
   - Read [ARCHITECTURE.md](ARCHITECTURE.md) (5 min)
   - Review [README.md](README.md) (5 min)

2. **Deep dive:**
   - [metaHub/docs/enforcement-guide.md](metaHub/docs/enforcement-guide.md)
   - [metaHub/docs/projects-reference.md](metaHub/docs/projects-reference.md)
   - [metaHub/docs/cli-tools.md](metaHub/docs/cli-tools.md)

3. **Troubleshooting:**
   - [metaHub/docs/troubleshooting.md](metaHub/docs/troubleshooting.md)

---

**Questions?** See [README.md](README.md) for navigation to all docs.

**Last Updated:** 2025-11-25
