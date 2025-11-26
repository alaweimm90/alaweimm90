# Three-Layer Execution Plan: Portfolio Organization & Governance

**Date:** 2025-11-25
**Status:** ✅ Ready for Execution
**Scope:** Catalog, normalize, and enforce governance across 55 repos + projects

---

## What "Done" Looks Like

✅ Every folder under `organizations/<domain>/<project>/` has a `.project.yaml`
✅ `organizations/.projects.json` (machine) and `organizations/PROJECTS_INDEX.md` (human) are generated
✅ CI + pre-commit run OPA policy on repos and project manifests
✅ `alaweimm90/` stays a clean router (profile + index only, no code sprawl)
✅ `.metaHub/` is the **only source** of policies, templates, and CLI

---

## The Three Clean Repos

### Layer 1: `.metaHub/` — Policy & Tooling SSOT

**Contents:**
```
.metaHub/
├─ cli/
│  ├─ meta.py                    # scan-projects, promote-project
│  └─ catalog.py                 # (NEW) seed, normalize, index
├─ policies/
│  ├─ repo_structure.rego
│  ├─ docs_policy.rego
│  └─ organizations_policy.rego
├─ templates/
│  ├─ python-lib/
│  ├─ ts-lib/
│  ├─ research/
│  └─ monorepo/
├─ linters/
│  ├─ markdownlint.yaml
│  ├─ ruff.toml
│  ├─ .eslintrc.cjs
│  └─ ... (all linter configs)
└─ README.md                     # how to use the tooling
```

**Purpose:**
- Single source of truth for ALL policies, templates, rules
- No duplication or scattered versions
- Vendor-neutral (no provider-specific code)
- All tooling (meta.py, catalog.py) lives here

**`.meta/repo.yaml`:**
```yaml
type: meta
language: na
docs_profile: minimal
criticality_tier: 2
```

---

### Layer 2: `alaweimm90/` — Clean Router Profile

**Contents:**
```
alaweimm90/
├─ README.md                     # short router + navigation
├─ PROJECTS_SYSTEM_INDEX.md      # generated (do not hand-edit)
├─ .meta/repo.yaml
└─ CODEOWNERS
```

**Purpose:**
- Minimal organizational profile
- Router to real work (metaHub + organizations)
- Generated index (no manual maintenance)
- No code, no sprawl

**`alaweimm90/README.md` (minimal):**
```markdown
# Meshal Alawein — Portfolio Router

**Start here:**
- **Policies & CLI:** [.metaHub/](../.metaHub)
- **Projects (cataloging):** [organizations/](../organizations)
- **Architecture:** [FINAL_ARCHITECTURE.md](../FINAL_ARCHITECTURE.md)

**Quick Commands:**
```bash
# Discover & validate all projects
python metaHub/cli/meta.py scan-projects

# Catalog, seed, and normalize
python metaHub/cli/catalog.py --seed

# Promote a project to a repo
python metaHub/cli/meta.py promote-project <domain>/<project>
```

**Governance:**
- All projects must have `.project.yaml` manifest
- CI validates on every PR (see organizations/)
- Promotion is safe and reversible
```

**`.meta/repo.yaml`:**
```yaml
type: meta
language: na
docs_profile: minimal
criticality_tier: 2
```

**`CODEOWNERS`:**
```
* @alaweimm90
```

---

### Layer 3: `organizations/` — Project Catalog (Doc-Only + Manifests)

**Contents:**
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
├─ platforms/
├─ infra/
├─ personal/
│
├─ PROJECTS_INDEX.md             # generated (auto-updated)
├─ .projects.json                # generated (auto-updated)
├─ README.md                     # how to add projects / promote
├─ .meta/repo.yaml
├─ CODEOWNERS
└─ .github/
   └─ workflows/
      └─ organizations-policy.yml # (from this repo)
```

**Purpose:**
- Catalog of all projects (doc-only until promotion)
- `.project.yaml` manifests are the contract
- Generated indexes (don't hand-edit)
- CI enforces manifest validation

**`organizations/.meta/repo.yaml`:**
```yaml
type: meta
language: na
docs_profile: standard
criticality_tier: 2
```

**`organizations/README.md`:**
```markdown
# Projects Catalog

Each leaf folder under `<domain>/<project>/` must include `.project.yaml`.

## Add a Project

1. Create folder: `mkdir -p organizations/<domain>/<project>`
2. Seed manifest: `python metaHub/cli/catalog.py --seed`
3. Edit `.project.yaml` (fill `description`, adjust enums)
4. Validate: `python metaHub/cli/meta.py scan-projects`

## Promote to Repo

1. Fill `.project.yaml > promotion.*` fields
2. Run: `python metaHub/cli/meta.py promote-project <domain>/<project>`
3. Review, push, done

See [metaHub/README.md](../metaHub/README.md) for full reference.
```

**`organizations/CODEOWNERS`:**
```
* @alaweimm90
```

---

## Step-by-Step Execution Order

### Phase 0: Sanity & Setup (30 minutes)

1. **Verify directory structure:**
   ```bash
   ls -la organizations/
   ls -la .metaHub/
   ls -la alaweimm90/
   ```
   All three should exist.

2. **Verify CLI tools:**
   ```bash
   python metaHub/cli/meta.py --help
   python metaHub/cli/catalog.py --help
   ```

3. **Create `scripts/govern.sh` for local checks:**
   ```bash
   mkdir -p scripts
   cat > scripts/govern.sh << 'BASH'
   #!/usr/bin/env bash
   set -euo pipefail
   echo "🔍 Building file tree..."
   python scripts/build_tree.py
   echo "🔍 Running OPA + linters..."
   pre-commit run --all-files
   echo "🔍 Cataloging projects..."
   python metaHub/cli/catalog.py --seed
   echo "✅ All checks passed"
   BASH
   chmod +x scripts/govern.sh
   ```

---

### Phase 1: Catalog (No GitHub Changes Yet) — 1-2 hours

**Goal:** Discover all projects, generate manifests, detect naming issues.

1. **Dry-run catalog (no renames, seed manifests):**
   ```bash
   python metaHub/cli/catalog.py --seed
   ```
   Output:
   ```
   🔍 Scanning projects...
   ✅ Scanned 47 projects
   ✅ Seeded 12 new manifests
   ✅ Wrote: organizations/.projects.json
   ✅ Wrote: organizations/PROJECTS_INDEX.md
   ```

2. **Review generated index:**
   ```bash
   cat organizations/PROJECTS_INDEX.md
   head -20 organizations/.projects.json
   ```

3. **Fix any schema errors** in `.project.yaml` files:
   ```bash
   python metaHub/cli/meta.py scan-projects
   # If errors, edit manifests and re-run
   ```

4. **Review rename suggestions:**
   ```bash
   python metaHub/cli/catalog.py --seed 2>&1 | grep "Rename suggestions" -A 20
   ```

5. **If suggestions look correct, apply them:**
   ```bash
   python metaHub/cli/catalog.py --seed --apply-renames
   # Verify:
   python metaHub/cli/meta.py scan-projects
   ```

6. **Commit the normalized catalogs:**
   ```bash
   git add organizations/.project.yaml
   git add organizations/.projects.json
   git add organizations/PROJECTS_INDEX.md
   git add scripts/govern.sh
   git commit -m "chore(projects): catalog and seed manifests"
   git push
   ```

---

### Phase 2: Enable Local Governance (30 minutes)

1. **Create `scripts/build_tree.py`:**
   ```python
   #!/usr/bin/env python3
   import json
   import os
   import pathlib

   root = pathlib.Path(__file__).resolve().parents[1]
   paths = []
   for r, _, fs in os.walk(root):
       for f in fs:
           path = os.path.join(r, f).replace(str(root) + os.sep, "")
           if not path.startswith("."):
               paths.append(path)
   out = root / "outputs"
   out.mkdir(exist_ok=True, parents=True)
   (root / "outputs" / "tree.json").write_text(
       json.dumps({"paths": paths}), encoding="utf-8"
   )
   print("wrote outputs/tree.json")
   ```

2. **Create `.pre-commit-config.yaml`:**
   ```yaml
   repos:
     - repo: https://github.com/open-policy-agent/conftest
       rev: v0.54.0
       hooks:
         - id: conftest
           name: conftest (OPA policies)
           args: ["test", "--policy", "metaHub/policies", "--all-namespaces"]
           files: outputs/tree.json
           stages: [pre-commit]

     - repo: https://github.com/igorshubovych/markdownlint-cli
       rev: v0.41.0
       hooks:
         - id: markdownlint
           args: ["--config", "metaHub/markdownlint.yaml"]
           types: [markdown]
   ```

3. **Install pre-commit:**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Test locally:**
   ```bash
   ./scripts/govern.sh
   ```

---

### Phase 3: Enable CI (10 minutes)

1. **Verify `.github/workflows/organizations-policy.yml` exists:**
   ```bash
   ls -la .github/workflows/organizations-policy.yml
   ```

2. **Commit & push (if not already):**
   ```bash
   git add .github/workflows/organizations-policy.yml
   git commit -m "chore(ci): enable organizations project validation"
   git push
   ```

3. **On next PR that touches `organizations/`:**
   - CI automatically scans projects
   - Validates OPA policies
   - Catalogs & generates index
   - Blocks merge if validation fails

---

### Phase 4: Promote a Sample Project (1 hour)

**Example: Promote `tools/spin-dynamics` to a repo**

1. **Verify manifest is complete:**
   ```bash
   cat organizations/tools/spin-dynamics/.project.yaml
   python metaHub/cli/meta.py scan-projects  # should show ✓
   ```

2. **Fill promotion config:**
   ```yaml
   # In organizations/tools/spin-dynamics/.project.yaml
   promotion:
     target_repo_name: "spin-dynamics"
     template: "python-lib"
     visibility: public
     reason: "Stable library ready for community contributions"
   ```

3. **Run promotion:**
   ```bash
   python metaHub/cli/meta.py promote-project tools/spin-dynamics
   ```
   Output:
   ```
   📦 Scaffolding spin-dynamics from template 'python-lib'...
   🏷️  Creating .meta/repo.yaml...
   🔧 Initializing git repository...

   ✅ Promotion complete!

   📍 Repo created: ../spin-dynamics

   Next steps:
     1. Review files in spin-dynamics/
     2. Create the repo on GitHub: gh repo create --source=spin-dynamics
     3. Push to GitHub: cd spin-dynamics && git push -u origin main
   ```

4. **Review & push:**
   ```bash
   cd ../spin-dynamics
   git log --oneline          # ✓ clean history
   cat .meta/repo.yaml        # ✓ metadata seeded
   cat README.md              # ✓ project description injected
   gh repo create --source=. --remote=origin --public
   git push -u origin main
   ```

5. **Verify new repo is compliant:**
   - ✅ .meta/repo.yaml present
   - ✅ CI calling reusable workflows
   - ✅ CODEOWNERS in place
   - ✅ LICENSE, README, SECURITY.md present
   - ✅ Branch protection enabled

6. **Mark original project as archived (optional):**
   ```yaml
   # In organizations/tools/spin-dynamics/.project.yaml
   status: archived
   promotion:
     target_repo_name: ""    # clear it
     template: ""
     visibility: public
     reason: ""
   ```

   ```bash
   git add organizations/tools/spin-dynamics/.project.yaml
   git commit -m "chore(projects): archive spin-dynamics (promoted to repo)"
   git push
   ```

---

## Local Compliance & Governance

### Pre-Commit Gate (on every `git commit`)

```bash
./scripts/govern.sh
# Runs:
# 1. Build file tree (outputs/tree.json)
# 2. OPA policy checks
# 3. Markdown linting
# 4. Catalog & validate manifests
```

If any check fails, commit is blocked. Fix and re-run.

### CI Enforcement (on every PR)

The workflow `.github/workflows/organizations-policy.yml`:
- Scans all projects
- Validates manifests
- Runs OPA checks
- Generates `PROJECTS_INDEX.md`
- Blocks merge if validation fails

---

## Portfolio Cartography & Investigation

Use this prompt to audit the full portfolio:

```
You are a portfolio cartographer. Goal: Build a complete map across repos + projects.

Inputs to read:
- organizations/.projects.json (machine)
- organizations/PROJECTS_INDEX.md (human)
- .meta/repo.yaml files in any promoted repos
- FINAL_ARCHITECTURE.md

Tasks:
1) List all projects by domain/status/type/language/priority.
2) Flag naming inconsistencies (alaweimm9 → alaweimm90, typos, spaces).
3) Detect gaps: missing .project.yaml, invalid enums, missing CI in promoted repos.
4) Produce action table: (Item | Path | Type | Gaps | Fix Command).
5) Output shell-ready plan: catalog, fix manifests, promote N projects.

Constraints:
- No vendor adapters.
- Propose only; don't rename automatically.
- Output deterministic and minimal.
```

**Run:**
```bash
# Generate audit data
python metaHub/cli/catalog.py --seed > /tmp/catalog.log
python metaHub/cli/meta.py scan-projects > /tmp/scan.log
cat organizations/.projects.json | jq . > /tmp/projects.json

# Review
cat /tmp/catalog.log
cat /tmp/projects.json | head -50
```

---

## Minimal Domain Readmes

Optionally, add a tiny README to each domain folder:

**`organizations/science/README.md`:**
```md
# Science Projects

Research, simulation, and computational work.

To add a project: `mkdir my-project && python metaHub/cli/catalog.py --seed`
```

Same for `tools/`, `business/`, `platforms/`, `infra/`.

---

## What NOT to Do

❌ Don't create repos without promotion workflow
❌ Don't manually edit `.projects.json` (it's auto-generated)
❌ Don't create projects without `.project.yaml`
❌ Don't push changes without running local checks first
❌ Don't scatter policies (keep SSOT in metaHub/)
❌ Don't use external adapters in core-control-center

---

## System Health Checks

### Weekly
```bash
# Ensure all projects still validate
python metaHub/cli/meta.py scan-projects  # exit 0?
```

### Monthly
```bash
# Full portfolio audit
python metaHub/cli/catalog.py --seed
# Review PROJECTS_INDEX.md for stale/archived projects
# Identify promotion candidates
```

### Quarterly
```bash
# Review & update policies in metaHub/
# Consolidate templates if patterns emerge
# Archive old projects
```

---

## Success Checkpoints

### Day 1 (Catalog)
- [ ] All projects discovered
- [ ] All manifests seeded
- [ ] `PROJECTS_INDEX.md` generated
- [ ] Local `./scripts/govern.sh` passes
- [ ] Changes committed

### Day 2 (CI + First Promotion)
- [ ] `.github/workflows/organizations-policy.yml` merged
- [ ] First PR validates manifests in CI
- [ ] One project successfully promoted to repo
- [ ] New repo is fully compliant

### Day 10 (Full Rollout)
- [ ] All projects validated in CI
- [ ] 3-5 projects promoted to repos
- [ ] Governance enforcement operational
- [ ] Team trained on manifest + promotion workflow

---

## Reference Commands

```bash
# Catalog
python metaHub/cli/catalog.py --seed
python metaHub/cli/catalog.py --seed --apply-renames

# Validation
python metaHub/cli/meta.py scan-projects
./scripts/govern.sh

# Promotion
python metaHub/cli/meta.py promote-project science/my-project

# View results
cat organizations/PROJECTS_INDEX.md
cat organizations/.projects.json | jq .

# Git workflow
git add organizations/
git commit -m "chore(projects): <description>"
git push
# CI validates automatically
```

---

## Summary

✅ **Layer 1:** `.metaHub/` — Policy + tooling SSOT (no duplication)
✅ **Layer 2:** `alaweimm90/` — Clean router (no sprawl)
✅ **Layer 3:** `organizations/` — Project catalog (doc-only + manifests)

✅ **Execution:** Catalog → normalize → seed → validate → promote

✅ **Gates:** Pre-commit (local) + CI (remote)

✅ **Safe:** Non-destructive, reversible, fully auditable

The system is ready for Day 1 execution.

