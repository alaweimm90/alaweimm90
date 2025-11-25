# Implementation Guide: alaweimm90 Golden Path Migration

**Status:** Ready to execute
**Timeline:** 8 weeks (Weeks 1-2: foundations, Weeks 3-8: retrofit)
**Effort:** 120-150 hours total
**Team:** Solo (you) with scripting automation

---

## What You Have

You now have **4 complete audit documents** + **1 production architecture**:

1. **[inventory.json](inventory.json)** - 35 repos cataloged with full metadata
2. **[gaps.md](gaps.md)** - Gap analysis + compliance scorecard
3. **[actions.md](actions.md)** - P0/P1/P2 prioritized fixes with patches
4. **[features.md](features.md)** - 28 capabilities cataloged with cross-links
5. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete starter code for 13 new repos + retrofit templates

---

## What to Do Now (Week 1-2)

### Phase 1: Create 5 Foundations (Day 1-3)

```bash
#!/bin/bash
# bootstrap-foundations.sh

set -e

echo "🚀 Creating foundation repositories..."

# 1. Create .github repo
gh repo create alaweimm90/.github \
  --public \
  --description="Org-wide reusable workflows and health checks" \
  --confirm

# 2. Create standards repo
gh repo create alaweimm90/standards \
  --public \
  --description="SSOT: policies (OPA), styles, naming conventions" \
  --confirm

# 3. Create core orchestrator
gh repo create alaweimm90/core-control-center \
  --public \
  --description="Typed DAG orchestrator + provider interfaces" \
  --confirm

# 4. Retrofit profile repo (already exists)
echo "✅ Profile repo (alaweimm90) ready for retrofit"

# 5. Create adapters
for adapter in adapter-claude adapter-openai adapter-lammps adapter-siesta; do
  gh repo create "alaweimm90/$adapter" \
    --public \
    --description="Provider adapter for control-center" \
    --confirm
done

echo "✅ Created 8 foundation repos"
```

### Phase 1.5: Push Foundation Code (Day 3-5)

For each foundation repo, clone the files from `ARCHITECTURE.md` and push:

```bash
#!/bin/bash
# bootstrap-push-foundations.sh

REPOS=(
  ".github"
  "standards"
  "core-control-center"
  "adapter-claude"
)

for repo in "${REPOS[@]}"; do
  echo "Pushing $repo..."
  cd "/tmp/alaweimm90-$repo"

  # Files from ARCHITECTURE.md go here
  git init -b main
  git remote add origin "https://github.com/alaweimm90/$repo.git"
  git add .
  git commit -m "chore: initial commit from production architecture"
  git push -u origin main

  # Set branch protection
  gh repo edit "alaweimm90/$repo" \
    --enable-auto-merge \
    --enable-discussions=false \
    --enable-projects=false

  # Require ci + policy
  gh api "repos/alaweimm90/$repo/branches/main/protection" \
    -X PUT \
    -f required_status_checks.strict=true \
    -f required_status_checks.contexts='["ci","policy"]' \
    -f required_pull_request_reviews.dismiss_stale_reviews=true \
    -f required_pull_request_reviews.require_code_owner_reviews=true

  cd -
done

echo "✅ Pushed foundation code with branch protection"
```

### Phase 2: Create Templates + Infra (Day 5-7)

```bash
#!/bin/bash
# bootstrap-templates.sh

# Templates
for template in template-python-lib template-ts-lib template-research template-monorepo; do
  gh repo create "alaweimm90/$template" \
    --public \
    --description="Starter template for new repos" \
    --confirm
done

# Infra
gh repo create alaweimm90/infra-actions \
  --public \
  --description="Composite GitHub Actions for CI" \
  --confirm

gh repo create alaweimm90/infra-containers \
  --public \
  --description="GHCR base images for CI speed" \
  --confirm

gh repo create alaweimm90/demo-physics-notebooks \
  --public \
  --description="Runnable Jupyter examples" \
  --confirm

echo "✅ Created 7 template + infra repos"
```

### Phase 3: Retrofit Priority Repos (Week 2-3)

**Priority 1 (highest impact, do these first):**

```bash
#!/bin/bash
# retrofit-priority1.sh

# These repos have the most CI workflows and should call reusable
PRIORITY1=(
  "organizations/alaweimm90-business/repz"
  "organizations/alaweimm90-business/live-it-iconic"
  "organizations/AlaweinOS/optilibria"
  "organizations/AlaweinOS/AlaweinOS"
  "organizations/alaweimm90-science/mag-logic"
)

for repo_path in "${PRIORITY1[@]}"; do
  repo_name=$(basename "$repo_path")
  org=$(basename $(dirname "$repo_path"))

  echo "Retrofitting $repo_path..."

  # Add required files if missing
  [ ! -f "$repo_path/LICENSE" ] && cp templates/LICENSE "$repo_path/"
  [ ! -f "$repo_path/SECURITY.md" ] && cp templates/SECURITY.md "$repo_path/"
  [ ! -f "$repo_path/CONTRIBUTING.md" ] && cp templates/CONTRIBUTING.md "$repo_path/"
  [ ! -f "$repo_path/.meta/repo.yaml" ] && cp templates/.meta-repo.yaml "$repo_path/.meta/repo.yaml"

  # Update CI to call reusable (examples in ARCHITECTURE.md)
  if [ -f "$repo_path/.github/workflows/ci.yml" ]; then
    echo "CI exists; review manually to call reusable workflows"
  fi

  echo "✅ Retrofitted $repo_name"
done
```

---

## What to Track

Use this checklist per week:

### Week 1-2 Checklist (Foundations)

- [ ] `.github` repo created + code pushed + branch protection
- [ ] `standards` repo created + code pushed + branch protection
- [ ] `core-control-center` repo created + code pushed + CI passing
- [ ] `adapter-claude`, `adapter-openai`, `adapter-lammps`, `adapter-siesta` created
- [ ] All 13 new repos have `.meta/repo.yaml` + required files
- [ ] Branch protection on all foundations requires ci + policy

### Week 3-4 Checklist (Templates)

- [ ] `template-python-lib` created + tested
- [ ] `template-ts-lib` created + tested
- [ ] `template-research` created + tested
- [ ] `template-monorepo` created + tested
- [ ] `infra-actions` created with at least 3 composite actions
- [ ] `infra-containers` created with base images

### Week 5-8 Checklist (Retrofit)

**Priority 1 (5 repos):**
- [ ] repz: calls reusable workflows, has .meta/repo.yaml
- [ ] live-it-iconic: calls reusable workflows, has .meta/repo.yaml
- [ ] optilibria: calls reusable workflows, has .meta/repo.yaml
- [ ] AlaweinOS: calls reusable workflows, has .meta/repo.yaml
- [ ] mag-logic: has ≥80% test coverage, has .meta/repo.yaml

**Priority 2 (15 repos):**
- [ ] alaweimm90-science/* (5 repos): add .meta/repo.yaml, SECURITY.md, LICENSE
- [ ] alaweimm90-tools/* (10 repos): add .meta/repo.yaml, SECURITY.md, LICENSE

**Priority 3 (10+ repos):**
- [ ] alaweimm90-business/* (6 repos): add missing files
- [ ] MeatheadPhysicist: add missing files
- [ ] Others: archive or exempt

---

## Scripts You'll Need

### 1. Generate `.meta/repo.yaml` for all repos

```bash
#!/bin/bash
# generate-meta-yaml.sh

# Repos and their metadata
declare -A REPOS=(
  ["organizations/alaweimm90-business/repz"]="core|platform|80"
  ["organizations/alaweimm90-business/live-it-iconic"]="core|platform|80"
  ["organizations/alaweimm90-science/mag-logic"]="lib|scientific-library|80"
  ["organizations/alaweimm90-science/qmat-sim"]="lib|quantum-materials|80"
  # ... add all 35
)

for repo_path in "${!REPOS[@]}"; do
  IFS='|' read -r prefix type coverage <<< "${REPOS[$repo_path]}"
  repo_name=$(basename "$repo_path")

  mkdir -p "$repo_path/.meta"

  cat > "$repo_path/.meta/repo.yaml" <<EOF
type: $type
language: python  # TODO: detect
description: TODO  # Extract from README
docs_profile: standard
criticality_tier: 2
owner: "@alaweimm90"
created_date: "2025-11-25"
last_updated: "2025-11-25"
EOF

  echo "✅ Created .meta/repo.yaml for $repo_name"
done
```

### 2. Bulk add missing files

```bash
#!/bin/bash
# add-missing-files.sh

find organizations -name ".git" -type d | while read git_dir; do
  repo_path=$(dirname "$git_dir")
  repo_name=$(basename "$repo_path")

  # Add LICENSE if missing
  if [ ! -f "$repo_path/LICENSE" ]; then
    cp templates/LICENSE "$repo_path/" && echo "✅ Added LICENSE to $repo_name"
  fi

  # Add SECURITY.md if missing
  if [ ! -f "$repo_path/SECURITY.md" ]; then
    cp templates/SECURITY.md "$repo_path/" && echo "✅ Added SECURITY.md to $repo_name"
  fi

  # Add CONTRIBUTING.md if missing
  if [ ! -f "$repo_path/CONTRIBUTING.md" ]; then
    cp templates/CONTRIBUTING.md "$repo_path/" && echo "✅ Added CONTRIBUTING.md to $repo_name"
  fi

  # Add CODEOWNERS if missing
  if [ ! -f "$repo_path/.github/CODEOWNERS" ]; then
    mkdir -p "$repo_path/.github"
    cp templates/CODEOWNERS "$repo_path/.github/" && echo "✅ Added CODEOWNERS to $repo_name"
  fi
done
```

### 3. Update CI to call reusable workflows

```bash
#!/bin/bash
# retrofit-ci-to-reusable.sh

# For Python repos
for repo_path in organizations/alaweimm90-science/*; do
  if grep -q "pytest" "$repo_path/pyproject.toml" 2>/dev/null; then
    cat > "$repo_path/.github/workflows/ci.yml" <<'EOF'
name: CI
on: [push, pull_request]

jobs:
  python:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
EOF
    echo "✅ Updated $repo_path CI"
  fi
done
```

---

## Success Metrics

### By Week 4 (End of Phase 2)

- ✅ 13 new foundation/template repos created
- ✅ All new repos pass ci + policy
- ✅ All new repos have .meta/repo.yaml
- ✅ 100% required files present in new repos

### By Week 8 (End of Phase 3)

- ✅ 35 repos have .meta/repo.yaml (100%)
- ✅ 35 repos have LICENSE (100%)
- ✅ 35 repos have SECURITY.md (100%)
- ✅ 35 repos have CONTRIBUTING.md (100%)
- ✅ 22 repos call reusable workflows (100% of active CI)
- ✅ All libraries/tools ≥80% test coverage
- ✅ All demos ≥70% test coverage
- ✅ 0 repos with missing required files

### Post-Week 8 (Ongoing)

- ✅ Monthly compliance audit (automated)
- ✅ Quarterly gap review
- ✅ Policy/standards versioning with bot PRs to consumers

---

## Rollback Plan

If something breaks:

1. **CI fails:** Roll back reusable workflow to previous tag
   ```bash
   # In affected repo
   git checkout <commit_before_workflow_change> .github/workflows/ci.yml
   git push
   ```

2. **Policy too strict:** Update `standards/EXCEPTIONS.md`
   ```yaml
   ## repo-name
   - **Exception:** Skip <check>
   - **Owner:** @alaweimm90
   - **Expiry:** <date>
   - **Reason:** <reason>
   ```

3. **Full rollback:** All repos are independent; revert changes per repo

---

## Key Decisions Made

### Why Reusable Workflows (Not Shared Actions)?

- **Reusable workflows** = entire job templates (faster, easier to maintain)
- **Shared actions** = single tool invocations (harder to version)
- **Trade-off:** Slightly more YAML per repo, but centralized enforcement

### Why OPA (Not Renovate Bot)?

- **OPA** = policy-as-code (gates PRs before merge)
- **Renovate** = dependency bot (automatic updates)
- **Both:** OPA gates Renovate-generated PRs to ensure they pass policy

### Why Provider Protocols (Not Concrete Classes)?

- **Protocols** = loose coupling (adapters are plug-and-play)
- **Concrete** = tight coupling (hard to swap providers)
- **Python 3.10+** has `Protocol` with runtime checking

---

## Next Steps (After Week 8)

1. **Publish architecture** to org landing page
2. **Train team** on paved road (optional but recommended)
3. **Quarterly hygiene:** Run compliance audit, triage gaps
4. **Evolve:** Add new tools/policies incrementally via standards PRs

---

## Quick Links

- **[inventory.json](inventory.json)** - Repo metadata
- **[gaps.md](gaps.md)** - What's missing per repo
- **[actions.md](actions.md)** - P0/P1/P2 fixes with patches
- **[features.md](features.md)** - Capabilities and cross-links
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete starter code
- **[standards/](standards/)** - SSOT policies and conventions
- **[.github/](github)** - Reusable workflows and health checks

---

## Questions?

1. **Which repo should I start with?** → repz (highest impact, most CI patterns)
2. **How long per repo?** → 30 min for retrofit (add 4 files + update CI)
3. **Can I do this incrementally?** → Yes; each repo is independent
4. **What if a repo is dead?** → Move to archive/ or add to EXCEPTIONS.md
5. **Do I need to retrain devs?** → No; paved road is self-documenting via CI

---

## Summary

You have everything needed to:

1. **Week 1-2:** Build 13 foundation repos with production code
2. **Week 3-8:** Retrofit 35 existing repos to Golden Path
3. **Week 9+:** Maintain via automated compliance checks

**Result:** 100% repos with required files, consistent CI/CD, policy-as-code enforcement, and a paved road that scales.

Let's execute. 🚀
