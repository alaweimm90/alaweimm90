# Master Execution Plan — Complete Portfolio Governance Stack

**Scope:** Transform 35 repos across 6 organizations into a compliant, policy-enforced ecosystem
**Timeline:** 8 weeks (or 2 weeks for quick-win)
**Status:** ✅ All components ready

---

## Stack Overview (4 Layers)

### Layer 1: Foundation (Central Governance)
```
alaweimm90/
├─ .github/                    # Reusable workflows + health files
├─ standards/                  # SSOT (OPA, naming, docs)
├─ core-control-center/        # DAG orchestrator (vendor-neutral)
├─ adapter-claude/             # LLM integrations
├─ adapter-openai/
├─ adapter-lammps/             # Solver integrations
├─ adapter-siesta/
├─ template-python-lib/        # Golden starters
├─ template-ts-lib/
├─ template-research/
├─ template-monorepo/
├─ infra-actions/              # Shared CI tooling
└─ infra-containers/           # Base images
```

### Layer 2: Organization-Level Governance (Per-Org)
```
<ORG>/.github/
├─ profile/README.md           # Org page + landing
├─ workflows/
│  ├─ reusable-python-ci.yml
│  ├─ reusable-ts-ci.yml
│  ├─ reusable-policy.yml
│  └─ reusable-release.yml
├─ policy/opa/                 # Org-specific rules
├─ CODE_OF_CONDUCT.md
├─ SECURITY.md
├─ labels.json
└─ dependabot.yml
```

### Layer 3: Repository Metadata (Per-Repo)
```
<REPO>/
├─ .meta/repo.yaml             # Type, tier, docs profile
├─ .github/CODEOWNERS          # Ownership clarity
├─ .github/workflows/ci.yml    # Calls org reusable
├─ .github/workflows/policy.yml
├─ tests/                      # ≥80% libs/tools, ≥70% research
└─ src/ or packages/
```

### Layer 4: Portfolio Catalog (Discovery & Status)
```
catalog/
├─ CATALOG_SUMMARY.md
├─ catalog_alaweimm90-business.md
├─ catalog_alaweimm90-science.md
├─ catalog_alaweimm90-tools.md
├─ catalog_AlaweinOS.md
└─ catalog_MeatheadPhysicist.md
```

---

## Phase-by-Phase Timeline

### Phase 0: Foundation Setup (Days 1-5)
**Goal:** Create central governance layer + org catalogs

**Day 1-2: Run Census + Generate Catalogs**
```bash
# Export org list
export GH_ORGS="alaweimm90 AlaweinOS alaweimm90-science alaweimm90-business alaweimm90-tools MeatheadPhysicist"
export GH_USER="alaweimm90"

# Run census sweep (auto-generates per-org catalogs)
bash census.sh 2>&1 | tee census.log

# Review outputs
cat inventory.json          # Machine-readable
cat CATALOG_SUMMARY.md      # Human-readable
ls catalog_*.md             # Per-org overviews
```

**Expected Output:**
- `inventory.json` with all repos cataloged
- Per-org `.md` files showing status, archival, CI flags
- `projects_without_repo` list (orphans to promote)

**Day 3: Promote Orphans**
```bash
# Interactive promotion (Projects → Repos)
bash census-promote.sh

# Verify
bash census.sh              # Re-run to confirm orphans → 0
```

**Day 4-5: Bootstrap Foundation Layer**
```bash
# Preview
bash bootstrap.sh --dry-run

# Execute (creates 14 repos)
bash bootstrap.sh

# Verify
gh repo list alaweimm90 --limit 50 | grep -E "^alaweimm90/(core-|lib-|adapter-|template-|infra-)" | wc -l
# Should show: 14
```

**Checkpoint:**
- [ ] inventory.json complete
- [ ] Per-org catalogs generated
- [ ] Orphans promoted (or documented in EXCEPTIONS.md)
- [ ] 14 foundation repos created
- [ ] All new repos have passing CI

---

### Phase 1: Org-Level Governance (Days 6-10)
**Goal:** Create `.github` repos for each org + define working agreements

**For each organization (in parallel):**

```bash
ORG="alaweimm90"  # repeat for: AlaweinOS, alaweimm90-science, alaweimm90-business, alaweimm90-tools, MeatheadPhysicist

# Create org .github repo
gh repo create "$ORG/.github" --public --confirm

# Clone and populate
git clone https://github.com/$ORG/.github
cd .github

# Copy structure from reference (see section below)
# - profile/README.md (org landing page)
# - workflows/*.yml (reusable: python-ci, ts-ci, policy, release)
# - policy/opa/*.rego (org-specific rules)
# - CODE_OF_CONDUCT.md
# - SECURITY.md
# - labels.json
# - dependabot.yml

git add .
git commit -m "chore: initialize org-level governance"
git push -u origin main

# Enable branch protection on main
gh api repos/$ORG/.github/branches/main/protection \
  -X PUT \
  -f required_status_checks.strict=true \
  -f required_pull_request_reviews.dismiss_stale_reviews=true \
  -f required_pull_request_reviews.require_code_owner_reviews=true
```

**Expected Output:**
- 6 new `.github` repos (one per org)
- Each with reusable workflows visible in GitHub UI
- Each with org-level profile/README on org page

**Checkpoint:**
- [ ] `.github` repo created for each org
- [ ] Reusable workflows tested (at least one successful call)
- [ ] Org profile/README visible on GitHub org page
- [ ] Branch protection enabled on main
- [ ] Labels synced across org repos

---

### Phase 2: Priority Repos Retrofit (Weeks 2-3)
**Goal:** Retrofit 5-10 high-impact repos to use org reusable workflows

**Target repos (from gaps.md P0 list):**
1. repz (highest CI complexity)
2. live-it-iconic (e-commerce platform)
3. optilibria (optimization + research)
4. AlaweinOS (workspace framework)
5. mag-logic (library, needs test coverage)

**Per repo:**

```bash
REPO="repz"
ORG="alaweimm90-business"

# Clone
git clone https://github.com/$ORG/$REPO
cd $REPO

# Add .meta/repo.yaml if missing
mkdir -p .meta
cat > .meta/repo.yaml <<'EOF'
type: platform
language: mixed
docs_profile: standard
criticality_tier: 1
owner: "@alaweimm90"
EOF

# Add CODEOWNERS if missing
mkdir -p .github
cat > .github/CODEOWNERS <<'EOF'
* @alaweimm90
.meta/ @alaweimm90
.github/ @alaweimm90
EOF

# Update CI to call reusable workflow
# (See template in GITHUB_OS_MERGED.md section 9)
cat > .github/workflows/ci.yml <<'EOF'
name: ci
on: [push, pull_request]
jobs:
  call-reusable:
    uses: $ORG/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'
  policy:
    uses: $ORG/.github/.github/workflows/reusable-policy.yml@main
EOF

# Commit and push
git add .meta/repo.yaml .github/CODEOWNERS .github/workflows/ci.yml
git commit -m "chore: retrofit to org governance

- Add .meta/repo.yaml (metadata contract)
- Add CODEOWNERS (ownership clarity)
- Update CI to call org reusable workflows
- Enable policy gate (required files + OPA)
"
git push origin main
```

**Expected Output:**
- Each repo has `.meta/repo.yaml`
- Each repo has CODEOWNERS
- Each repo CI calls org reusable workflows
- All pass policy gates (required files check)
- Coverage enforced (≥80% libs/tools)

**Checkpoint:**
- [ ] 5-10 repos retrofitted
- [ ] All new CI runs pass
- [ ] Coverage gates pass (or documented exceptions)
- [ ] OPA policy gates pass
- [ ] No breaking changes to existing CI

---

### Phase 3: Bulk Retrofit (Weeks 4-8)
**Goal:** Add governance files to remaining 25+ repos

**Bulk operation script:**

```bash
# For each org
for ORG in $GH_ORGS; do
  # Get all repos
  REPOS=$(gh repo list "$ORG" -L 200 --json name -q '.[].name')

  for REPO in $REPOS; do
    # Skip archived
    gh api "repos/$ORG/$REPO" --jq '.archived' | grep -q 'true' && continue

    # Check if already has .meta/repo.yaml
    gh api "repos/$ORG/$REPO/contents/.meta/repo.yaml" >/dev/null 2>&1 && continue

    # Clone, add files, push
    echo "Retrofitting $ORG/$REPO"

    git clone "https://github.com/$ORG/$REPO" "/tmp/$REPO" 2>/dev/null || continue
    cd "/tmp/$REPO"

    # Add missing files
    mkdir -p .meta .github

    # .meta/repo.yaml (infer type from repo name or README)
    cat > .meta/repo.yaml <<'EOF'
type: library
language: python
docs_profile: minimal
criticality_tier: 2
owner: "@alaweimm90"
EOF

    # CODEOWNERS
    cat > .github/CODEOWNERS <<'EOF'
* @alaweimm90
EOF

    # CI (if not present)
    if [ ! -f ".github/workflows/ci.yml" ]; then
      cat > .github/workflows/ci.yml <<'EOF'
name: ci
on: [push, pull_request]
jobs:
  call:
    uses: $ORG/.github/.github/workflows/reusable-python-ci.yml@main
  policy:
    uses: $ORG/.github/.github/workflows/reusable-policy.yml@main
EOF
    fi

    # Commit and push (only if changes)
    if [ -n "$(git status --short)" ]; then
      git add .meta/repo.yaml .github/CODEOWNERS .github/workflows/ci.yml
      git commit -m "chore: add governance metadata"
      git push origin main
      echo "✓ Pushed"
    fi

    cd - >/dev/null
    rm -rf "/tmp/$REPO"
  done
done
```

**Expected Output:**
- All active repos have `.meta/repo.yaml`
- All active repos have CODEOWNERS
- All active repos call org reusable workflows
- Coverage enforced across portfolio

**Checkpoint:**
- [ ] 35+ repos have `.meta/repo.yaml`
- [ ] 35+ repos have CODEOWNERS
- [ ] 35+ repos have updated CI
- [ ] Zero repos with broken CI
- [ ] OPA policies passing on all new PRs

---

### Phase 4: Ongoing Enforcement (Weeks 9+)
**Goal:** Automate compliance checks + monthly census

**Monthly census (scheduled workflow):**

```yaml
# .github/workflows/monthly-census.yml
name: Monthly Census

on:
  schedule:
    - cron: '0 2 * * 1'  # Monday 2am UTC
  workflow_dispatch:

jobs:
  census:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run census
        run: bash census.sh

      - name: Check for changes
        id: check
        run: |
          if git diff --quiet inventory.json; then
            echo "changed=false" >> $GITHUB_OUTPUT
          else
            echo "changed=true" >> $GITHUB_OUTPUT
          fi

      - name: Create PR if changes
        if: steps.check.outputs.changed == 'true'
        run: |
          git config user.name "portfolio-census"
          git add inventory.json
          git commit -m "chore(census): monthly inventory update"
          gh pr create --title "Portfolio Census: Monthly Update" \
            --body "Automated monthly census run"

      - name: Report orphans
        run: |
          ORPHANS=$(jq '.projects_without_repo | length' inventory.json)
          if [ "$ORPHANS" -gt 0 ]; then
            gh issue create --title "Portfolio Census: Orphan Projects Detected" \
              --body "Run: bash census-promote.sh to remediate"
          fi
```

**Expected Output:**
- Monthly PR with inventory updates
- Issues auto-filed if orphans detected
- Portfolio compliance tracked over time

---

## Quick-Reference: File Locations

### Foundation Layer (alaweimm90 org)

| Component | Path | Created By |
|-----------|------|-----------|
| Reusable CI | alaweimm90/.github/.github/workflows/ | bootstrap.sh |
| OPA policies | alaweimm90/standards/policy/ | GITHUB_OS_MERGED.md §7 |
| Orchestrator | alaweimm90/core-control-center/src/ | GITHUB_OS_MERGED.md §3 |
| Adapters | alaweimm90/adapter-{claude,openai,lammps,siesta}/ | bootstrap.sh |
| Templates | alaweimm90/template-{python-lib,ts-lib,research,monorepo}/ | bootstrap.sh |

### Organization-Level (.github repo per org)

| Component | Path | Manual |
|-----------|------|--------|
| Org profile README | <ORG>/.github/profile/README.md | See reference below |
| Reusable workflows | <ORG>/.github/.github/workflows/ | Copy from foundation or create org-specific |
| Community health | <ORG>/.github/CODE_OF_CONDUCT.md | Template provided |
| Labels | <ORG>/.github/labels.json | Template provided |

### Per-Repo Enforced

| File | Purpose | Required |
|------|---------|----------|
| .meta/repo.yaml | Type, tier, docs profile | ✅ All |
| .github/CODEOWNERS | Ownership | ✅ All |
| .github/workflows/ci.yml | Calls org reusable | ✅ Active |
| .github/workflows/policy.yml | OPA + lint | ✅ Active |
| tests/ | ≥80% libs/tools | ✅ libs/tools only |

---

## Org-Specific Profile README Template

Create this for each org:

```markdown
# <ORG NAME> — GitHub Organization

**Mission:** <one-liner>

**Structure:** Prefix taxonomy: core-, lib-, adapter-, tool-, template-, demo-, infra-, paper-

**Catalog:** [Auto-generated catalog](./catalog_<org>.md)

## Quick Links

- [Standards](https://github.com/alaweimm90/standards) — Naming, OPA policies, AI specs
- [Core Control Center](https://github.com/alaweimm90/core-control-center) — DAG orchestrator
- [Adapters](https://github.com/alaweimm90?q=adapter-) — Provider integrations

## Working Agreements

1. **Typed & Tested:** All code is strictly typed; libraries/tools ≥80% test coverage
2. **Reproducible:** Outputs under `outputs/YYYYMMDD-HHMMSS/`
3. **CI/CD:** PRs require policy + CI to pass; branch protection enforced
4. **Governance:** Repos follow prefix taxonomy; `.meta/repo.yaml` is SSOT

## Getting Started

1. Clone a golden template: `template-python-lib`, `template-ts-lib`, or `template-research`
2. Read [NAMING.md](https://github.com/alaweimm90/standards/blob/main/NAMING.md) for conventions
3. Update `.meta/repo.yaml` with your repo metadata
4. Wire CI to call org reusable workflows (see `.github/workflows/` in this repo)
5. Push and watch policy + coverage gates enforce compliance

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md)

## Support

- Questions: [@alaweimm90](https://github.com/alaweimm90)
- Issues: File on the relevant repo
- Security: See [SECURITY.md](./SECURITY.md)
```

---

## Validation Checklist (Post-Implementation)

### Foundation Layer
- [ ] 14 foundation repos created
- [ ] .github repo exists + reusable workflows callable
- [ ] standards repo exists + OPA policies present
- [ ] core-control-center exists + tests pass
- [ ] All 4 adapters exist + tests pass
- [ ] All 4 templates exist + ready to clone

### Org-Level
- [ ] `.github` repo created for each org (6 total)
- [ ] Org profile README visible on org page
- [ ] Reusable workflows tested (at least one call succeeded)
- [ ] Labels synced to all repos
- [ ] dependabot.yml active (weekly updates)

### Per-Repo
- [ ] 100% of active repos have `.meta/repo.yaml`
- [ ] 100% of active repos have CODEOWNERS
- [ ] 100% of active repos have updated ci.yml
- [ ] 100% of repos pass policy gate on new PRs
- [ ] Libraries/tools show ≥80% coverage
- [ ] Zero orphan projects (or documented in EXCEPTIONS.md)

### Portfolio-Wide
- [ ] inventory.json tracks all repos
- [ ] Monthly census runs automatically
- [ ] OPA policies enforced in CI
- [ ] Compliance trending upward
- [ ] Team trained on golden path

---

## Success Metrics

| Metric | Baseline | Target | Timeline |
|--------|----------|--------|----------|
| Repos with `.meta/repo.yaml` | 1/35 | 35/35 | Week 8 |
| Repos with CODEOWNERS | 1/35 | 35/35 | Week 8 |
| Repos calling reusable CI | 0/22 | 22/22 | Week 6 |
| Coverage ≥80% (libs/tools) | 3/11 | 11/11 | Week 8 |
| Portfolio compliance | 55.7% | 100% | Week 8 |
| Orphan projects | 3-7 | 0 | Week 1 |

---

## If Generating Org-Level `.github` Repos...

I can auto-generate the following for each org (6 repos):

**Per-org `.github` skeleton includes:**
- `profile/README.md` (org landing page)
- `.github/workflows/reusable-python-ci.yml` (or org-specific variant)
- `.github/workflows/reusable-ts-ci.yml`
- `.github/workflows/reusable-policy.yml`
- `.github/workflows/reusable-release.yml` (semver)
- `policy/opa/` (org-specific governance rules)
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `CONTRIBUTING.md`
- `labels.json` (pre-populated with standard labels)
- `dependabot.yml` (weekly pip/npm/docker updates)
- `.gitignore`

**Request format:**

```bash
# I can generate tarballs or individual repos ready to push:
# - catalog_<org>_github_skeleton.tar.gz (contains above structure)
# - One per org: alaweimm90, AlaweinOS, alaweimm90-science, alaweimm90-business, alaweimm90-tools, MeatheadPhysicist

# Each includes:
# - Pre-filled README (org-specific)
# - Copy-paste reusable workflows (ready to test)
# - Consistent label set (cross-org, searchable)
# - Ready-to-commit file structure

# Usage: untar, review, commit, push, wire up org settings
```

---

## Next Actions

1. **This Week:**
   - [ ] Read this plan
   - [ ] Run `bash census.sh`
   - [ ] Review `CATALOG_SUMMARY.md` + per-org catalogs
   - [ ] Promote orphans with `bash census-promote.sh`

2. **Next Week:**
   - [ ] Run `bash bootstrap.sh` (create 14 foundation repos)
   - [ ] Populate from GITHUB_OS_MERGED.md sections 3-6
   - [ ] Test reusable workflows

3. **Weeks 2-3:**
   - [ ] Create `.github` repo for each org
   - [ ] Retrofit 5-10 priority repos

4. **Weeks 4-8:**
   - [ ] Bulk retrofit remaining 25+ repos
   - [ ] Track compliance trending

5. **Week 9+:**
   - [ ] Monthly census automation
   - [ ] Quarterly governance review
   - [ ] Continuous improvement cycle

---

**Status: 🟢 Ready to Execute**

All components in place. Choose:
- **Full Path:** 8 weeks, 100% compliance
- **Quick-Win:** 2 weeks, 60% benefit
- **Census Only:** 2 days, complete inventory

**Start:** `bash census.sh`

