# Complete Deployment Guide: alaweimm90 Golden Path

**Scope:** Implement production-ready GitHub OS for 55 repositories across 5 organizations
**Timeline:** 10 business days
**Team Size:** 1-3 people
**Success Criteria:** All 55 repos with `.meta/repo.yaml`, CODEOWNERS, reusable CI, and policy enforcement

---

## Overview

This guide deploys the complete architecture described in the bootstrap document:
- **Phase 1 (Days 1-2):** Foundation repos (`.github`, `standards`, `core-control-center`)
- **Phase 2 (Days 3-5):** Templates and adapters
- **Phase 3 (Days 6-10):** Migrate 55 existing repos and archive stale ones

---

## Phase 1: Foundation (Days 1-2)

### Pre-Flight Checklist

```bash
# Verify you have required tools
gh --version          # GitHub CLI
git --version         # Git 2.30+
python3 --version     # Python 3.10+
pip install pyyaml    # For YAML parsing in migration script
```

### Day 1: Deploy `.github` Repository

#### Step 1: Create Empty Repo

```bash
# Via GitHub web UI or:
gh repo create alaweimm90/.github \
  --public \
  --description "Org-wide governance and reusable workflows" \
  --confirm
```

#### Step 2: Clone and Initialize

```bash
cd ~/repos/alaweimm90
git clone https://github.com/alaweimm90/.github.git
cd .github

git config user.email "ops@alaweimm90.dev"
git config user.name "alaweimm90 bot"
```

#### Step 3: Copy All Files

Create all files listed in `BOOTSTRAP.md` under ".github repo" section:
- `.github/workflows/reusable-python-ci.yml`
- `.github/workflows/reusable-ts-ci.yml`
- `.github/workflows/reusable-policy.yml`
- `.github/workflows/reusable-release.yml`
- `.github/ISSUE_TEMPLATE/{bug.yml,feature.yml}`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/CODEOWNERS`
- `.github/labels.json`
- `.github/dependabot.yml`
- `README.md`
- `LICENSE`
- `SECURITY.md`
- `CONTRIBUTING.md`
- `.meta/repo.yaml`

**Quick Copy (from local files)**

```bash
# Create directories
mkdir -p .github/workflows .github/ISSUE_TEMPLATE

# Copy files (assume they're provided in text form)
# [Paste each file from BOOTSTRAP.md]
```

#### Step 4: Commit & Push

```bash
git add .
git commit -m "Initial: reusable workflows, issue templates, org governance

- Add reusable Python CI (test, lint, type-check, coverage)
- Add reusable TypeScript CI (test, lint, coverage)
- Add policy enforcement workflow (OPA, Markdown linting)
- Add release automation
- Add issue templates (bug, feature, question)
- Add labels and Dependabot config
- Add SECURITY, CONTRIBUTING, and governance docs

Co-authored-by: alaweimm90 <ops@alaweimm90.dev>"

git branch -M main
git remote add origin https://github.com/alaweimm90/.github.git
git push -u origin main
```

#### Step 5: Configure Branch Protection

```bash
# Via GitHub CLI
gh repo edit alaweimm90/.github \
  --enable-discussions \
  --enable-security-advisories

# Via web UI: Settings → Branches → Add Rule
# - Branch name pattern: main
# - Require status checks: ci, policy
# - Require code owner reviews: yes
# - Require up-to-date branches: yes
```

#### Step 6: Enable and Configure Dependabot

```bash
# Via web UI: Settings → Code security & analysis
# Enable:
# - Dependabot alerts
# - Dependabot security updates
# - Secret scanning
```

**Verification (Day 1 end)**

```bash
# Check workflows are valid
gh workflow list -R alaweimm90/.github
# Should show: python-ci, ts-ci, policy, release (ACTIVE)

# Verify labels are created (when first workflow runs)
gh label list -R alaweimm90/.github
```

---

### Day 2: Create `standards` and `core-control-center`

Repeat the same process for:
1. **`alaweimm90/standards`** — Policy documents, OPA rules, linter configs
2. **`alaweimm90/core-control-center`** — DAG orchestrator
3. **`alaweimm90/alaweimm90`** — Profile landing page

**File Structure for `standards/`** (from `STANDARDS_REPO.md`):

```
standards/
├─ docs/
│  ├─ AI-SPECS.md
│  ├─ NAMING.md
│  ├─ DOCS_GUIDE.md
│  ├─ TIER_DEFINITIONS.md
│  └─ GOVERNANCE.md
├─ opa/
│  ├─ repo_structure.rego
│  ├─ docs_policy.rego
│  └─ workflows_policy.rego
├─ linters/
│  ├─ markdownlint.yaml
│  ├─ ruff.toml
│  ├─ black.toml
│  └─ .eslintrc.cjs
├─ templates/
│  ├─ .meta/repo.yaml
│  ├─ README.md
│  ├─ CONTRIBUTING.md
│  └─ SECURITY.md
├─ README.md
├─ LICENSE
├─ .meta/repo.yaml
└─ .github/workflows/ci.yml
```

**File Structure for `core-control-center/`**:

```
core-control-center/
├─ pyproject.toml
├─ src/control_center/
│  ├─ __init__.py
│  ├─ engine/{orchestrator.py, node.py}
│  ├─ providers/base.py
│  ├─ agents/base.py
│  └─ tools/base.py
├─ tests/test_orchestrator.py
├─ README.md
├─ LICENSE
├─ .meta/repo.yaml
├─ SECURITY.md
├─ CONTRIBUTING.md
└─ .github/workflows/{ci.yml,policy.yml}
```

**ci.yml for both repos:**

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  python:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

**Commit & Push (both repos)**

```bash
git add .
git commit -m "Initial: standards and governance policies"
git branch -M main
git remote add origin https://github.com/alaweimm90/<repo-name>.git
git push -u origin main
```

---

## Phase 2: Templates & Adapters (Days 3-5)

### Create Four Golden Templates

#### `template-python-lib/`

```
template-python-lib/
├─ pyproject.toml              # Pre-configured for coverage gate
├─ src/pkgname/__init__.py      # Minimal package init
├─ tests/test_smoke.py          # One passing test
├─ .meta/repo.yaml              # Pre-filled template
├─ README.md                     # Includes instruction to rename
├─ LICENSE
├─ CONTRIBUTING.md
├─ SECURITY.md
├─ .github/workflows/ci.yml      # Calls reusable-python-ci
├─ .github/workflows/policy.yml  # Calls reusable-policy
├─ .github/CODEOWNERS
├─ mypy.ini                      # Type checking config
├─ ruff.toml                     # Linting config
└─ .pre-commit-config.yaml       # Local development hooks
```

**Key content in `pyproject.toml`:**

```toml
[project]
name = "pkgname"  # TODO: rename
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []
[project.optional-dependencies]
dev = ["pytest","pytest-cov>=5","mypy","ruff","black"]

[tool.pytest.ini_options]
addopts = "-q --cov=pkgname --cov-report=term-missing --cov-fail-under=80"
```

**Usage:**

```bash
# Clone and rename
git clone https://github.com/alaweimm90/template-python-lib.git my-new-lib
cd my-new-lib

# Update project name in 3 places
sed -i 's/pkgname/my_new_lib/g' pyproject.toml src/pkgname/__init__.py tests/test_smoke.py
mv src/pkgname src/my_new_lib

# Push
git branch -M main
git remote set-url origin https://github.com/alaweimm90/lib-my-new-lib.git
git push -u origin main
```

#### `template-ts-lib/`

Similar structure but with:
- `package.json` (pre-configured for coverage)
- `src/index.ts`
- `tsconfig.json`
- `vitest.config.ts`
- `.eslintrc.cjs`

#### `template-research/`

```
template-research/
├─ uv.lock              # or conda env.yaml
├─ notebooks/
│  └─ 00_intro.ipynb    # Smoke test notebook
├─ src/                 # Python code under test
├─ data/                # Raw data (use LFS)
├─ results/
├─ papers/              # LaTeX/PDF outputs
└─ Makefile             # Common tasks (run, test, build)
```

#### `template-monorepo/`

```
template-monorepo/
├─ turbo.json                    # (if JS)
├─ pnpm-workspace.yaml           # or uv workspace in pyproject.toml
├─ apps/                         # Applications
│  └─ app1/
├─ packages/                     # Shared packages
│  └─ lib1/
├─ .meta/repo.yaml
└─ .github/workflows/
```

---

### Create Four Adapter Repos

#### `adapter-claude/`

```
adapter-claude/
├─ pyproject.toml
├─ src/adapter_claude/provider.py
├─ tests/test_provider.py
├─ README.md                      # Example usage with control-center
├─ LICENSE
├─ .meta/repo.yaml
└─ .github/workflows/ci.yml
```

**`src/adapter_claude/provider.py`** (from architecture document)

#### `adapter-openai/`

Same structure, implementing OpenAI API.

#### `adapter-lammps/`, `adapter-siesta/`

Tools adapters implementing `Tool` protocol.

---

## Phase 3: Migrate 55 Existing Repos (Days 6-10)

### Step 1: Prepare Migration

```bash
cd ~/repos/alaweimm90

# Copy inventory.json and migration script
cp inventory.json .
cp MIGRATION_SCRIPT.py .

# Install dependencies
pip install pyyaml
```

### Step 2: Run Migration Script (Dry Run)

```bash
python3 MIGRATION_SCRIPT.py
```

Output will show:
- Which repos are being updated
- What files are being created
- Summary of changes

**Review:** `migration-results.json` for details

### Step 3: Commit Changes in Each Repo

```bash
#!/bin/bash
# migration-commit.sh

for org in alaweimm90-business alaweimm90-science alaweimm90-tools AlaweinOS MeatheadPhysicist; do
  echo "Processing $org..."
  cd organizations/$org

  for repo in */; do
    repo_name="${repo%/}"
    echo "  → $repo_name"

    cd "$repo_name"

    # Check if there are changes
    if git status --porcelain | grep -q .; then
      git add .
      git commit -m "chore: golden path compliance

- Add .meta/repo.yaml for governance
- Add .github/CODEOWNERS for ownership
- Update CI to use reusable workflows
- Add policy checks

See: https://github.com/alaweimm90/standards"

      git push origin main
    fi

    cd ..
  done

  cd ../..
done
```

**Run it:**

```bash
chmod +x migration-commit.sh
./migration-commit.sh 2>&1 | tee migration.log
```

### Step 4: Verify All Repos

```bash
#!/bin/bash
# verify-migration.sh

for org in alaweimm90-business alaweimm90-science alaweimm90-tools AlaweinOS MeatheadPhysicist; do
  echo "Verifying $org..."
  for repo in organizations/$org/*/; do
    repo_name=$(basename "$repo")
    [[ -f "$repo/.meta/repo.yaml" ]] && echo "  ✓ $repo_name" || echo "  ✗ $repo_name"
  done
done
```

### Step 5: Monitor CI Runs

```bash
# Watch CI status across all repos
watch -n 30 'gh run list -R alaweimm90/core-control-center --limit 5'
```

**Check workflow syntax:**

```bash
# This will catch YAML errors early
for org in alaweimm90-business alaweimm90-science alaweimm90-tools AlaweinOS MeatheadPhysicist; do
  for repo in organizations/$org/*/; do
    repo_name=$(basename "$repo")
    if [[ -f "$repo/.github/workflows/ci.yml" ]]; then
      echo "Validating $org/$repo_name..."
      gh workflow list -R "alaweimm90/$repo_name" 2>/dev/null || echo "  ⚠️  Not yet pushed"
    fi
  done
done
```

### Step 6: Archive Stale Repos

Based on the gap analysis, 10-15 repos should be archived:

```bash
# Move to archive with tag
for repo in calla-lily-couture dr-alowein-portfolio api education mlops notebooks visualizations; do
  echo "Archiving $repo..."
  # Option 1: Via GitHub web UI → Settings → Archive this repository
  # Option 2: Via GH CLI (requires owner access)
  # gh repo edit alaweimm90/$repo --archived
done
```

### Step 7: Configure Branch Protection Org-Wide

```bash
#!/bin/bash
# Set branch protection for all repos

REPOS=(
  "core-control-center"
  "standards"
  # ... add others as prioritized
)

for repo in "${REPOS[@]}"; do
  echo "Configuring $repo..."

  # Via GitHub UI:
  # Settings → Branches → Add rule
  # Pattern: main
  # Require: ci, policy
  # Code owners: required
  # Update before merge: required
done
```

Or use a more sophisticated approach with GitHub GraphQL.

### Step 8: Enable Org-Level Secrets

```bash
# Store shared secrets (if any)
gh secret set CODECOV_TOKEN --org alaweimm90 --value "your-token"
gh secret set PYPI_TOKEN --org alaweimm90 --value "your-token"
```

---

## Validation & Success Metrics

### Metrics Dashboard

Track these metrics over the 10-day period:

```bash
#!/bin/bash
# metrics.sh

echo "=== Golden Path Compliance Metrics ==="
echo ""

# Count repos with .meta/repo.yaml
meta_count=$(find organizations -name "repo.yaml" | wc -l)
echo "✓ Repos with .meta/repo.yaml: $meta_count / 55"

# Count repos with CODEOWNERS
owners_count=$(find organizations -name "CODEOWNERS" | wc -l)
echo "✓ Repos with CODEOWNERS: $owners_count / 55"

# Count repos calling reusable CI
ci_count=$(grep -r "uses: alaweimm90/.github/.github/workflows/reusable" organizations | wc -l)
echo "✓ Repos calling reusable CI: $ci_count / 55"

# Average CI duration (requires API access)
echo "✓ Average CI duration: [requires API]"

# Coverage stats (from codecov or local)
echo "✓ Repos at ≥80% coverage: [requires API]"

echo ""
echo "=== Day-by-Day Checklist ==="
```

### Day-by-Day Success Criteria

**Day 1 End:**
- [ ] `.github` repo created with workflows, templates, and docs
- [ ] All workflows pass validation
- [ ] Branch protection enabled on `main`

**Day 2 End:**
- [ ] `standards` repo live with policies and linter configs
- [ ] `core-control-center` bootstrapped with DAG engine
- [ ] Profile repo (`alaweimm90/alaweimm90`) pinning core repos

**Day 5 End:**
- [ ] All 4 templates created and tested
- [ ] All 4 adapters bootstrapped (at minimum, `adapter-claude` + `adapter-openai`)

**Day 10 End:**
- [ ] 55 existing repos have `.meta/repo.yaml` + `CODEOWNERS`
- [ ] 55 existing repos calling reusable CI (no local duplication)
- [ ] 10-15 stale repos archived
- [ ] Top 5 libraries passing coverage gate (≥80%)
- [ ] CI running successfully across portfolio
- [ ] Documentation updated and policy enforced

---

## Troubleshooting

### Workflow Validation Failures

```bash
# Check workflow syntax locally
act -j ci --dry-run  # Requires 'act' tool

# Or validate YAML
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

### Python/TypeScript CI Failures

**Python:**
```bash
# Run locally to debug
pip install -e ".[dev]"
ruff check .
black --check .
mypy .
pytest --cov
```

**TypeScript:**
```bash
# Run locally
pnpm install
pnpm lint
pnpm test -- --run --coverage
```

### `.meta/repo.yaml` Schema Errors

```bash
# Validate schema
python3 << 'EOF'
import yaml
with open('.meta/repo.yaml') as f:
    meta = yaml.safe_load(f)
    assert 'type' in meta
    assert 'languages' in meta or meta['type'] == 'meta'
    print("✓ Valid")
EOF
```

### Migration Script Issues

```bash
# Re-run with verbose output
python3 -c "
import MIGRATION_SCRIPT
# ... adjust as needed
"

# Check individual repo
ls -la organizations/alaweimm90-science/qmat-sim/.meta/
ls -la organizations/alaweimm90-science/qmat-sim/.github/
```

---

## Post-Rollout Maintenance

### Weekly Hygiene

```bash
# Check repo compliance
python3 MIGRATION_SCRIPT.py > compliance-report.txt

# Review policy violations
gh search issues --org alaweimm90 --label "policy-violation" --state open

# Monitor CI health
gh run list --org alaweimm90 --status failure --limit 20
```

### Monthly Reviews

- [ ] Update linter configs in `standards/`
- [ ] Bump Python/Node.js versions if needed
- [ ] Review dependency security advisories
- [ ] Adjust coverage gates based on trends
- [ ] Onboard new repos with templates

### Quarterly Policy Reviews

- [ ] Update AI-SPECS in `standards/`
- [ ] Review and refine OPA rules
- [ ] Archive repos with <6 months inactivity
- [ ] Promote libraries to higher tiers if mature

---

## Quick Reference

### Common Commands

```bash
# Status of all repos
gh repo list alaweimm90 --limit 100

# List workflows
gh workflow list -R alaweimm90/core-control-center

# Trigger workflow
gh workflow run ci.yml -R alaweimm90/lib-quantum

# View logs
gh run view <run-id> -R alaweimm90/lib-quantum --log

# Create release
gh release create v1.0.0 -R alaweimm90/lib-quantum --generate-notes

# Sync fork (if needed)
git pull upstream main --rebase
git push origin main
```

### Repository Template Usage

```bash
# Create a new Python lib
gh repo create alaweimm90/lib-mylib --template alaweimm90/template-python-lib

# Or manually
git clone https://github.com/alaweimm90/template-python-lib.git lib-mylib
cd lib-mylib
# Update pyproject.toml, .meta/repo.yaml, README
git remote set-url origin https://github.com/alaweimm90/lib-mylib.git
git push -u origin main
```

---

## Support & Escalation

**Questions or blockers?**

1. Check `standards/docs/GOVERNANCE.md` for decision process
2. Post to org discussions (GitHub)
3. Email: ops@alaweimm90.dev

**Critical issues:**

1. Document in issue (include: repo, error, steps to reproduce)
2. Tag `@alaweimm90` and label `policy-violation` or `coverage-gap`
3. Roll back if needed, then post-mortem

---

## Final Notes

This 10-day rollout directly addresses the gaps from your audit:

| Gap | Solution |
|-----|----------|
| Missing LICENSE (62%) | Added to all via templates + migration |
| Missing SECURITY (76%) | Template provided, enforced by policy |
| Missing CI/CD (33%) | Reusable workflows eliminate local duplication |
| Zero `.meta/repo.yaml` (100%) | Generated and enforced by OPA policy |
| No CODEOWNERS (100%) | Added via migration script |
| 60% missing tests | Templates gate coverage at ≥80% (libs/tools) |
| Stale docs | Profile + standards repo as SSOT |
| Tool sprawl | Core + adapters architecture |

**Success looks like:**
- All 55 repos compliant on Day 10
- Zero "policy-violation" labels on `main`
- New repos automatically compliant via templates
- Governance via code (OPA/conftest), not spreadsheets
- <5% CI flake rate
- ≥80% coverage on core libraries

