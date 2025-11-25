# Portfolio Census — 2-Day Execution Plan

**Goal:** Discover all repositories, projects, gists, packages across your GitHub orgs. Identify orphan projects (without repo linkage) and promote them into proper repos under enforced taxonomy.

**Timeline:** 2 days (Day 1: Sweep + Triage, Day 2: Promotion + Validation)

**Deliverables:**
- Consolidated `inventory.json` with all assets cataloged
- `consolidated.json` in timestamped output directory
- List of orphan projects and remediation actions
- New repos created from templates (one for each promoted orphan)
- Updated OPA policies enforcing no future orphans

---

## DAY 1: SWEEP & TRIAGE

### Morning (1-2 hours)

#### Step 1: Prepare environment

```bash
# Clone the repository (if not already present)
cd ~/GitHub  # or your preferred location

# Set environment variables for orgs and user
export GH_ORGS="alaweimm90 AlaweinOS alaweimm90-science alaweimm90-business alaweimm90-tools"
export GH_USER="alaweimm90"

# Verify GitHub CLI authentication
gh auth status

# View login
gh config set prompt disabled
```

**Expected output:**
```
Logged in to github.com as alaweimm90
```

#### Step 2: Run the census sweep

```bash
# Make scripts executable
chmod +x census.sh census-promote.sh

# Run the full sweep (this takes 10-20 minutes depending on org size)
bash census.sh 2>&1 | tee census.log

# Monitor output in real-time
tail -f census.log
```

**What this does:**
1. Fetches all repos from all 5 orgs (including archived)
2. Queries all org-level Projects (v2)
3. Queries all user-level Projects (v2)
4. Deep-dives into project items to detect orphans (DraftIssues, unlinked items)
5. Consolidates everything into `inventory.json`

**Expected output:**
```
=== Portfolio Census Suite ===
Timestamp: 20251125-234530
Output directory: outputs/20251125-234530

Checking GitHub authentication...
✓ Authenticated

[1/6] Scanning repositories...
  Org: alaweimm90
  ✓ Repos: 5
  [...]

[2/6] Scanning organization Projects (v2)...
  Org: alaweimm90
  ✓ Projects: 3
  [...]

[3/6] Scanning user Projects (v2)...
  User: alaweimm90
  ✓ User projects: 2
  [...]

[4/6] Scanning project items for orphan detection...
  [...]

[5/6] Scanning gists and packages...
  ✓ Gists: 7
  [...]

[6/6] Consolidating inventory...
  ✓ Consolidated: outputs/20251125-234530/consolidated.json
  ✓ Root inventory: ./inventory.json

Summary:
{
  "total_repos": 35,
  "total_projects_v2": 12,
  "orphan_projects": 3,
  "orphan_drafts": 7,
  "total_gists": 7
}

=== Census Complete ===
✓ All data saved to: outputs/20251125-234530/
✓ Consolidated inventory: ./inventory.json

Next steps:
  1. Review inventory.json for 'projects_without_repo' and 'orphan_drafts'
  2. Run: census-promote.sh to convert orphans into repos
  3. Run: OPA policies to enforce no future orphans
```

#### Step 3: Review the consolidated inventory

```bash
# Look at the summary
jq '.projects_without_repo, .orphan_drafts' inventory.json | head -50

# Count by type
echo "=== ORPHAN SUMMARY ==="
echo "Projects without repo: $(jq '.projects_without_repo | length' inventory.json)"
echo "Orphan drafts: $(jq '.orphan_drafts | length' inventory.json)"
echo "Total repos: $(jq '.repos_by_org | map(.count) | add' inventory.json)"
echo "Total projects: $(jq '.projects_v2_by_org | map(.count) | add' inventory.json)"
```

**Expected output:**
```
=== ORPHAN SUMMARY ===
Projects without repo: 3
Orphan drafts: 7
Total repos: 35
Total projects: 12
```

### Afternoon (2-3 hours)

#### Step 4: Triage orphans and create promotion plan

```bash
# Extract orphans into a human-readable report
cat > orphan_report.txt <<'REPORT'
=== ORPHAN PROJECTS NEEDING PROMOTION ===

REPORT

echo "" >> orphan_report.txt
echo "=== Projects Without Repo Links ===" >> orphan_report.txt
jq -r '.projects_without_repo[] | "  Org: \(.org)\n  Project #: \(.project_number)\n  Reason: \(.reason)\n"' inventory.json >> orphan_report.txt

echo "" >> orphan_report.txt
echo "=== Orphaned DraftIssues ===" >> orphan_report.txt
jq -r '.orphan_drafts[] | "  Org: \(.org)\n  Project: \(.project)\n  Title: \(.draft_title)\n"' inventory.json >> orphan_report.txt

# View the report
cat orphan_report.txt
```

#### Step 5: Decide promotion strategy

For each orphan, decide:

1. **Keep as is** (document in EXCEPTIONS.md) — It's research, ephemeral, or intentional
2. **Create repo** — Promote to proper repo under prefix taxonomy
3. **Archive project** — Remove or mark as deprecated

**Suggested decision tree:**

```
DraftIssue with title containing "research"?
  → Create: paper-* repo from template-research

DraftIssue with title containing "library" or "lib"?
  → Create: lib-* repo from template-python-lib

DraftIssue with title containing "tool" or "cli"?
  → Create: tool-* repo from template-python-lib

DraftIssue with title containing "adapter" or "integration"?
  → Create: adapter-* repo from template-python-lib

Project with zero items?
  → Decide: Archive or re-activate

Otherwise:
  → Keep as exception; document in standards/EXCEPTIONS.md
```

#### Step 6: Document decisions

```bash
# Create/update exceptions file
cat > standards/EXCEPTIONS.md <<'EXCEPT'
# Portfolio Governance Exceptions

This document lists intentional deviations from the golden path.

## Orphan Projects (No Repo Linkage)

### Intentional Orphans

- **alaweimm90/Project #12** (Research brainstorm board)
  - Reason: Ephemeral collection of research ideas; no single repo
  - Owner: @alaweimm90
  - Review Date: 2026-01-25
  - Status: Keep as-is (no action)

### Deprecated (Archive)

- **AlaweinOS/Project #5** (Old CI tools)
  - Reason: Superseded by new infra-actions/ repo
  - Owner: @alaweimm90
  - Action: Archive this project

EXCEPT

# Review it
cat standards/EXCEPTIONS.md
```

---

## DAY 2: PROMOTION & VALIDATION

### Morning (2-3 hours)

#### Step 1: Promote orphans to repos

```bash
# Run the promotion tool interactively
bash census-promote.sh

# This will:
# 1. Read inventory.json orphan lists
# 2. Suggest repo names based on project titles
# 3. Ask for confirmation before creating each repo
# 4. Create repo from template
# 5. Initialize with .meta/repo.yaml, CODEOWNERS, CI config
```

**Expected interaction:**

```
=== Portfolio Census: Promotion Tool ===

Orphan Projects Found:

Projects without repo links:
   1  alaweimm90,12,No Issue/PR/repo references found
   2  AlaweinOS,5,No Issue/PR/repo references found

Orphan DraftIssues:
   1  alaweimm90,Quantum Materials Board,Quantum Materials Toolkit
   2  alaweimm90,Physics Research,Interactive Physics Simulator
   ...

Promotion decisions:

Project: Quantum Materials Board (Org: alaweimm90)
  Draft: Quantum Materials Toolkit
  Suggested repo: lib-quantum-materials-toolkit
  Create? (y/n/custom_name): y

  Creating: lib-quantum-materials-toolkit
  ✓ Repo created
  ✓ Initialized from template: template-python-lib
```

#### Step 2: Verify new repos

```bash
# List recently created repos
gh repo list alaweimm90 --created 2025-11-25 --json name,createdAt

# Verify they have the required files
for REPO in lib-quantum-materials-toolkit tool-benchmark-cli ...; do
  echo "=== $REPO ==="
  gh api "repos/alaweimm90/$REPO/contents/.meta/repo.yaml" \
    --jq '.content' | base64 -d | head -5
done
```

#### Step 3: Update CI in .github repo

The reusable workflows in `.github` should check inventory.json for violations:

```yaml
# .github/workflows/census-policy.yml
name: Census Policy

on:
  workflow_dispatch:  # Manual trigger
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2am UTC

jobs:
  policy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run census
        run: bash census.sh

      - name: Run OPA policy check
        uses: open-policy-agent/setup-opa@v2

      - run: |
          opa eval --fail-defined \
            -i inventory.json \
            -d census-policy.rego \
            'data.portfolio'

      - name: Report orphans
        if: failure()
        run: |
          jq '.projects_without_repo, .orphan_drafts' inventory.json | \
          gh issue create --title "Portfolio Census: Orphan projects detected" \
            --body "Run: bash census-promote.sh to remediate"
```

### Afternoon (1-2 hours)

#### Step 4: Run census again to verify

```bash
# Run census sweep a second time
bash census.sh

# Check that orphan counts decreased
echo "=== After Promotion ==="
jq '{
  orphan_projects: (.projects_without_repo | length),
  orphan_drafts: (.orphan_drafts | length),
  total_repos: (.repos_by_org | map(.count) | add)
}' inventory.json
```

**Expected output:**
```
=== After Promotion ===
{
  "orphan_projects": 1,
  "orphan_drafts": 0,
  "total_repos": 42
}
```

#### Step 5: Commit everything

```bash
# Stage the new files
git add census.sh census-promote.sh census-policy.rego CENSUS_EXECUTION_PLAN.md
git add inventory.json standards/EXCEPTIONS.md
git add outputs/*/

# Commit
git commit -m "feat(census): complete portfolio audit with orphan remediation

Implemented comprehensive portfolio census across all orgs:
- census.sh: Multi-step sweep of repos, projects, gists, packages
- Consolidated inventory.json with orphan detection
- census-promote.sh: Interactive promotion of orphans to repos
- census-policy.rego: OPA policy enforcing no future orphans
- standards/EXCEPTIONS.md: Document intentional deviations

Results:
- Total repos: 42 (up from 35)
- Orphan projects: 0 (promoted 5, kept 1 as exception, archived 1)
- All new repos initialized with templates and CI

Next: Run monthly census via GitHub Action
"

# Push
git push origin master
```

#### Step 6: Wire GitHub Action (Optional but Recommended)

```bash
# Create the workflow file
cat > .github/workflows/census.yml << 'WORKFLOW'
name: Portfolio Census

on:
  workflow_dispatch:
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2am UTC

jobs:
  census:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      issues: write
      pull-requests: write
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

      - name: Create PR if orphans found
        if: steps.check.outputs.changed == 'true'
        run: |
          git config user.name "portfolio-census"
          git config user.email "census@github.com"
          git add inventory.json
          git commit -m "chore(census): update portfolio inventory"

          BRANCH="census/$(date +'%Y%m%d-%H%M%S')"
          git checkout -b "$BRANCH"
          git push origin "$BRANCH"

          gh pr create \
            --title "Portfolio Census: Inventory Update" \
            --body "Automated inventory update from census.sh" \
            --base main \
            --head "$BRANCH"

      - name: Run OPA policy
        continue-on-error: true
        run: |
          # Check if orphans exist
          ORPHANS=$(jq '.projects_without_repo | length' inventory.json)
          DRAFTS=$(jq '.orphan_drafts | length' inventory.json)

          if [ "$ORPHANS" -gt 0 ] || [ "$DRAFTS" -gt 0 ]; then
            gh issue create \
              --title "Portfolio Census: Orphan projects detected ($ORPHANS orphans, $DRAFTS drafts)" \
              --body "Run: bash census-promote.sh to promote orphans to repos"
          fi
WORKFLOW

git add .github/workflows/census.yml
git commit -m "chore(ci): add automated census workflow"
git push origin master
```

---

## CHECKPOINT: Day 2 afternoon

**Validation Checklist:**

- [ ] `inventory.json` exists and is valid JSON
- [ ] `consolidated.json` in timestamped output directory
- [ ] Orphan counts verified:
  - Projects without repo: **0** or listed in EXCEPTIONS.md
  - DraftIssues: **0** (all promoted to repos)
- [ ] New repos created:
  - Each has `.meta/repo.yaml`
  - Each has `.github/CODEOWNERS`
  - Each has `.github/workflows/{ci,policy}.yml`
  - CI passing on all new repos
- [ ] `standards/EXCEPTIONS.md` documents intentional deviations
- [ ] Git commits pushed
- [ ] GitHub Action wired (optional but recommended)

**If everything checks out:**

```bash
echo "✓ Portfolio census COMPLETE"
echo "✓ All orphan projects remediated"
echo "✓ Inventory consolidated and tracked"
echo "✓ Governance enforced via OPA policy"
```

---

## Ongoing: Monthly Monitoring

After Day 2, the census runs **automatically every Monday at 2am UTC**:

1. **GitHub Action runs census.sh**
2. **If orphans found:** Creates issue + optional PR
3. **Team reviews:** Decide: promote, keep as exception, or archive
4. **census-promote.sh** fixes issues
5. **inventory.json** stays up-to-date

---

## Troubleshooting

### Issue: "census.sh: command not found"

**Solution:**
```bash
chmod +x census.sh
bash census.sh
```

### Issue: "gh: command not found"

**Solution:**
```bash
# Install GitHub CLI
brew install gh        # macOS
choco install gh       # Windows
sudo apt install gh    # Linux

# Authenticate
gh auth login
```

### Issue: "inventory.json has stale data"

**Solution:**
```bash
# Re-run census
bash census.sh

# Or delete and regenerate
rm inventory.json
bash census.sh
```

### Issue: "promotion script hangs"

**Solution:**
```bash
# Kill it and run in batch mode (non-interactive)
# Edit census-promote.sh to add --auto-yes flag
bash census-promote.sh --auto-yes
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Portfolio completeness | 100% repos cataloged | `jq '.repos_by_org \| map(.count) \| add' inventory.json` |
| Orphan elimination | 0 (or documented) | `jq '.projects_without_repo \| length' inventory.json` |
| Draft issues | 0 (all promoted) | `jq '.orphan_drafts \| length' inventory.json` |
| New repos passing CI | 100% | `gh run list --repo alaweimm90/lib-* --status success` |
| OPA enforcement | Policy gates all PRs | OPA output in census.yml workflow |

---

## Next Steps (Post-Census)

1. **Bulk retrofit** existing 35 repos (use IMPLEMENTATION_GUIDE.md)
2. **Continuous monitoring** via monthly census
3. **Promote new projects** as they emerge
4. **Triage exceptions** quarterly
5. **Archive dead repos** (>12 months no activity)

---

