# Bootstrap Quick-Start Guide

**Time to execute:** 20-30 minutes (automated)
**Repos created:** 14 foundation repos
**Code provided:** Yes (all repos initialized with starter code)
**Manual setup required after:** Branch protection, main branch set as default

---

## 🚀 One-Command Bootstrap

```bash
# Preview what will be created (no changes)
bash bootstrap.sh --dry-run

# Execute bootstrap (creates 14 repos, pushes code)
bash bootstrap.sh

# Or: Create repos but skip GitHub push (push manually later)
bash bootstrap.sh --skip-push
```

---

## ✅ What Gets Created

The script creates and initializes **14 repos**:

### Core Infrastructure
1. **`.github/`** — Reusable CI workflows (Python, TypeScript, policy)
2. **`standards/`** — SSOT for policies, naming, docs (stub repo, manual setup required)
3. **`core-control-center/`** — DAG orchestrator (stub repo, manual setup required)

### Adapters (Provider Integrations)
4. **`adapter-claude/`** — Claude API integration
5. **`adapter-openai/`** — OpenAI API integration
6. **`adapter-lammps/`** — LAMMPS molecular dynamics runner
7. **`adapter-siesta/`** — SIESTA quantum chemistry runner

### Golden Templates
8. **`template-python-lib/`** — Python library starter (stub)
9. **`template-ts-lib/`** — TypeScript library starter (stub)
10. **`template-research/`** — Jupyter research starter (stub)
11. **`template-monorepo/`** — Monorepo starter (stub, pnpm/uv)

### Infrastructure & Demos
12. **`infra-actions/`** — Composite GitHub Actions (stub)
13. **`infra-containers/`** — GHCR base images (stub)
14. **`demo-physics-notebooks/`** — Reference examples (stub)

---

## 📋 Pre-Flight Checklist

Before running bootstrap:

- [ ] You have GitHub CLI (`gh`) installed and authenticated
  ```bash
  gh auth login  # If not already authenticated
  ```

- [ ] You have `git` installed
  ```bash
  git --version
  ```

- [ ] You're in the alaweimm90 GitHub account (not a different org)
  ```bash
  gh auth status
  ```

- [ ] You have bash (or can use WSL on Windows)
  ```bash
  bash --version
  ```

---

## 🎯 Execution Steps

### Step 1: Preview Bootstrap (1 min)

```bash
cd /path/to/alaweimm90-repo
bash bootstrap.sh --dry-run
```

**Expected output:** List of 14 repos that will be created (no actual changes).

### Step 2: Execute Bootstrap (10 min)

```bash
bash bootstrap.sh
```

**Expected output:**
```
✅ Created: alaweimm90/.github
✅ Created: alaweimm90/standards
✅ Created: alaweimm90/core-control-center
...
[14 repos total]
✅ Bootstrap complete!
```

**What happens:**
1. All 14 repos created on GitHub
2. `.github` repo initialized with reusable workflows + pushed
3. Other repos created as empty (you populate them in step 4)

### Step 3: Verify Repos Created (2 min)

```bash
# List all new repos
gh repo list alaweimm90 --limit 20 --json name

# Or visit: https://github.com/alaweimm90?tab=repositories
```

You should see 14 new repos listed.

### Step 4: Populate Remaining Repos (10 min)

For repos other than `.github`, you need to populate them with code:

**For each stub repo:**

```bash
# Example: populate standards/
mkdir -p /tmp/standards
cd /tmp/standards
git init -b main
git remote add origin https://github.com/alaweimm90/standards.git

# Copy code from GITHUB_OS.md section "2) standards/ — SSOT"
# Into relevant directories

git add .
git commit -m "chore: initialize standards with OPA policies"
git push -u origin main
```

**Repos that need manual population:**
- `standards/` — Copy from GITHUB_OS.md section 2
- `core-control-center/` — Copy from GITHUB_OS.md section 3
- `template-python-lib/` — Copy from GITHUB_OS.md section 6.1
- `template-ts-lib/` — Copy from GITHUB_OS.md section 6.2
- `template-research/` — Copy from GITHUB_OS.md section 6.3
- `template-monorepo/` — Copy from GITHUB_OS.md section 6.4
- `infra-actions/` — Copy from GITHUB_OS.md section 7.1
- `infra-containers/` — Copy from GITHUB_OS.md section 7.2
- `demo-physics-notebooks/` — Create Jupyter example

**Repos that are auto-populated by bootstrap:**
- `.github/` ✅ Done
- `adapter-claude/` ✅ Stub (needs dependencies only)
- `adapter-openai/` ✅ Stub (needs dependencies only)
- `adapter-lammps/` ✅ Stub (needs dependencies only)
- `adapter-siesta/` ✅ Stub (needs dependencies only)

---

## 🛡️ Post-Bootstrap Setup (Manual)

### 1. Set Main Branch as Default

For each repo, go to **Settings → Branches** and set `main` as the default branch.

Or use:

```bash
gh repo edit alaweimm90/.github --default-branch main
gh repo edit alaweimm90/standards --default-branch main
# ... repeat for each repo
```

### 2. Enable Branch Protection (Optional but Recommended)

For `.github` repo:

```bash
gh api repos/alaweimm90/.github/branches/main/protection \
  -X PUT \
  -f required_status_checks.strict=true \
  -f required_status_checks.contexts='["ci","policy"]' \
  -f required_pull_request_reviews.dismiss_stale_reviews=true \
  -f required_pull_request_reviews.require_code_owner_reviews=true
```

### 3. Add Repo Topics (Optional)

```bash
gh repo edit alaweimm90/.github --add-topic governance --add-topic ci
gh repo edit alaweimm90/standards --add-topic policy --add-topic opa
gh repo edit alaweimm90/core-control-center --add-topic orchestration --add-topic dag
```

---

## 🐛 Troubleshooting

### Error: "Repository already exists"

**Cause:** Repo was already created earlier
**Solution:** Skip that step or delete the repo first
```bash
gh repo delete alaweimm90/.github --yes
```

### Error: "Not authorized to perform this action"

**Cause:** GitHub CLI not authenticated or not logged into correct account
**Solution:**
```bash
gh auth logout
gh auth login  # Re-authenticate
```

### Script fails with "command not found: gh"

**Cause:** GitHub CLI not installed
**Solution:**
```bash
# macOS
brew install gh

# Windows
choco install gh

# Linux
sudo apt install gh
```

### repos created but not appearing on GitHub

**Cause:** GitHub is caching
**Solution:** Wait 30 seconds and refresh https://github.com/alaweimm90

---

## ✨ What's Next

After bootstrap completes:

1. **Populate stub repos** (see Step 4 above)
2. **Retrofit priority repos** — Add `.meta/repo.yaml`, CODEOWNERS, call reusable CI to:
   - `organizations/alaweimm90-business/repz`
   - `organizations/alaweimm90-business/live-it-iconic`
   - `organizations/AlaweinOS/optilibria`
   - `organizations/AlaweinOS/AlaweinOS`
   - `organizations/alaweimm90-science/mag-logic`

3. **Monitor compliance** — Run monthly audit to track gaps closing

See **IMPLEMENTATION_GUIDE.md** for week-by-week timeline.

---

## 📊 Success Verification

After bootstrap, verify:

```bash
# Count repos
gh repo list alaweimm90 --limit 50 | wc -l
# Should show ≥14 new repos

# Check .github repo has workflows
gh api repos/alaweimm90/.github/contents/.github/workflows --jq '.[].name'
# Should show: reusable-python-ci.yml, reusable-ts-ci.yml, reusable-policy.yml

# Verify default branch
gh api repos/alaweimm90/.github --jq .default_branch
# Should show: main (if set in post-bootstrap step)
```

---

## 📝 Notes

- **Dry-run is safe** — Run `--dry-run` first to preview
- **Idempotent** — Running bootstrap twice won't duplicate repos (skips existing ones)
- **Network:** Script makes ~14 API calls to GitHub (takes 30-60 seconds)
- **No data loss:** Bootstrap only creates empty repos; no existing repos modified

---

## 🎓 Learning Path

After bootstrap, learn how repos are used:

1. **Reusable CI** → See `.github` repo workflows
2. **OPA Policies** → See `standards` repo
3. **Core orchestrator** → See `core-control-center` repo
4. **Adapters** → See any `adapter-*` repo
5. **Templates** → Clone `template-python-lib` as starting point for new projects

---

## 📞 Support

- **Script issues:** Run with `--dry-run` first to debug
- **Repo structure questions:** See `GITHUB_OS.md`
- **Implementation timeline:** See `IMPLEMENTATION_GUIDE.md`
- **Audit results:** See `AUDIT_SUMMARY.md` + `gaps.md`

---

## ⏱️ Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| Dry-run preview | 1 min | Verification list |
| Actual bootstrap | 10 min | 14 repos on GitHub |
| Populate stubs | 30 min | All repos have code |
| Set defaults + protection | 10 min | Ready for CI |
| **Total** | **50 min** | **Foundation ready** |

Then: Retrofit priority repos (see IMPLEMENTATION_GUIDE.md for timeline).

---

Ready? Run:

```bash
bash bootstrap.sh --dry-run
```

Then if preview looks good:

```bash
bash bootstrap.sh
```

Good luck! 🚀
