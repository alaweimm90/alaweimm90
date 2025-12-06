# Repository Maintenance Guide

Quick reference for keeping the Foundry repository clean and organized.

---

## 🗂️ Current Structure

```
Foundry/
├── README.md                 # Project overview (START HERE)
├── DOCUMENTATION_INDEX.md    # All documentation reference
├── CLEANUP_SUMMARY.md        # Latest cleanup record
├── MAINTENANCE_GUIDE.md      # This file
├── CHANGELOG.md              # Version history
├── FAQ.md                    # Common questions
│
├── docs/                     # Primary documentation
│   ├── guides/              # All guides & references
│   ├── analysis/
│   ├── audits/
│   ├── legal/
│   ├── roadmaps/
│   └── visuals/
│
├── .archive/                # Archived materials
│   ├── sessions/           # Session-dated archives
│   │   └── 2025-11-19/    # Latest session
│   └── [other archives]
│
├── backend/                 # Backend services
├── frontend/                # Frontend applications
├── infrastructure/          # Infrastructure as code
├── testing/                 # Test suites
├── lib/                     # Shared libraries
├── types/                   # TypeScript definitions
├── utils/                   # Utility functions
└── [other directories]      # Feature-specific code
```

---

## 🔧 Monthly Maintenance Tasks

### 1. Clean Merged Branches (First Monday of month)

```bash
# List branches to be deleted
git branch --merged main

# Delete local merged branches (safe)
git branch --merged main | grep -v "main\|master" | xargs git branch -d

# Clean up remote tracking branches
git remote prune origin

# Verify cleanup
git branch -a
```

**Why:** Merged branches clutter your workspace and confuse new developers.

---

### 2. Archive Session Documents (End of session/milestone)

```bash
# Create dated session archive
mkdir -p .archive/sessions/$(date +%Y-%m-%d)

# Move session docs to archive
mv *SESSION*.md .archive/sessions/$(date +%Y-%m-%d)/ 2>/dev/null
mv *REPORT*.md .archive/sessions/$(date +%Y-%m-%d)/ 2>/dev/null

# Verify archive
ls -la .archive/sessions/$(date +%Y-%m-%d)/

# Commit archive
git add .archive/sessions/$(date +%Y-%m-%d)/
git commit -m "archive: session docs from $(date +%Y-%m-%d)"
```

**When:** After major milestones, sprint completions, or decision points
**What:** Session summaries, sprint reports, decision logs, status updates

---

### 3. Organize New Documentation (As docs are created)

**Guides & References** → `docs/guides/`
```bash
mv SOME_GUIDE.md docs/guides/
```

**Analysis Documents** → `docs/analysis/`
```bash
mv ANALYSIS_REPORT.md docs/analysis/
```

**Keep in Root Only:**
- `README.md` - Project start point
- `CHANGELOG.md` - Version history
- `DOCUMENTATION_INDEX.md` - Doc roadmap
- Config files (tailwind.config.js, docker-compose.yml, etc.)

---

### 4. Update Documentation Index (Monthly)

Edit `DOCUMENTATION_INDEX.md`:

```markdown
## Latest Guides (docs/guides/)
- [Guide Name](docs/guides/GUIDE_NAME.md)
- New guides added here

## Archive Sessions
- [2025-11-19 Session](docs/.archive/sessions/2025-11-19/)
```

**Why:** Helps team find what they need quickly.

---

### 5. Check Root Directory Health (Monthly)

```bash
# Count files in root
ls -1 | wc -l

# List markdown files in root
ls -1 *.md

# Remove any obvious temp/duplicate files
ls -la | grep -E "(temp_|old_|backup_|\.tmp)"
```

**Target:** Keep root files minimal (< 50 files)

---

## 🚨 Common Issues & Solutions

### Issue: Too Many Files in Root

**Solution:**
```bash
# 1. Identify non-essential docs
ls -1 *.md | sort

# 2. Move guides to docs/guides/
mv GUIDE_NAME.md docs/guides/

# 3. Archive completed docs
mkdir -p .archive/old
mv COMPLETED_DOC.md .archive/old/

# 4. Commit cleanup
git add .
git commit -m "chore: organize docs into proper directories"
```

### Issue: Stale Branches Accumulating

**Solution:**
```bash
# See all branches with last commit date
git branch -vv

# Delete specific stale branch
git branch -d branch-name

# Delete remote stale branch
git push origin --delete branch-name
```

### Issue: Broken Documentation Links

**Solution:**
```bash
# Search for broken links
grep -r "docs/FILENAME.md" . --include="*.md"

# If file was moved, update all references
# Use your editor's find-and-replace:
# OLD: docs/FILENAME.md
# NEW: docs/guides/FILENAME.md
```

---

## 📋 Pre-Release Checklist

Before releasing a new version:

- [ ] All session docs archived
- [ ] Merged branches cleaned
- [ ] Root directory organized
- [ ] Documentation index updated
- [ ] Broken links fixed
- [ ] CHANGELOG.md updated
- [ ] No temp/backup files in root
- [ ] `.archive/` contains dated sessions
- [ ] `docs/guides/` is organized
- [ ] README.md is current

---

## 🔄 Automated Maintenance Script

Create `.githooks/cleanup-reminder.sh`:

```bash
#!/bin/bash
# Runs on git commit - checks if cleanup needed

ROOT_FILES=$(ls -1 | wc -l)

if [ $ROOT_FILES -gt 50 ]; then
  echo "⚠️  Root directory has $ROOT_FILES files (target: < 50)"
  echo "Run: git branch --merged | xargs git branch -d"
fi

STALE_BRANCHES=$(git branch -vv | grep gone | wc -l)

if [ $STALE_BRANCHES -gt 0 ]; then
  echo "⚠️  Found $STALE_BRANCHES stale branches"
  echo "Run: git remote prune origin"
fi
```

---

## 📞 Questions?

- **How to find a guide?** → Check `DOCUMENTATION_INDEX.md`
- **Where do archived docs go?** → `.archive/sessions/YYYY-MM-DD/`
- **What belongs in root?** → Only essential docs & configs
- **Can I delete something?** → Archive it first, then delete if unused after 1 month

---

## 📅 Maintenance Calendar

```
Week 1:  Branch cleanup (merged branches)
Week 2:  Check root directory health
Week 3:  Update documentation index
Week 4:  Archive old session docs
```

---

## ✅ Last Completed Maintenance

- **Date:** 2025-11-19
- **Type:** Full repository cleanup
- **Files Organized:** 17 (4 archived, 13 moved to docs/guides/)
- **Root Reduction:** 50 → 46 files
- **Summary:** See `CLEANUP_SUMMARY.md`

---

Keep the repository clean. Happy coding! 🚀
