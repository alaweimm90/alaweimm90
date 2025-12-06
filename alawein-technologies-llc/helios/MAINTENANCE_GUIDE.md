# Repository Maintenance Guide

Quick reference for keeping HELIOS repository clean and organized.

## 🗓️ Monthly Maintenance Checklist

### Week 1: Documentation Review
- [ ] Check root directory for new files that should go to `docs/`
- [ ] Archive any new session reports to `.archive/sessions/YYYY-MM-DD/`
- [ ] Review and update README.md if needed
- [ ] Check for broken documentation links

### Week 2: Branch Cleanup
- [ ] List merged branches: `git branch --merged`
- [ ] Delete local merged branches: `git branch -d <branch-name>`
- [ ] Clean up remote tracking: `git remote prune origin`
- [ ] List stale branches (>30 days): `git for-each-ref --sort=-committerdate refs/heads/ --format='%(refname:short) %(committerdate:short)'`

### Week 3: Archive Maintenance
- [ ] Review `.archive/` structure
- [ ] Check for duplicates or orphaned files
- [ ] Organize new sessions into dated folders
- [ ] Verify archive backups

### Week 4: General Cleanup
- [ ] Run linters and formatters
- [ ] Check for unused dependencies
- [ ] Review .gitignore for relevance
- [ ] Update documentation as needed

## 📁 Directory Structure Rules

### Root Directory Policy
**KEEP ONLY:**
- Core project files: README.md, LICENSE, CONTRIBUTING.md
- Security/conduct: SECURITY.md, CODE_OF_CONDUCT.md
- Architecture: STRUCTURE.md, PROJECT.md
- Config: pyproject.toml, .gitignore, .editorconfig, .env.example
- Implementation: helios/ (source code)

**MOVE TO `docs/`:**
- Guides: QUICK_START.md, DEPLOYMENT.md, DESIGN_SYSTEM.md
- Implementation details: ENTERPRISE_FEATURES_GUIDE.md, HELIOS_ULTRA_ACCELERATION_ENGINE.md
- References: BRANDING_GUIDELINES.md

**MOVE TO `.archive/`:**
- Session reports: *SUMMARY.md, *REPORT.md, *READINESS*.md
- Release notes: RELEASE_NOTES_*.md
- Deprecated features
- Temporary experimental files

### File Organization Workflow

```
New file created:
  ↓
Is it a core doc? (README, LICENSE, etc.)
  ├─ YES → Keep in root
  └─ NO ↓
    Is it a guide/reference?
      ├─ YES → Move to docs/
      └─ NO ↓
        Is it a session report or historical doc?
          ├─ YES → Move to .archive/sessions/YYYY-MM-DD/
          └─ NO → Decide based on relevance
```

## 🔧 Common Maintenance Tasks

### Archive a Session Report
```bash
# Create session folder with today's date
mkdir -p .archive/sessions/$(date +%Y-%m-%d)

# Move session files
mv *SUMMARY.md .archive/sessions/$(date +%Y-%m-%d)/
mv *REPORT.md .archive/sessions/$(date +%Y-%m-%d)/

# Commit
git add .archive/
git commit -m "chore: archive session reports from $(date +%Y-%m-%d)"
git push origin <branch-name>
```

### Clean Merged Branches (LOCAL)
```bash
# List merged branches
git branch --merged

# Delete merged branches (safe)
git branch --merged | grep -v "^\*\|main\|master" | xargs git branch -d

# Verify
git branch -v
```

### Clean Merged Branches (REMOTE)
```bash
# List remote merged branches
git branch -r --merged

# Delete remote branches (requires authorization)
git push origin --delete <branch-name>

# Clean up local tracking
git remote prune origin
```

### Move Files to docs/
```bash
# Ensure docs/ directory exists
mkdir -p docs

# Move specific file
git mv <filename> docs/

# Move multiple files
git mv DESIGN_SYSTEM.md docs/
git mv DEPLOYMENT.md docs/

# Commit
git commit -m "docs: move guides to docs directory"
git push origin <branch-name>
```

### Update Documentation Links
```bash
# Find references to moved files
grep -r "DESIGN_SYSTEM.md" .

# Update links from root to docs/
# Before: [Design System](DESIGN_SYSTEM.md)
# After:  [Design System](docs/DESIGN_SYSTEM.md)
```

## 📊 Repository Health Metrics

### Current Baseline (Post-Cleanup)
- **Root files**: 11 essential files
- **Organized files**: 10 in subdirectories
- **Directories**: 3 main (helios/, docs/, .archive/)
- **Health score**: ✅ Clean (A+)

### Target Metrics (Monthly)
- **Root files**: ≤ 15 (maximum)
- **Merge commits**: Clean history
- **Stale branches**: 0
- **Unresolved conflicts**: 0
- **Documentation**: Up-to-date

### Commands to Check Health
```bash
# Count root files
find . -maxdepth 1 -type f | wc -l

# View structure
tree -L 2 -I "helios|.git" -a

# List all branches
git branch -a

# Check for stale branches
git for-each-ref --sort=-committerdate --format='%(refname:short) %(committerdate:short)' refs/heads/

# Verify clean working tree
git status

# Check recent commits
git log --oneline -10
```

## ✅ Documentation Standards

### New Guide Files
- Location: `docs/`
- Naming: `FEATURE_GUIDE.md`, `COMPONENT_DESIGN.md`
- Include: title, overview, usage examples, troubleshooting
- Link from: README.md or STRUCTURE.md

### Session Reports
- Location: `.archive/sessions/YYYY-MM-DD/`
- Naming: `SESSION_SUMMARY.md`, `READINESS_REPORT.md`
- Include: date, objectives, results, next steps
- Keep for: historical reference

### Architecture Documents
- Location: Root (STRUCTURE.md, PROJECT.md)
- Keep in root for easy discoverability
- Update when major changes occur
- Link from README.md

## 🚀 Automation Scripts

### Daily Cleanup Check
```bash
#!/bin/bash
# Show root directory status
echo "Root files: $(find . -maxdepth 1 -type f | wc -l)"
echo ""
echo "Root directory:"
ls -1 . | grep -v "^[\.]" | grep -v "helios"
echo ""
echo "Branch status:"
git branch -v
```

### Weekly Archive Check
```bash
#!/bin/bash
# Check archive structure
echo "Archive contents:"
find .archive -type f | sort
echo ""
echo "Total archived files: $(find .archive -type f | wc -l)"
```

### Monthly Full Cleanup
```bash
#!/bin/bash
set -e

echo "🧹 Running monthly cleanup..."

# Step 1: Clean merged branches
echo "1. Cleaning merged branches..."
git branch --merged | grep -v "^\*\|main\|master\|develop" | xargs -r git branch -d

# Step 2: Prune remote branches
echo "2. Pruning remote branches..."
git remote prune origin

# Step 3: Archive session reports
echo "3. Archiving session reports..."
mkdir -p .archive/sessions/$(date +%Y-%m-%d)
find . -maxdepth 1 -name "*SUMMARY*.md" -o -name "*REPORT*.md" | xargs -I {} mv {} .archive/sessions/$(date +%Y-%m-%d)/ 2>/dev/null || true

# Step 4: Status report
echo ""
echo "✅ Cleanup complete!"
echo "Root files: $(find . -maxdepth 1 -type f | wc -l)"
echo "Active branches: $(git branch | wc -l)"
echo "Archived files: $(find .archive -type f | wc -l)"
```

## 📋 Troubleshooting

### Issue: File was moved, but history lost
**Solution**: Use `git log --follow <file>` to see full history
```bash
git log --follow docs/DESIGN_SYSTEM.md
```

### Issue: Links pointing to old locations
**Solution**: Search and update references
```bash
grep -r "old_path" . --include="*.md" --include="*.py"
# Then update using find/sed or manually
```

### Issue: Archive grew too large
**Solution**: Compress and move very old sessions
```bash
# Backup and compress
tar czf .archive/sessions/2024.tar.gz .archive/sessions/2024-*/
rm -rf .archive/sessions/2024-*/
```

### Issue: Stale branches accumulating
**Solution**: Regular cleanup schedule
```bash
# Delete branches older than 30 days
git for-each-ref --sort=-committerdate --format='%(refname:short) %(committerdate:short)' refs/heads/ | \
  awk -v date=$(date -d "30 days ago" +%Y-%m-%d) '$2 < date {print $1}' | \
  xargs -r git branch -d
```

## 📞 Need Help?

**Common Questions:**

Q: Is it safe to delete archived files?
A: Yes, once they're in `.archive/` with `git push`. Git preserves history. Create a backup first.

Q: Can I recover accidentally deleted files?
A: Yes! Use `git reflog` and `git checkout` to recover from recent deletions.

Q: How often should I run cleanup?
A: Monthly is recommended. More frequently if you have high PR volume.

Q: Should I archive branches as well as files?
A: No. Use `git branch -d` for merged branches and git history for records.

## 📖 See Also

- [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - Details of the last cleanup
- [STRUCTURE.md](STRUCTURE.md) - Architecture and structure overview
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [Git Documentation](https://git-scm.com/doc)

---

**Last Updated**: 2025-11-19
**Maintenance Schedule**: Monthly
**Status**: Active
