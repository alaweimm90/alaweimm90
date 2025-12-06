# live-it-iconic Documentation Consolidation Plan
## Reducing 43 Root Files → 10 Essential Files

**Current State:** 43 MD files at root + 72 in subdirectories = 115 total
**Target State:** 10 at root + 30 organized in docs/ + archive old reports = ~40 total

---

## 📋 Root Level Files (Keep 10 Essential)

### ✅ Keep at Root
1. **README.md** - Main entry point
2. **PROJECT.md** - Project overview (NEW)
3. **STRUCTURE.md** - Directory guide (NEW)
4. **QUICK_START.md** - Getting started guide
5. **WELLNESS_PLATFORM_DOCUMENTATION.md** - Main feature documentation
6. **DEPLOYMENT_INFRASTRUCTURE_PLAN.md** - Deployment guide
7. **CHANGELOG.md** - Version history
8. **CONTRIBUTING.md** - Contribution guidelines
9. **SECURITY.md** - Security policies
10. **LICENSE** - Legal (not MD)

### ❌ Remove from Root (Move or Archive)

---

## 🗂️ New docs/ Structure

```
docs/
├── README.md                           # Navigation index (NEW)
│
├── getting-started/
│   ├── installation.md                 # From QUICK_START.md content
│   ├── configuration.md                # Environment setup
│   └── first-steps.md                  # Initial walkthrough
│
├── guides/
│   ├── wellness-features.md            # Consolidated wellness guide
│   ├── i18n-guide.md                   # MERGE: I18N_* files
│   ├── testing-guide.md                # MERGE: E2E_TESTING_GUIDE + TEST-SUMMARY
│   ├── storybook.md                    # MOVE: STORYBOOK_SETUP
│   ├── stripe-integration.md           # MOVE: STRIPE_INTEGRATION
│   ├── social-media.md                 # MERGE: TWITCH_YOUTUBE_SETUP + YOUTUBE_STRATEGY
│   └── admin-dashboard.md              # MOVE: ADMIN_DASHBOARD_DOCS
│
├── api/
│   ├── README.md                       # API overview
│   ├── endpoints.md                    # From API.md
│   ├── authentication.md               # Auth endpoints
│   └── webhooks.md                     # Webhook documentation
│
├── architecture/
│   ├── system-design.md                # From ARCHITECTURE.md
│   ├── database-schema.md              # Database structure
│   ├── security.md                     # Security architecture
│   └── performance.md                  # From PERFORMANCE.md
│
├── deployment/
│   ├── infrastructure.md               # From DEPLOYMENT.md + CICD.md
│   ├── monitoring.md                   # From MONITORING.md
│   └── launch-checklist.md             # MOVE: LAUNCH_CHECKLIST
│
├── reference/
│   ├── commit-messages.md              # MOVE: COMMIT_MESSAGE_GUIDE
│   ├── pull-requests.md                # MOVE: PULL_REQUEST
│   ├── governance.md                   # MOVE: GOVERNANCE
│   ├── reference-card.md               # MOVE: REFERENCE_CARD
│   └── environment-variables.md        # Configuration reference
│
├── ai/
│   ├── claude-prompts.md               # MOVE: CLAUDE.md
│   └── superprompt.md                  # MOVE: ICONIC_SUPERPROMPT
│
└── archive/
    ├── implementation-reports/
    │   ├── accessibility-audit.md      # ARCHIVE: ACCESSIBILITY_*
    │   ├── backup-system.md            # ARCHIVE: BACKUP_SYSTEM_SUMMARY
    │   ├── deliverables.md             # ARCHIVE: DELIVERABLES_SUMMARY
    │   ├── i18n-implementation.md      # ARCHIVE: I18N_IMPLEMENTATION_SUMMARY
    │   ├── integration-summary.md      # ARCHIVE: INTEGRATION_SUMMARY
    │   ├── performance-report.md       # ARCHIVE: PERFORMANCE_OPTIMIZATION_REPORT
    │   ├── security-fixes.md           # ARCHIVE: SECURITY_FIXES_APPLIED
    │   └── implementation.md           # ARCHIVE: IMPLEMENTATION_SUMMARY
    └── README.md                       # Archive index
```

---

## 📦 File Mapping

### Move to `docs/guides/`
- I18N_DOCUMENTATION.md → docs/guides/i18n-guide.md
- I18N_INTEGRATION_GUIDE.md → docs/guides/i18n-guide.md (merge)
- I18N_QUICK_REFERENCE.md → docs/guides/i18n-guide.md (merge)
- E2E_TESTING_GUIDE.md → docs/guides/testing-guide.md
- STORYBOOK_SETUP.md → docs/guides/storybook.md
- STRIPE_INTEGRATION.md → docs/guides/stripe-integration.md
- TWITCH_YOUTUBE_SETUP.md → docs/guides/social-media.md
- YOUTUBE_STRATEGY.md → docs/guides/social-media.md (merge)
- ADMIN_DASHBOARD_DOCS.md → docs/guides/admin-dashboard.md

### Move to `docs/reference/`
- COMMIT_MESSAGE_GUIDE.md → docs/reference/commit-messages.md
- PULL_REQUEST.md → docs/reference/pull-requests.md
- GOVERNANCE.md → docs/reference/governance.md
- REFERENCE_CARD.md → docs/reference/reference-card.md

### Move to `docs/ai/`
- CLAUDE.md → docs/ai/claude-prompts.md
- ICONIC_SUPERPROMPT.md → docs/ai/superprompt.md

### Move to `docs/deployment/`
- LAUNCH_CHECKLIST.md → docs/deployment/launch-checklist.md
- DEPLOY_NOW.md → docs/deployment/quick-deploy.md
- LAUNCH_README.md → docs/deployment/launch-guide.md

### Move to `docs/planning/` (NEW)
- BUSINESS_PLAN.md → docs/planning/business-plan.md

### Archive to `docs/archive/implementation-reports/`
- ACCESSIBILITY_AUDIT.md
- ACCESSIBILITY_SUMMARY.md
- BACKUP_SYSTEM_SUMMARY.md
- DELIVERABLES_SUMMARY.md
- FIX_GOOGLE_FAVICON.md
- I18N_IMPLEMENTATION_SUMMARY.md
- I18N_INTEGRATION_CHECKLIST.md
- IMPLEMENTATION_SUMMARY.md
- INTEGRATION_SUMMARY.md
- PERFORMANCE_OPTIMIZATION_REPORT.md
- PRODUCT_CATALOG_DELIVERY.md
- SECURITY_FIXES_APPLIED.md
- SECURITY_IMPLEMENTATION_SUMMARY.md
- TEST-SUMMARY.md
- DOCUMENTATION_COMPLETION_SUMMARY.txt
- FILES_CREATED.txt

### Delete (Redundant)
- MONOREPO_STRUCTURE.md (replaced by root STRUCTURE.md)
- docs/QUICK_START.md (duplicate of root)
- STORYBOOK_README.txt (replaced by STORYBOOK_SETUP.md)

---

## 🎯 Implementation Steps

### Step 1: Create New Structure (10 min)
```bash
mkdir -p docs/{getting-started,guides,api,architecture,deployment,reference,ai,planning,archive/implementation-reports}
```

### Step 2: Create Navigation Files (15 min)
- docs/README.md - Main documentation index
- PROJECT.md - Project overview
- STRUCTURE.md - Directory guide

### Step 3: Move Files (30 min)
Execute file moves according to mapping above

### Step 4: Merge Related Files (30 min)
- Merge I18N_* files into single guide
- Merge social media files
- Consolidate testing docs

### Step 5: Archive Reports (10 min)
Move implementation reports to archive

### Step 6: Delete Redundant (5 min)
Remove duplicate files

### Step 7: Update Links (20 min)
Fix internal references in moved files

### Step 8: Verify (10 min)
Check all links work

**Total Time:** ~2 hours

---

## 📊 Before & After

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Root MD files | 43 | 10 | 77% reduction |
| Total MD files | 115 | ~40 | 65% reduction |
| Documentation depth | Flat | 3 levels | Organized |
| Navigation | None | Index + guides | Discoverable |
| Duplicates | Many | 0 | Clean |

---

## ✅ Success Criteria

- [ ] ≤ 10 MD files at root
- [ ] Clear 3-level hierarchy (docs/category/file.md)
- [ ] Navigation index (docs/README.md)
- [ ] No duplicate content
- [ ] All links working
- [ ] Archive has index
- [ ] PROJECT.md created
- [ ] STRUCTURE.md created

---

**Ready to execute!** 🚀
