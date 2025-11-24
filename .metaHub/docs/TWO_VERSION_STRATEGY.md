# Two-Version Strategy Guide

This document explains how to maintain two versions of your repository: one for public SaaS distribution and one for private development work.

## Table of Contents

- [Overview](#overview)
- [Version Definitions](#version-definitions)
- [Repository Setup](#repository-setup)
- [File Organization](#file-organization)
- [Workflow Examples](#workflow-examples)
- [Synchronization](#synchronization)
- [Branch Strategy](#branch-strategy)
- [Git Configuration](#git-configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Why Two Versions?

**Public Version (SaaS)**

- Clean, scaffolded projects for users
- Clear templates and examples
- Documentation and guides
- Ready-to-use structure
- No implementation details

**Private Version (Development)**

- Full implementation code
- Production logic and algorithms
- Development work-in-progress
- Real configurations
- Internal documentation

### Goals

✅ **Public version** is user-ready and distributed
✅ **Private version** contains full work
✅ **Synchronized** structure and standards
✅ **No secrets** in either repository
✅ **Easy switching** between versions

---

## Version Definitions

### Public SaaS Version

**Purpose**: Distribute to users, show structure and examples

**What's Included**:

- Project scaffolding with correct structure
- README.md files with clear explanations
- CLAUDE.md with project instructions
- Example configurations and sample data
- Template implementations showing patterns
- Complete documentation
- Empty or placeholder implementations
- .gitignore and standard files

**What's NOT Included**:

- Actual implementation code
- Real API keys or secrets
- Production data
- Internal development notes
- Work-in-progress code
- Incomplete features

**Target Users**:

- New developers learning the structure
- Users who want to fork and extend
- Teams integrating your tools
- Open source community

---

### Private Development Version

**Purpose**: Your actual development work

**What's Included**:

- Complete implementation code
- Real configurations
- Production logic
- All internal documentation
- Work-in-progress branches
- Development experiments
- Testing data
- Performance optimizations

**What's NOT Included**:

- API keys (use .env or secrets manager)
- Passwords or credentials
- Sensitive customer data
- Proprietary business logic (if secret)

**Access**:

- Only you and trusted team members
- Never public
- All development happens here
- Push to private repository

---

## Repository Setup

### Option 1: Separate Git Remotes (Recommended)

Set up two remotes in your local repository:

```bash
# Clone the public version first
git clone https://github.com/your-org/your-repo.git
cd your-repo

# Add your private repository as a second remote
git remote add private https://github.com/your-org/your-repo-private.git

# Fetch both remotes
git fetch origin      # public repository
git fetch private     # private repository

# List remotes
git remote -v
# origin   https://github.com/your-org/your-repo.git (fetch)
# origin   https://github.com/your-org/your-repo.git (push)
# private  https://github.com/your-org/your-repo-private.git (fetch)
# private  https://github.com/your-org/your-repo-private.git (push)
```

### Option 2: Separate Clone Directories

Keep two completely separate clones:

```bash
# Public version
git clone https://github.com/your-org/your-repo.git \
    ~/code/your-repo-public

# Private version
git clone https://github.com/your-org/your-repo-private.git \
    ~/code/your-repo-private
```

### Option 3: Branch-Based Strategy

Keep everything in one repository but use different branches:

```bash
# Main branch (public scaffolding)
git checkout main

# Development branch (full implementation)
git checkout -b development
git checkout development
```

**Recommendation**: Use Option 1 (separate remotes) - best of both worlds

---

## File Organization

### Shared Between Versions

These files should be **identical** in both versions:

```
metaHub/
├── README.md
├── CLAUDE.md
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
├── .gitignore
├── CHANGELOG.md
│
├── docs/                  # Structure same, content synchronized
├── scripts/               # Copy scripts from public to private
├── templates/             # Keep synchronized
└── config/                # Keep synchronized

REPO_STANDARDS.md          # Keep synchronized
LICENSE_POLICY.md          # Keep synchronized
ARCHIVE_POLICY.md          # Keep synchronized
```

### Only in Public Version

```
AlaweinOS/
├── README.md              # General purpose description
├── PROJECT_STRUCTURE.md   # Explains what goes where
├── EXAMPLE_PROJECTS.md    # Shows example layouts
└── [Each project]/
    ├── README.md          # Template/example content
    ├── CLAUDE.md          # Generic instructions
    ├── src/               # ← EMPTY or EXAMPLE code
    ├── tests/             # ← EMPTY or EXAMPLE tests
    └── docs/              # ← EXAMPLE documentation

Same structure for all 5 organizations
```

### Only in Private Version

```
AlaweinOS/
├── [Each project]/
    ├── src/               # ← REAL implementation
    ├── tests/             # ← REAL tests
    ├── .env.local         # ← LOCAL ONLY, not committed
    ├── config/            # ← Real configs
    └── docs/              # ← Development notes
```

### .gitignore for Both Versions

**Public version** (root .gitignore):

```bash
# Standard ignores
__pycache__/
node_modules/
dist/
build/

# Don't ignore implementation files in public version!
# (They're just examples/scaffolding)
```

**Private version** (add .gitignore-private):

```bash
# All standard ignores
__pycache__/
node_modules/
dist/

# CRITICAL: Secrets
.env
.env.local
.env.*.local
*.key
*.pem
secrets/
credentials/

# Development
.vscode/
.idea/
*.swp

# Logs
*.log
logs/

# Build artifacts (if not in distributed build/)
*.egg-info/
dist-private/
```

---

## Workflow Examples

### Example 1: Update Public Scaffolding

**Goal**: Improve the project template structure for users

```bash
# 1. Work on public version
git checkout main
cd metaHub/templates/python-project

# Update the template
# - Modify README.md
# - Update pyproject.toml structure
# - Add example comments to src/

git status
# On branch main
# modified: metaHub/templates/python-project/README.md
# modified: metaHub/templates/python-project/pyproject.toml

# 2. Commit to public version
git add metaHub/templates/
git commit -m "docs: improve Python project template

- Updated README with clearer instructions
- Added example configuration to pyproject.toml
- Added helpful comments to example code
"

git push origin main

# 3. Sync to private version
git fetch origin
git checkout private/main
git merge origin/main

# OR if using branches
git checkout development
git pull origin main

git push private main
```

### Example 2: Add Implementation to Private Version

**Goal**: Add actual implementation code to AlaweinOS

```bash
# 1. Work on private version
git checkout private/main
# OR: git checkout development

cd AlaweinOS/Attributa/

# Add real implementation
# - Add src/attribution.py with actual logic
# - Add tests/ with real tests
# - Update docs/ with real API docs

git status

# 2. Commit to private version
git add AlaweinOS/Attributa/
git commit -m "feat: implement attribution analysis

- Add core attribution algorithm
- Add tests for attribution functions
- Add API documentation
"

git push private main
# OR: git push origin development
```

### Example 3: Update Standards (Both Versions)

**Goal**: Update REPO_STANDARDS.md in both versions

```bash
# 1. Update in public version
git checkout main

# Edit REPO_STANDARDS.md
# - Update requirements
# - Add new standards

git add REPO_STANDARDS.md
git commit -m "docs: update repository standards

- Added new naming conventions
- Updated compliance requirements
- Clarified documentation standards
"

git push origin main

# 2. Sync to private
git fetch origin
git checkout private/main
git merge origin/main
git push private main

# OR if using branches
git checkout development
git pull origin main

# If there are conflicts in private version:
git status  # See conflicts
# Resolve conflicts manually
git add .
git commit -m "Merge public standards update"
git push private main
```

### Example 4: Cherry-Pick from Private

**Goal**: Bring a useful script from private to public

```bash
# 1. Have the public and private versions set up
git remote -v
# origin   ...organizations.git (public)
# private  ...organizations-private.git (private)

# 2. Get the commit hash from private version
git fetch private
git log private/main --oneline | head -20

# Example output:
# abc1234 scripts: add new compliance tool
# def5678 feat: improve test coverage

# 3. Cherry-pick the useful commit to public
git checkout main
git cherry-pick abc1234

# OR if conflicts, handle them:
git status  # See conflicts
# Resolve conflicts
git add .
git cherry-pick --continue

# 4. Push to public
git push origin main

# 5. Sync public to private
git checkout private/main
git pull origin main
git push private main
```

---

## Synchronization

### What to Synchronize

**Metabase files** (always keep in sync):

- metaHub/README.md, CLAUDE.md, CONTRIBUTING.md, SECURITY.md
- metaHub/docs/ (structure and shared docs)
- metaHub/scripts/ (utility scripts)
- metaHub/templates/ (keep synchronized)
- metaHub/config/ (shared configurations)
- REPO_STANDARDS.md (critical!)
- LICENSE_POLICY.md
- ARCHIVE_POLICY.md
- .github/ (shared workflows)

**Organization READMEs** (structure synchronized):

- README.md in each organization
- CLAUDE.md in each organization
- CONTRIBUTING.md in each organization
- LICENSE in each organization

### Synchronization Frequency

- **Daily**: Manual pull of public updates to private
- **After Changes**: Push public changes, pull to private
- **Weekly**: Review for drift between versions
- **Monthly**: Full audit of synchronization

### Synchronization Script

Create `.scripts/sync-versions.sh`:

```bash
#!/bin/bash

set -euo pipefail

echo "Synchronizing Public → Private..."

# Fetch latest from both remotes
git fetch origin
git fetch private

# Switch to private main
git checkout private/main

# Merge changes from public main
git merge origin/main

# If there are no conflicts
git push private main

echo "✅ Synchronization complete!"
```

Run periodically:

```bash
bash .scripts/sync-versions.sh
```

---

## Branch Strategy

### Public Repository Branches

```
main (v1.0.0)
├── Latest stable version for users
└── Always ready for distribution

develop
├── Current development branch
└── Where new features are prepared

feature/*
├── feature/improve-templates
├── feature/add-docs
└── feature/new-org

bugfix/*
├── bugfix/broken-link
└── bugfix/formatting
```

### Private Repository Branches

```
main (mirrors public main + your code)
├── Stable version with full implementation
└── Merged when ready for public release

development (mirrors public develop + your code)
├── Development branch with full implementation
└── Where you work on new features

feature/*
├── feature/new-algorithm
├── feature/optimization
└── feature/performance

experimental/*
├── experimental/research
└── experimental/new-approach

work-in-progress/*
├── wip/debugging
└── wip/refactoring
```

### Branch Protection Rules

**Public Repository**:

```
main:
  - Require pull request reviews (1 approval)
  - Require status checks to pass
  - Include administrators
  - Restrict who can push

develop:
  - Require status checks to pass
```

**Private Repository**:

```
main:
  - Same as public (maintains sync)

development:
  - Require status checks to pass

All others:
  - No restrictions (personal work)
```

---

## Git Configuration

### Configure Multiple Remotes

```bash
# Add second remote
git remote add private https://github.com/your-org/your-repo-private.git

# View all remotes
git remote -v

# Fetch both
git fetch --all

# Fetch specific remote
git fetch private
git fetch origin
```

### Push to Specific Remotes

```bash
# Push to public (origin)
git checkout main
git push origin main

# Push to private
git checkout private/main
git push private main

# Or configure push default
git config --global push.default simple
```

### Create Tracking Branches

```bash
# Track private main
git checkout -b private-main --track private/main
git checkout private-main

# Track origin main
git checkout -b main --track origin/main
git checkout main

# Switch between them
git checkout main           # public
git checkout private-main   # private
```

### Git Aliases for Convenience

Add to ~/.gitconfig:

```ini
[alias]
  public = "checkout main"
  private = "checkout private-main"
  sync = "!git fetch --all && git merge origin/main"
  push-all = "!git push origin main && git checkout private-main && git push private main"
```

Usage:

```bash
git public      # Switch to public version
git private     # Switch to private version
git sync        # Sync public → private
git push-all    # Push both versions
```

---

## Best Practices

### 1. **Never Commit Secrets**

❌ **Wrong**:

```bash
API_KEY = "sk_live_abcd1234"  # In committed code
password = "mypassword123"     # In committed file
```

✅ **Right**:

```bash
# Use .env files (not committed)
API_KEY = "${process.env.API_KEY}"

# Use environment variables
os.environ.get("API_KEY")

# Use secrets managers
from github import Github
secrets = Github().get_user().get_secret("API_KEY")
```

### 2. **Keep Structure Synchronized**

Every time you change the structure in public:

```bash
# After pushing to public
git fetch origin
git checkout private/main
git merge origin/main
git push private main
```

### 3. **Clear File Naming**

Use naming to indicate version:

```
myproject.public.json      # For public version
myproject.private.json     # For private version
config.example.json        # Example configuration
```

### 4. **Comprehensive .gitignore**

In private version:

```bash
# Environment
.env
.env.*
!.env.example

# Secrets
*.key
*.pem
credentials/

# Generated
dist/
build/

# Logs
*.log

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

### 5. **README Clarity**

Public version README:

```markdown
## Quick Start

1. Clone this repo
2. Copy a template from metaHub/templates/
3. Follow the template's README
4. Implement your project
```

Private version README:

```markdown
## Setup for Development

1. Clone the private repository
2. Install dev dependencies
3. Run tests to verify setup
4. Create a feature branch
5. Start implementing
```

### 6. **Document the Strategy**

Keep this file updated and reference it:

```bash
# In CONTRIBUTING.md
This project uses a two-version strategy.
See metaHub/TWO_VERSION_STRATEGY.md for details.

# In CI/CD
Check which version is being tested
```

---

## Troubleshooting

### Issue 1: Merged Secret Accidentally

**Problem**: You committed an API key to the private version

**Solution**:

```bash
# Remove from history
git filter-branch --tree-filter 'rm -f config/secrets.json' HEAD

# OR use git-filter-repo (better)
pip install git-filter-repo
git filter-repo --path config/secrets.json --invert-paths

# OR if just committed, not pushed
git reset HEAD~1                  # Undo commit
rm config/secrets.json           # Remove file
git add .
git commit -m "Remove secrets"
```

### Issue 2: Can't Remember Which Version I'm On

**Solution**:

```bash
# Add to prompt
git branch -a | grep "*"

# Or use git config
git config --local --get-regexp 'branch'

# Or create helpful aliases
git config --global alias.which \
  "!echo 'Current branch:' && git branch --show-current"
```

### Issue 3: Conflicts When Syncing

**Problem**: Changes in both public and private versions conflicted

**Solution**:

```bash
# See the conflict
git status
git diff

# Resolve conflicts manually
# - Keep public structure
# - Keep private implementation

# Mark as resolved
git add .

# Complete the merge
git commit -m "Resolve sync conflict

- Kept public documentation updates
- Kept private implementation
"

git push private main
```

### Issue 4: Accidentally Pushed Private to Public

**Problem**: Committed private code to public repository

**Solution**:

```bash
# If not pushed yet
git reset HEAD~1

# If pushed, remove from history
git revert <commit-hash>
git push origin main

# Then update private
git fetch origin
git checkout private/main
git merge origin/main
```

### Issue 5: Out of Sync Versions

**Problem**: Public and private versions have drifted

**Solution**:

```bash
# Check differences
git fetch --all
git diff origin/main private/main -- metaHub/

# In private version
git checkout private/main

# Find divergence point
git merge-base origin/main private/main

# Option A: Rebase (recommended)
git rebase origin/main

# Option B: Merge
git merge origin/main

# Resolve any conflicts
git status
# [resolve conflicts]
git add .
git commit -m "Sync with public version"

git push private main
```

---

## Checklist for Two-Version Workflow

### Before Starting Work

- [ ] Know which version I'm working on (`git branch`)
- [ ] Fetched latest from all remotes (`git fetch --all`)
- [ ] Know where to push my changes

### While Developing

- [ ] Making changes to correct version
- [ ] Not committing secrets or API keys
- [ ] Following REPO_STANDARDS.md
- [ ] Writing meaningful commit messages

### Before Pushing

- [ ] Verified I'm on correct branch
- [ ] Ran tests if applicable
- [ ] Reviewed changes (`git diff`)
- [ ] Updated CHANGELOG.md if needed

### After Pushing

- [ ] Pushed to correct remote
- [ ] Synced other version if needed
- [ ] Updated tracking if new changes

### Weekly

- [ ] Reviewed synchronization status
- [ ] Resolved any drift between versions
- [ ] Checked for accidentally committed secrets
- [ ] Updated TWO_VERSION_STRATEGY.md if needed

---

## Summary

Two-version strategy allows you to:

✅ **Public Version**: Clean scaffolding for users
✅ **Private Version**: Full implementation and work
✅ **Synchronized**: Structure and standards
✅ **Separation**: Development from distribution
✅ **Flexibility**: Easy to switch between

**Key Rules**:

1. Never commit secrets
2. Keep structure synchronized
3. Use clear branch names
4. Document what goes where
5. Review before syncing

---

**Last Updated**: November 21, 2025
**Location**: metaHub/TWO_VERSION_STRATEGY.md
**Contact**: maintainer@example.com
