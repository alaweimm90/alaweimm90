# 🚀 KILO RADICAL SIMPLIFICATION - ACTION PLAN

**Status:** Ready to Execute  
**Timeline:** 4 Weeks  
**Current Phase:** Phase 2 (Deletion)

---

## 📋 PRE-EXECUTION CHECKLIST

Before starting ANY deletions:

- [ ] **Backup current state**

  ```bash
  git add -A
  git commit -m "Pre-KILO cleanup snapshot"
  git tag pre-kilo-cleanup
  git push origin main --tags
  ```

- [ ] **Verify git status**

  ```bash
  git status
  # Should show clean working directory
  ```

- [ ] **Create cleanup branch**

  ```bash
  git checkout -b kilo-cleanup
  ```

- [ ] **Review audit report**
  - Read [`KILO-AUDIT-REPORT.md`](KILO-AUDIT-REPORT.md)
  - Understand deletion targets
  - Identify any files to preserve

---

## 🗑️ WEEK 1: RUTHLESS DELETION

### Day 1: Delete Migration Archive & Old Docs

#### Step 1: Run Dry Run

```powershell
.\scripts\kilo-cleanup.ps1 -DryRun
```

#### Step 2: Review Output

- Check what will be deleted
- Verify no critical files in deletion list
- Note space savings

#### Step 3: Execute Deletion

```powershell
.\scripts\kilo-cleanup.ps1 -Force
```

#### Step 4: Verify & Commit

```bash
git status
git add -A
git commit -m "KILO Week 1 Day 1: Delete migration archive and old docs"
```

**Expected Results:**

- ✅ Deleted: `docs/migration-archive/` (~50 files)
- ✅ Deleted: `docs/archive/` (~20 files)
- ✅ Space freed: ~5-10 MB
- ✅ Cleaner docs structure

---

### Day 2: Clean Infrastructure Directory

#### Manual Review Required

```bash
# List infrastructure contents
ls -R tools/infrastructure/

# Identify what should move to templates
# Identify what should be deleted
```

#### Move to Templates (if needed)

```bash
# Example: Move useful Kubernetes configs
mkdir -p templates/devops/infrastructure/
cp -r tools/infrastructure/kubernetes/base templates/devops/infrastructure/
```

#### Delete Redundant Infrastructure

```bash
# Delete Ansible (should be in templates if needed)
rm -rf tools/infrastructure/ansible

# Delete GitOps (ArgoCD + FluxCD - redundant)
rm -rf tools/infrastructure/gitops

# Delete Terraform environments (not needed in meta repo)
rm -rf tools/infrastructure/terraform/environments
```

#### Commit

```bash
git add -A
git commit -m "KILO Week 1 Day 2: Clean infrastructure directory"
```

**Expected Results:**

- ✅ Deleted: ~200 infrastructure files
- ✅ Moved: Essential configs to templates
- ✅ Space freed: ~15-20 MB

---

### Day 3: Remove Debug Statements

#### Automated Cleanup

The cleanup script handles this, but verify:

```bash
# Find remaining console.log
grep -r "console\.log" --include="*.ts" --include="*.js" --exclude-dir=node_modules --exclude-dir=templates

# Find remaining print statements
grep -r "print(" --include="*.py" --exclude-dir=node_modules --exclude-dir=templates
```

#### Manual Review

- Check if any are needed for CLI output
- Keep intentional logging
- Remove debug statements

#### Commit

```bash
git add -A
git commit -m "KILO Week 1 Day 3: Remove debug statements"
```

**Expected Results:**

- ✅ Removed: 140+ debug statements
- ✅ Cleaner production code
- ✅ Kept: Intentional CLI output

---

### Day 4: Standardize YAML Extensions

#### Automated Rename

```powershell
# Already handled by cleanup script
# Verify all .yml → .yaml
Get-ChildItem -Recurse -Filter *.yml | Where-Object { $_.FullName -notmatch 'node_modules|\.git' }
# Should return empty
```

#### Update References

```bash
# Find files referencing .yml
grep -r "\.yml" --include="*.ts" --include="*.js" --include="*.py" --include="*.md"

# Update to .yaml
# (Manual or script-based)
```

#### Commit

```bash
git add -A
git commit -m "KILO Week 1 Day 4: Standardize YAML extensions"
```

**Expected Results:**

- ✅ All .yml → .yaml
- ✅ Updated references
- ✅ Consistent naming

---

### Day 5: Address TODO/FIXME Comments

#### Generate Report

```bash
# Find all TODOs
grep -rn "TODO\|FIXME\|HACK\|XXX" --include="*.ts" --include="*.js" --include="*.py" > todo-report.txt
```

#### Categorize

1. **Fix immediately** - Critical issues
2. **Create issues** - Future work
3. **Delete** - Outdated comments

#### Action

```bash
# For each TODO:
# - Fix the issue, OR
# - Create GitHub issue, OR
# - Delete the comment

# Example: Remove outdated TODOs
sed -i '/TODO: Old feature/d' file.ts
```

#### Commit

```bash
git add -A
git commit -m "KILO Week 1 Day 5: Address TODO/FIXME comments"
```

**Expected Results:**

- ✅ All TODOs addressed
- ✅ Issues created for future work
- ✅ Cleaner codebase

---

### Week 1 Summary

**Metrics to Track:**

```bash
# Before Week 1
Total Files: 5,239
Total Lines: 719,543
Markdown Files: 1,831
Config Files: 697

# After Week 1 (Target)
Total Files: ~4,900 (-339)
Total Lines: ~670,000 (-49,543)
Markdown Files: ~1,760 (-71)
Config Files: ~560 (-137)
```

**Commit & Push:**

```bash
git push origin kilo-cleanup
# Create PR for review
```

---

## 🔄 WEEK 2: CONSOLIDATION

### Day 1: Consolidate Tool Directories

#### Current Structure Analysis

```bash
# List all tool directories
ls -la tools/

# Count files per directory
find tools/ -type f | cut -d/ -f2 | sort | uniq -c
```

#### Create New Structure

```bash
# Create src/ directory
mkdir -p src/{cli,governance,orchestration,utils}

# Move TypeScript tools to src/cli/
mv tools/devops/*.ts src/cli/

# Move Python governance to src/governance/
mv tools/governance/*.py src/governance/

# Move orchestration to src/orchestration/
mv tools/orchestration/*.py src/orchestration/
mv tools/mcp-servers/*.py src/orchestration/mcp/

# Move utilities
mv tools/devops/fs.ts src/utils/
mv tools/devops/config.ts src/utils/
```

#### Update Imports

```bash
# Find and update import paths
grep -r "tools/devops" --include="*.ts" --include="*.js"
# Update to src/cli or src/utils

grep -r "tools/governance" --include="*.py"
# Update to src/governance
```

#### Commit

```bash
git add -A
git commit -m "KILO Week 2 Day 1: Consolidate tool directories"
```

---

### Day 2: Merge Duplicate Scripts

#### Identify Duplicates

```bash
# Find similar shell scripts
find tools/ai-orchestration -name "*.sh" -exec basename {} \; | sort
find tools/security -name "*.sh" -exec basename {} \; | sort
```

#### Consolidate

```bash
# Create organized scripts directory
mkdir -p scripts/{build,deploy,test,security,orchestration}

# Move and merge scripts
mv tools/security/*.sh scripts/security/
mv tools/ai-orchestration/*.sh scripts/orchestration/

# Merge similar functionality
# (Manual review and merge)
```

#### Commit

```bash
git add -A
git commit -m "KILO Week 2 Day 2: Consolidate shell scripts"
```

---

### Day 3: Consolidate Configuration Files

#### Audit Current Configs

```bash
# Find all config files
find . -name "*.yaml" -o -name "*.json" | grep -v node_modules | grep -v templates
```

#### Create Unified Configs

```bash
# Create single config.yaml
cat > config.yaml << 'EOF'
# Application Configuration
app:
  name: meta-governance
  version: 1.0.0

# DevOps Settings
devops:
  templates_dir: ./templates/devops
  output_dir: ./output

# Governance Settings
governance:
  rules_dir: ./src/governance/rules
  policies_dir: ./src/governance/policies

# MCP Settings
mcp:
  config_file: ./.mcp/config.json
  servers_dir: ./src/orchestration/mcp
EOF
```

#### Delete Redundant Configs

```bash
# Keep only essential configs:
# - config.yaml (app config)
# - .env.example (environment template)
# - package.json (Node deps)
# - tsconfig.json (TypeScript)
# - eslint.config.js (linting)
# - .prettierrc (formatting)
# - vitest.config.ts (testing)

# Delete everything else
```

#### Commit

```bash
git add -A
git commit -m "KILO Week 2 Day 3: Consolidate configuration files"
```

---

### Day 4: Consolidate Documentation

#### Create New Docs Structure

```bash
# Keep only essential docs
mkdir -p docs-new

# Core documentation
cp README.md docs-new/
cat > docs-new/README.md << 'EOF'
# Documentation Index

- [Quick Start](QUICK-START.md)
- [CLI Reference](CLI.md)
- [API Reference](API.md)
- [Architecture](ARCHITECTURE.md)
- [Templates Guide](TEMPLATES.md)
- [Governance Rules](GOVERNANCE.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
EOF
```

#### Consolidate Content

```bash
# Merge related docs
# Extract key info from 1,831 markdown files
# Create 10 comprehensive docs

# Example: Consolidate all CLI docs
cat docs/AI-TOOL-PROFILES.md docs/DEVOPS-MCP-SETUP.md > docs-new/CLI.md
```

#### Replace Old Docs

```bash
rm -rf docs/
mv docs-new/ docs/
```

#### Commit

```bash
git add -A
git commit -m "KILO Week 2 Day 4: Consolidate documentation"
```

---

### Day 5: Consolidate Tests

#### Reorganize Tests

```bash
# Create test structure mirroring src/
mkdir -p tests/{cli,governance,orchestration,utils}

# Move tests to match source structure
mv tests/devops*.test.ts tests/cli/
mv tests/test_*.py tests/governance/
```

#### Remove Duplicate Tests

```bash
# Find and merge duplicate test files
# (Manual review required)
```

#### Commit

```bash
git add -A
git commit -m "KILO Week 2 Day 5: Consolidate tests"
```

---

### Week 2 Summary

**Metrics to Track:**

```bash
# After Week 2 (Target)
Total Files: ~2,900 (-2,000)
Total Lines: ~470,000 (-200,000)
Markdown Files: ~50 (-1,710)
Config Files: ~20 (-540)
```

---

## ⚡ WEEK 3: SIMPLIFICATION

### Day 1: Create Unified CLI Entry Points

#### Design CLI Structure

```typescript
// src/cli/index.ts
import { Command } from 'commander';

const program = new Command();

program.name('meta-governance').description('Meta-governance CLI tools').version('1.0.0');

// DevOps commands
program
  .command('devops')
  .description('DevOps template operations')
  .option('--list', 'List templates')
  .option('--build <template>', 'Build template')
  .option('--code <action>', 'Generate code')
  .action(devopsCommand);

// Governance commands
program
  .command('governance')
  .description('Governance validation')
  .option('--validate', 'Validate structure')
  .option('--enforce', 'Enforce policies')
  .action(governanceCommand);

// MCP commands
program
  .command('mcp')
  .description('MCP server operations')
  .option('--start', 'Start MCP servers')
  .option('--stop', 'Stop MCP servers')
  .action(mcpCommand);

program.parse();
```

#### Update package.json

```json
{
  "bin": {
    "meta-gov": "./src/cli/index.ts"
  },
  "scripts": {
    "cli": "tsx src/cli/index.ts"
  }
}
```

#### Commit

```bash
git add -A
git commit -m "KILO Week 3 Day 1: Create unified CLI"
```

---

### Day 2-5: Continue Simplification

(Similar detailed steps for remaining simplification tasks)

---

## 🛡️ WEEK 4: ENFORCEMENT

### Day 1: Set Up Pre-commit Hooks

#### Install Husky (already done)

```bash
npm install --save-dev husky lint-staged
npx husky install
```

#### Configure Pre-commit

```bash
# .husky/pre-commit
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

# Run linting
npm run lint

# Run formatting
npm run format:check

# Check file sizes
node scripts/check-file-sizes.js

# Check for console.log
if git diff --cached --name-only | grep -E '\.(ts|js)$' | xargs grep -n 'console\.log'; then
  echo "Error: console.log found in staged files"
  exit 1
fi
```

---

## 📊 FINAL METRICS REPORT

After 4 weeks, generate final report:

```bash
# Run metrics script
node scripts/generate-metrics.js > KILO-FINAL-REPORT.md
```

**Expected Final State:**

- Total Files: <1,500 (from 5,239) = **-71%**
- Total Lines: <150,000 (from 719,543) = **-79%**
- Markdown Files: <50 (from 1,831) = **-97%**
- Config Files: <20 (from 697) = **-97%**
- Dependencies: <15 (from 11) = Maintained

---

## 🚨 ROLLBACK PLAN

If anything goes wrong:

```bash
# Rollback to pre-cleanup state
git reset --hard pre-kilo-cleanup

# Or rollback specific commits
git revert <commit-hash>

# Or restore specific files
git checkout pre-kilo-cleanup -- path/to/file
```

---

## ✅ SUCCESS CRITERIA

- [ ] All deletion targets met
- [ ] All consolidation complete
- [ ] All simplification done
- [ ] Enforcement mechanisms in place
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Team trained on new structure
- [ ] Metrics report generated

---

**REMEMBER:** Every line of code is a liability. Be ruthless. Show no mercy to complexity.
