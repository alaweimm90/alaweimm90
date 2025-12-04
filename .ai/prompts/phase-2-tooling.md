# Phase 2: Tooling Consolidation

## Objective

Merge tools/ and scripts/ into .metaHub/tools/ for unified CLI and automation management.

## Pre-Flight Checks

```bash
# Create backup
git checkout -b consolidation-phase-2
git branch backup-pre-phase-2

# Document current structure
tree tools/ scripts/ .metaHub/tools/ -L 2 > phase2-before.txt
```

## Execution Steps

### Step 1: AI Tools

```bash
git mv tools/ai/ .metaHub/tools/ai/
```

### Step 2: Atlas Tools

```bash
git mv tools/atlas/ .metaHub/tools/atlas/
```

### Step 3: CLI Tools

```bash
git mv tools/cli/ .metaHub/tools/cli/
```

### Step 4: DevOps Tools

```bash
git mv tools/devops/ .metaHub/tools/devops/
```

### Step 5: Security Tools

```bash
git mv tools/security/ .metaHub/tools/security/
```

### Step 6: Archive Legacy Tools

```bash
git mv tools/legacy/ .archive/tools-legacy/
```

### Step 7: Move Scripts

```bash
git mv scripts/check-file-sizes.cjs .metaHub/scripts/
git mv scripts/check-protected-files.sh .metaHub/scripts/
git mv scripts/generate-codemap.ts .metaHub/scripts/
git mv scripts/kilo-cleanup.ps1 .metaHub/scripts/
```

### Step 8: Cleanup

```bash
rmdir tools/
rmdir scripts/
```

## Path Updates Required

### Update package.json

```json
{
  "scripts": {
    "devops": "tsx .metaHub/tools/cli/devops.ts",
    "atlas": "tsx .metaHub/tools/atlas/cli/index.ts",
    "ai": "tsx .metaHub/tools/ai/orchestrator.ts",
    "codemap": "tsx .metaHub/scripts/generate-codemap.ts"
  }
}
```

### Update Import Paths

Search and replace in all TypeScript/JavaScript files:

- `from '../tools/` → `from '../.metaHub/tools/`
- `from '../../tools/` → `from '../../.metaHub/tools/`
- `require('../tools/` → `require('../.metaHub/tools/`

### Update Python Imports

Search and replace in all Python files:

- `from tools.` → `from .metaHub.tools.`
- `import tools.` → `import .metaHub.tools.`

## Validation Tests

### Test 1: CLI Commands

```bash
npm run devops -- --help
npm run atlas -- --help
npm run ai -- --help
```

### Test 2: Script Execution

```bash
npm run codemap
node .metaHub/scripts/check-file-sizes.cjs
```

### Test 3: Import Resolution

```bash
npm run type-check
```

### Test 4: Test Suite

```bash
npm test
pytest
```

## Rollback Procedure

```bash
git checkout backup-pre-phase-2
git branch -D consolidation-phase-2
```

## Success Criteria

- [ ] All tools moved successfully
- [ ] All scripts moved successfully
- [ ] CLI commands work
- [ ] Import paths updated
- [ ] Type checking passes
- [ ] Test suite passes
- [ ] Git history preserved

## Commit Message

```
feat: consolidate tooling into .metaHub

- Move tools/ai/ → .metaHub/tools/ai/
- Move tools/atlas/ → .metaHub/tools/atlas/
- Move tools/cli/ → .metaHub/tools/cli/
- Move tools/devops/ → .metaHub/tools/devops/
- Move tools/security/ → .metaHub/tools/security/
- Move scripts/*.{ts,py,sh,ps1} → .metaHub/scripts/
- Archive tools/legacy/ → .archive/tools-legacy/
- Update all import paths and package.json scripts
- Remove empty tools/ and scripts/ folders

BREAKING CHANGE: Tool paths updated
```
