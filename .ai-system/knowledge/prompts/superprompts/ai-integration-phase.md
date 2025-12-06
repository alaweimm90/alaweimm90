# Phase 3: AI System Integration

## Objective

Merge ai-tools/ into .ai/tools/ for unified AI orchestration management.

## Pre-Flight Checks

```bash
# Create backup
git checkout -b consolidation-phase-3
git branch backup-pre-phase-3

# Document current structure
tree ai-tools/ .ai/ -L 2 > phase3-before.txt
```

## Execution Steps

### Step 1: Move Source Code

```bash
git mv ai-tools/src/ .ai/tools/src/
```

### Step 2: Move Integrations

```bash
git mv ai-tools/integrations/ .ai/integrations/
```

### Step 3: Move Configuration

```bash
git mv ai-tools/package.json .ai/tools/package.json
git mv ai-tools/tsconfig.json .ai/tools/tsconfig.json
git mv ai-tools/README.md .ai/tools/README.md
```

### Step 4: Move Utilities

```bash
git mv ai-tools/utils/ .ai/tools/utils/
```

### Step 5: Cleanup

```bash
rmdir ai-tools/
```

## Path Updates Required

### Update package.json (root)

```json
{
  "scripts": {
    "ai:start": "tsx .ai/tools/orchestrator.ts start",
    "ai:complete": "tsx .ai/tools/orchestrator.ts complete",
    "ai:context": "tsx .ai/tools/orchestrator.ts context",
    "ai:sync": "tsx .ai/sync.ts",
    "ai:dashboard": "tsx .ai/dashboard.ts"
  }
}
```

### Update Import Paths

Search and replace in all files:

- `from 'ai-tools/` → `from '.ai/tools/`
- `from '../ai-tools/` → `from '../.ai/tools/`
- `import('ai-tools/` → `import('.ai/tools/`

### Update TypeScript Config

Update `tsconfig.json` paths:

```json
{
  "compilerOptions": {
    "paths": {
      "@ai/*": [".ai/tools/src/*"]
    }
  }
}
```

## Validation Tests

### Test 1: AI Orchestration

```bash
npm run ai:start
npm run ai:context
```

### Test 2: Integration Tests

```bash
npm run test -- ai-tools
```

### Test 3: Type Checking

```bash
npm run type-check
```

### Test 4: Build Process

```bash
cd .ai/tools/
npm install
npm run build
```

## Rollback Procedure

```bash
git checkout backup-pre-phase-3
git branch -D consolidation-phase-3
```

## Success Criteria

- [ ] All AI tools moved successfully
- [ ] Import paths updated
- [ ] AI orchestration works
- [ ] Integration tests pass
- [ ] Type checking passes
- [ ] Build succeeds
- [ ] Git history preserved

## Commit Message

```
feat: integrate AI tools into .ai directory

- Move ai-tools/src/ → .ai/tools/src/
- Move ai-tools/integrations/ → .ai/integrations/
- Move ai-tools/package.json → .ai/tools/package.json
- Update all import paths and TypeScript config
- Remove empty ai-tools/ folder

BREAKING CHANGE: AI tool paths updated
```
