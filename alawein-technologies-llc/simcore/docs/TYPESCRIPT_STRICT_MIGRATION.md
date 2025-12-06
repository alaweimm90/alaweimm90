# TypeScript Strict Mode Migration Guide

## Overview

SimCore currently has TypeScript strict mode disabled in `tsconfig.app.json`. This document outlines a phased approach to enable strict type checking across the 342 TypeScript files in the codebase.

## Current State (2025-11-19)

### Files Affected
- **Total TypeScript files:** 342
- **Test files:** ~40+ (excluded from initial migration)
- **Core library files:** ~45 in `/src/lib/`
- **Component files:** ~194 UI components
- **Pages:** 34 simulation pages

### Current Configuration (tsconfig.app.json)

```json
{
  "strict": false,
  "noUnusedLocals": false,
  "noUnusedParameters": false,
  "noImplicitAny": false,
  "noFallthroughCasesInSwitch": false
}
```

### Issues This Causes

1. **No type safety on function parameters** - `noImplicitAny: false` allows `any` types everywhere
2. **Null/undefined bugs** - `strictNullChecks: false` allows null/undefined to slip through
3. **Dead code accumulation** - `noUnusedLocals: false` allows unused variables
4. **Function signature mismatches** - `strictFunctionTypes: false` allows unsafe function assignments

## Migration Strategy

### Phase 1: Enable Incremental Strict Mode (Week 1)

Use `tsconfig.strict.json` for new code and refactored modules.

**Test the configuration:**
```bash
# Test strict mode on entire codebase (expect errors)
npx tsc --project tsconfig.strict.json --noEmit

# Count errors by category
npx tsc --project tsconfig.strict.json --noEmit 2>&1 | grep "error TS" | cut -d"'" -f2 | sort | uniq -c | sort -rn
```

**Expected error categories:**
- `TS7006` - Parameter implicitly has 'any' type
- `TS2345` - Argument not assignable to parameter
- `TS2531` - Object is possibly 'null'
- `TS2532` - Object is possibly 'undefined'
- `TS18048` - May be 'undefined'
- `TS6133` - Variable declared but never used

### Phase 2: Fix High-Priority Modules (Weeks 2-3)

Fix critical modules in this order:

1. **Physics Core** (`/src/lib/physics-*.ts`)
   - `physics-base.ts`
   - `physics-utils.ts`
   - `physics-validation.ts`

2. **WebGPU & Performance** (`/src/lib/webgpu-*.ts`, `/src/lib/performance-*.ts`)
   - `webgpu-manager.ts`
   - `webgpu-physics.ts`
   - `performance-optimization.ts`

3. **Worker System** (`/src/lib/webworker-physics-engine.ts`)

4. **Shared Utilities** (`/src/lib/utils.ts`, `/src/lib/input-validation.ts`)

### Phase 3: Fix UI Components (Weeks 4-5)

Work through components by domain:

1. **Core UI Components** (`/src/components/ui/`)
2. **Hooks** (`/src/hooks/`)
3. **Simulation Pages** (`/src/pages/`)

### Phase 4: Enable Strict Mode (Week 6)

Once all errors are fixed:

```bash
# Verify no errors
npx tsc --project tsconfig.strict.json --noEmit

# Update tsconfig.app.json
# Set strict: true and enable all strict options
```

## Common Fixes

### 1. Implicit Any Parameters

**Before:**
```typescript
function calculateEnergy(mass, velocity) {
  return 0.5 * mass * velocity ** 2;
}
```

**After:**
```typescript
function calculateEnergy(mass: number, velocity: number): number {
  return 0.5 * mass * velocity ** 2;
}
```

### 2. Null/Undefined Checks

**Before:**
```typescript
function getElement(id: string) {
  const el = document.getElementById(id);
  el.textContent = "Hello"; // Error: el might be null
}
```

**After:**
```typescript
function getElement(id: string): void {
  const el = document.getElementById(id);
  if (!el) {
    console.error(`Element ${id} not found`);
    return;
  }
  el.textContent = "Hello";
}
```

### 3. Optional Chaining & Nullish Coalescing

**Before:**
```typescript
function getUserName(user: User | undefined) {
  return user && user.profile && user.profile.name || "Anonymous";
}
```

**After:**
```typescript
function getUserName(user: User | undefined): string {
  return user?.profile?.name ?? "Anonymous";
}
```

### 4. Type Assertions

**Before:**
```typescript
const data = JSON.parse(str);
data.forEach(item => console.log(item.name)); // Error: data is 'any'
```

**After:**
```typescript
interface DataItem {
  name: string;
  value: number;
}

const data = JSON.parse(str) as DataItem[];
data.forEach(item => console.log(item.name));
```

### 5. Unused Variables

**Before:**
```typescript
function process(data: Data) {
  const temp = transform(data); // Warning: 'temp' is never used
  return data;
}
```

**After (Option 1 - Remove):**
```typescript
function process(data: Data): Data {
  return data;
}
```

**After (Option 2 - Use it):**
```typescript
function process(data: Data): Data {
  const temp = transform(data);
  return temp;
}
```

**After (Option 3 - Acknowledge intent):**
```typescript
function process(data: Data): Data {
  const _temp = transform(data); // Prefix with _ to indicate intentionally unused
  return data;
}
```

## Tools & Scripts

### Check Specific File

```bash
npx tsc --noEmit --strict src/lib/physics-base.ts
```

### Count Errors by File

```bash
npx tsc --project tsconfig.strict.json --noEmit 2>&1 | \
  grep "src/" | \
  cut -d"(" -f1 | \
  sort | uniq -c | \
  sort -rn | \
  head -20
```

### Test Build Still Works

```bash
npm run typecheck  # Uses current (non-strict) config
npm run build      # Verify production build works
```

## Automation Strategy

### ESLint Configuration

Add to `eslint.config.js`:

```javascript
{
  rules: {
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/explicit-function-return-type': 'warn',
    '@typescript-eslint/no-unused-vars': ['warn', {
      argsIgnorePattern: '^_',
      varsIgnorePattern: '^_'
    }],
    '@typescript-eslint/strict-boolean-expressions': 'warn'
  }
}
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: typescript-strict
      name: TypeScript Strict Check (New Files)
      entry: bash -c 'git diff --cached --name-only --diff-filter=ACM | grep -E "\\.(ts|tsx)$" | xargs npx tsc --noEmit --strict || true'
      language: system
      pass_filenames: false
```

## Progress Tracking

Track migration progress with:

```bash
# Create baseline
npx tsc --project tsconfig.strict.json --noEmit 2>&1 | \
  grep "Found" | \
  tee migration-baseline.txt

# Compare after fixes
npx tsc --project tsconfig.strict.json --noEmit 2>&1 | \
  grep "Found"
```

## Module-by-Module Checklist

### Physics Core Modules
- [ ] `physics-base.ts`
- [ ] `physics-utils.ts`
- [ ] `physics-validation.ts`
- [ ] `physics-memory-manager.ts`
- [ ] `enhanced-physics.ts`

### WebGPU & Compute
- [ ] `webgpu-manager.ts`
- [ ] `webgpu-physics.ts`
- [ ] `webgpu-shaders.ts`
- [ ] `webworker-physics-engine.ts`

### Performance & Optimization
- [ ] `performance-optimization.ts`
- [ ] `performance-monitoring.ts`
- [ ] `performance-utils.ts`

### Domain-Specific Physics
- [ ] `quantum-tunneling-optimized.ts`
- [ ] `graphene-physics.ts`
- [ ] `graphene-tight-binding.ts`
- [ ] `llg-store.ts`
- [ ] `llg-numerics.ts`

### Utilities
- [ ] `utils.ts`
- [ ] `input-validation.ts`
- [ ] `error-handling.ts`
- [ ] `error-reporting.ts`

### UI Infrastructure
- [ ] `accessibility-utils.ts`
- [ ] `image-optimization.ts`
- [ ] `pwa-utils.ts`

## Success Metrics

1. **Zero TypeScript errors** with strict mode enabled
2. **No `any` types** except explicitly documented cases
3. **100% type coverage** on function signatures
4. **No unused code** warnings
5. **Build time** remains under 30 seconds

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Setup | Week 1 | `tsconfig.strict.json` created, baseline established |
| Phase 2: Core Modules | Weeks 2-3 | Physics & WebGPU modules strict-compliant |
| Phase 3: UI Components | Weeks 4-5 | All components strict-compliant |
| Phase 4: Enable | Week 6 | `tsconfig.app.json` updated, strict mode enabled |

## Resources

- [TypeScript Handbook - Strict Mode](https://www.typescriptlang.org/tsconfig#strict)
- [TypeScript Deep Dive - Strict](https://basarat.gitbook.io/typescript/intro-1/strict)
- [Effective TypeScript - Item 3: Understand Strict](https://effectivetypescript.com/)

## Contact

For questions or assistance with migration:
- **Owner:** Meshal Alawein (meshal@berkeley.edu)
- **Project:** SimCore - Interactive Scientific Computing Laboratory

---

**Last Updated:** 2025-11-19
**Status:** Phase 1 - Setup Complete
