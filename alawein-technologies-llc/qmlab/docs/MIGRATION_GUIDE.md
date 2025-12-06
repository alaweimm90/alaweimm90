# Migration Guide

## Repository Hygiene Refactor (2025)

This guide helps you understand the recent repository structure changes and how to adapt your code.

### Overview

The repository underwent a comprehensive hygiene pass to improve:
- Code organization and structure
- Style system consistency and configurability
- Tooling and automation
- Documentation clarity
- Developer experience

### Breaking Changes

#### 1. Logging

**Before:**
```typescript
console.log('User action:', data);
console.error('Error occurred:', error);
```

**After:**
```typescript
import { logger } from '@/lib/logger';

logger.info('User action', { data });
logger.error('Error occurred', { error });
```

**Migration:** Replace `console.log/info/warn/error` with `logger` methods. Console output still works in development but is now structured and filterable.

#### 2. ESLint Rules

**Changes:**
- `no-unused-vars` now warns for unused variables (use `_` prefix for intentionally unused)
- `no-console` warns by default (use logger instead)
- Import ordering is now enforced

**Migration:** 
```bash
# Auto-fix most issues
npm run lint:fix

# For unused vars, prefix with underscore
const handleClick = (_event) => { ... }
```

#### 3. File Naming (Gradual Migration)

**Target conventions:**
- Components: `PascalCase.tsx` (e.g., `ButtonPrimary.tsx`)
- Hooks: `use-name.tsx` (e.g., `use-mobile.tsx`)
- Utils: `kebab-case.ts` (e.g., `format-date.ts`)
- Tests: `*.test.ts(x)` or `*.spec.ts(x)`

**Migration:** Files will be renamed gradually. Update imports as you encounter them.

### New Features

#### 1. Centralized Theme Configuration

**Location:** `src/config/theme.config.ts`

**Usage:**
```typescript
import { activeTheme } from '@/config/theme.config';

// Access theme values
const primaryColor = activeTheme.colors.primary;
```

**Customization:** Edit `theme.config.ts` to change entire app theme. See [Theme Customization Guide](./THEME_CUSTOMIZATION.md).

#### 2. Design System Structure

**New locations:**
- Colors: `src/styles/design-system/colors.css`
- Effects: `src/styles/design-system/effects.css`
- Animations: `src/styles/design-system/animations.css`

**Legacy (still supported):**
- `src/styles/tokens.css` - Will be gradually migrated

**Usage:** No changes needed. Imports are handled automatically in `index.css`.

#### 3. Documentation

**New location:** `docs/` directory

**Available guides:**
- Architecture overview
- Development guide
- Theme customization
- Migration guide (this document)

### Updated Workflows

#### Git Commits

**Now enforced:** Conventional Commits format

```bash
# Valid commits
git commit -m "feat(circuit): Add quantum gate library"
git commit -m "fix(bloch): Correct rotation calculation"
git commit -m "docs(readme): Update setup instructions"

# Invalid (will be rejected)
git commit -m "fixed bug"
git commit -m "updates"
```

#### Pre-commit Checks

**Automatic on commit:**
1. Prettier formatting
2. ESLint fixing
3. Stylelint CSS linting

**Manual run:**
```bash
npm run format
npm run lint:fix
npm run style:lint
```

#### CI/CD

**New GitHub Actions workflow:**
- Runs on PRs and main branch
- Checks: lint, typecheck, format, stylelint, build
- Accessibility tests included

### Non-Breaking Changes

#### Project Structure

**New:**
- `docs/` - Documentation
- `src/config/` - Configuration files
- `src/styles/design-system/` - Core design tokens
- `.husky/` - Git hooks
- `.github/workflows/` - CI/CD workflows

**Unchanged:**
- `src/components/` - Components (including ui/)
- `src/hooks/` - Custom hooks
- `src/lib/` - Utilities
- `src/pages/` - Page components
- `src/styles/` - Styles (with new design-system/ subfolder)

#### Dependencies

**New dev dependencies:**
- `husky` - Git hooks
- `lint-staged` - Pre-commit linting
- `@commitlint/*` - Commit message linting
- `stylelint` - CSS linting
- `eslint-plugin-import` - Import ordering

**No changes to production dependencies.**

### Rollback Instructions

If you need to temporarily disable new tooling:

#### Disable Git Hooks
```bash
# Temporarily bypass
git commit --no-verify -m "message"

# Permanently disable
rm -rf .husky
```

#### Disable ESLint Rules
```javascript
// In eslint.config.js, temporarily turn off:
"no-console": "off",
"import/order": "off",
```

#### Use Old Logger
```typescript
// Temporarily OK (will trigger lint warning)
console.log('Still works in development');
```

### Getting Help

**Issues with migration?**
1. Check this guide first
2. Review [Development Guide](./DEVELOPMENT.md)
3. Search [GitHub Issues](https://github.com/your-org/qmlab/issues)
4. Ask in [Discussions](https://github.com/your-org/qmlab/discussions)

**Found a bug?**
[Open an issue](https://github.com/your-org/qmlab/issues/new) with:
- What you tried
- What happened
- What you expected
- Migration guide section reference

### Timeline

- **Phase 1** (Current): Infrastructure and tooling
- **Phase 2** (Next): Gradual file renaming
- **Phase 3** (Future): Style system consolidation
- **Phase 4** (Future): Documentation migration completion

### Feedback

We'd love your feedback on these changes!

- What's working well?
- What's causing friction?
- What documentation is missing?

Share in [GitHub Discussions](https://github.com/your-org/qmlab/discussions).
