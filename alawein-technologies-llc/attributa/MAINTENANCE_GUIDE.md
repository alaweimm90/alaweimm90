# Attributa Maintenance Guide

**Last Updated:** 2025-11-19
**Version:** 1.0

---

## Table of Contents

1. [Code Quality Standards](#code-quality-standards)
2. [Type Safety Best Practices](#type-safety-best-practices)
3. [ESLint & Linting Rules](#eslint--linting-rules)
4. [Security Vulnerability Management](#security-vulnerability-management)
5. [Performance Optimization](#performance-optimization)
6. [Testing Strategy](#testing-strategy)
7. [Git Workflow](#git-workflow)
8. [Development Checklist](#development-checklist)
9. [Common Issues & Solutions](#common-issues--solutions)
10. [Resources](#resources)

---

## Code Quality Standards

### Quality Metrics (Target)

| Metric | Target | Status |
|--------|--------|--------|
| ESLint Errors | 0 | ✅ Met |
| ESLint Warnings | ≤20 | ✅ Met (18) |
| TypeScript Errors | 0 | ✅ Met |
| Security Issues | 0 | ✅ Met |
| Test Coverage | >80% | ⚠️ Monitor |
| Build Time | <25s | ✅ Met (17.96s) |

### Code Organization

**Directory Structure:**
```
src/
├── pages/           # Route pages (lazy-loaded)
├── components/      # Reusable UI components
│   ├── ui/         # Core UI primitives (shadcn-ui)
│   ├── attribution/ # Attribution visualization
│   ├── results/    # Result display components
│   ├── ai/         # AI assistant components
│   └── dev/        # Development helpers
├── lib/            # Core algorithms & utilities
│   ├── nlp/        # Natural language processing
│   ├── citations/  # Citation management
│   ├── code/       # Code security scanning
│   └── watermark/  # Watermark detection
├── services/       # API layer & external services
├── hooks/          # Custom React hooks
├── store/          # Zustand state management
├── types/          # TypeScript type definitions
└── assets/         # Static images & icons
```

### Naming Conventions

**Files:**
- Components: PascalCase (e.g., `AttributionPanel.tsx`)
- Utilities: camelCase (e.g., `pdfExtractor.ts`)
- Tests: `*.test.ts` or `*.spec.ts`
- Styles: Inline Tailwind CSS (prefer over CSS files)

**Variables:**
- Constants: UPPER_SNAKE_CASE
- Functions: camelCase
- Types/Interfaces: PascalCase
- Boolean variables: prefix with `is`, `has`, `can`, or `should`

**Example:**
```typescript
// Good
const MAX_FILE_SIZE = 50 * 1024 * 1024;
function processTextContent(text: string): string[]
interface AnalysisResult { score: number; }
const isLoading = false;

// Avoid
const max_file_size = 50MB;
function processtextcontent(text)
type analysis_result = { score: number; }
```

---

## Type Safety Best Practices

### Rule 1: Never Use `any`

❌ **Bad:**
```typescript
function processData(data: any): any {
  return data.transform();
}
```

✅ **Good:**
```typescript
interface DataPayload {
  transform(): unknown;
}

function processData(data: DataPayload): unknown {
  return data.transform();
}
```

### Rule 2: Use `unknown` Instead of `any`

When the type is truly unknown (e.g., error handling):

```typescript
// Good: Requires type checking
try {
  // ...
} catch (error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
}

// Avoid: Doesn't require type checking
try {
  // ...
} catch (error: any) {
  console.error(error.message); // Runtime error if error is not Error
}
```

### Rule 3: Create Proper Interfaces

For API responses and complex data structures:

```typescript
// API Response Interface
interface Attribution {
  id: string;
  artifact_id: string;
  confidence_score: number;
  confidence_level: 'Low' | 'Medium' | 'High';
  signals: Record<string, unknown>;
  created_at: string;
}

// Component Props Interface
interface AttributionPanelProps {
  attribution: Attribution;
  onSelect?: (id: string) => void;
  variant?: 'compact' | 'detailed';
}
```

### Rule 4: Use Type Narrowing

Instead of type assertions, use type narrowing:

```typescript
// Good: Type narrowing
if (error instanceof Error) {
  console.error(error.message);
} else {
  console.error(String(error));
}

// Avoid: Type assertion
console.error((error as Error).message);
```

### Rule 5: Proper Generic Usage

For reusable functions and components:

```typescript
// Generic function
function processList<T>(items: T[], handler: (item: T) => T): T[] {
  return items.map(handler);
}

// Generic component
interface DataDisplayProps<T> {
  data: T[];
  renderItem: (item: T) => React.ReactNode;
}

function DataDisplay<T>({ data, renderItem }: DataDisplayProps<T>) {
  return <div>{data.map(renderItem)}</div>;
}
```

### Rule 6: Record Types for Dynamic Objects

```typescript
// Good: Unknown keys with unknown values
const metadata: Record<string, unknown> = {
  customField: 'value',
  anotherField: 123
};

// Better: Specific value types
const settings: Record<string, string | number | boolean> = {
  theme: 'dark',
  fontSize: 14,
  autoSave: true
};
```

### Rule 7: Union Types for Variants

```typescript
// Good: Explicit variants
type ConfidenceLevel = 'Low' | 'Medium' | 'High';
type SourceType = 'file' | 'url' | 'text' | 'github';

// Use in interfaces
interface Source {
  type: SourceType;
  title?: string;
}
```

---

## ESLint & Linting Rules

### Running Linting

```bash
# Check all files
npm run lint

# Check specific file
npx eslint src/components/MyComponent.tsx

# Fix auto-fixable errors
npx eslint --fix src/
```

### Common ESLint Errors

#### 1. `@typescript-eslint/no-explicit-any`

**Error:**
```
Unexpected any. Specify a different type
```

**Fix:**
```typescript
// Bad
const value: any = getData();

// Good
const value: unknown = getData();
// or with proper typing
interface DataResponse { success: boolean; }
const value: DataResponse = getData();
```

#### 2. `react-hooks/exhaustive-deps`

**Warning:**
```
React Hook useEffect has a missing dependency
```

**Fix 1: Add dependency**
```typescript
useEffect(() => {
  console.log(data);
}, [data]); // Added 'data' to dependencies
```

**Fix 2: Use useCallback to stabilize reference**
```typescript
const handleClick = useCallback(() => {
  console.log(data);
}, [data]); // Dependency managed by useCallback

useEffect(() => {
  handleClick();
}, [handleClick]); // Single dependency
```

#### 3. `no-empty`

**Error:**
```
Empty block statement
```

**Fix:**
```typescript
// Bad
try {
  // code
} catch (e) {
}

// Good
try {
  // code
} catch (e) {
  console.debug('Expected error:', e);
}
```

#### 4. `react-refresh/only-export-components`

**Warning:**
```
Fast refresh only works when exporting components
```

**Fix: Separate constants from component exports**
```typescript
// Bad - constants with component export
export const COLORS = { primary: '#blue' };
export function Button() { ... }

// Good - separate files
// constants.ts
export const COLORS = { primary: '#blue' };

// Button.tsx
import { COLORS } from './constants';
export function Button() { ... }
```

---

## Security Vulnerability Management

### Regular Security Audits

```bash
# Check for vulnerabilities
npm audit

# Fix vulnerabilities automatically (caution: may have breaking changes)
npm audit fix

# Force fix (use with caution)
npm audit fix --force
```

### Dependency Management

**Best Practices:**

1. **Keep Dependencies Updated**
   ```bash
   npm outdated
   npm update
   ```

2. **Review Breaking Changes**
   - Check CHANGELOG before major version updates
   - Test thoroughly after updates
   - Update lock file: `npm ci --package-lock-only`

3. **Security Policy**
   - Monitor GitHub security advisories
   - Enable dependabot for automatic PRs
   - Review dependencies quarterly

### Secrets Management

⚠️ **Never commit secrets!**

**Allowed:**
- Environment variables in `.env` (add to `.gitignore`)
- API keys from environment (process.env)
- Supabase configuration (via Supabase setup)

**Check for secrets:**
```bash
# Use gitleaks or similar tools
npx gitleaks detect --source . --verbose
```

---

## Performance Optimization

### Bundle Analysis

```bash
# Generate bundle analysis
npm run build

# Check bundle sizes
du -sh dist/assets/*.js
```

**Target Sizes (gzipped):**
- Main bundle: <200KB
- Route chunks: <50KB each
- Vendor bundles: <100KB each

### Code Splitting Strategy

**Automatic (Vite handles):**
- Route-based code splitting
- Vendor chunk extraction
- Manual chunks for large dependencies

**Manual chunks defined in vite.config.ts:**
```typescript
manualChunks: {
  'transformers': ['@huggingface/transformers'],
  'pdf': ['pdfjs-dist'],
  'charts': ['recharts'],
  'ui': ['@radix-ui/*'],
  'vendor': ['react', 'react-dom']
}
```

### Lazy Loading Components

```typescript
import { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

export function MyPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HeavyComponent />
    </Suspense>
  );
}
```

### Performance Monitoring

Check Web Vitals:
```typescript
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

getCLS(console.log);
getFID(console.log);
getFCP(console.log);
getLCP(console.log);
getTTFB(console.log);
```

---

## Testing Strategy

### Test Structure

```
tests/
├── scoring.test.ts          # Scoring algorithm tests
├── gltr.test.ts             # GLTR analysis tests
├── detectgpt.test.ts        # DetectGPT tests
├── citations.test.ts        # Citation validation
├── codeAnalysis.test.ts     # Code scanning
├── integration.test.ts      # API integration
└── accessibility.test.ts    # a11y compliance
```

### Running Tests

```bash
# Run all tests
npm run test

# Run specific test file
npx jest tests/scoring.test.ts

# Run with coverage
npm run test -- --coverage

# Watch mode
npm run test -- --watch
```

### E2E Testing

```bash
# Run E2E tests
npm run e2e

# Run specific test
npx playwright test tests/example.spec.ts

# Debug mode
npx playwright test --debug
```

### Test Guidelines

**Good test characteristics:**
- ✅ Isolated (no external dependencies)
- ✅ Fast (< 1 second for unit tests)
- ✅ Reliable (same result every run)
- ✅ Clear (easy to understand intent)

**Test structure:**
```typescript
describe('AttributionAnalyzer', () => {
  it('should calculate confidence score correctly', () => {
    // Arrange
    const input = { similarity: 0.8, gltrScore: 0.7 };

    // Act
    const result = analyzer.computeScore(input);

    // Assert
    expect(result).toBeCloseTo(0.75, 2);
  });
});
```

---

## Git Workflow

### Branch Naming

**Pattern:** `<type>/<description>-<session-id>`

Examples:
- `feature/add-pdf-upload-abc123`
- `fix/security-issue-def456`
- `refactor/improve-typing-ghi789`
- `docs/update-readme-jkl012`

### Commit Messages

**Format:** Follow conventional commits

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code reorganization
- `docs:` Documentation changes
- `test:` Test additions/updates
- `chore:` Build/dependency updates
- `perf:` Performance improvements

**Example:**
```
fix(types): replace any types with proper interfaces

- Replace 139 `any` types with typed interfaces
- Improve error handling with unknown type
- Create interfaces for NLP models and data
- Result: Full TypeScript compliance

Fixes #45
```

### Pre-Commit Checklist

Before committing:
- [ ] Code passes linting: `npm run lint`
- [ ] All tests pass: `npm run test`
- [ ] Types compile: `npx tsc --noEmit`
- [ ] No secrets committed: `npx gitleaks`
- [ ] Commit message follows convention
- [ ] Changes are logical and focused

### Pull Request Checklist

Before creating PR:
- [ ] Feature is complete and tested
- [ ] Passes full CI pipeline
- [ ] Documentation updated if needed
- [ ] No breaking changes (or documented)
- [ ] Backward compatible
- [ ] Code review checklist passed

---

## Development Checklist

### Before Starting Development

- [ ] Pull latest main: `git pull origin main`
- [ ] Create feature branch: `git checkout -b feature/your-feature`
- [ ] Install dependencies: `npm install`
- [ ] Run tests: `npm test`
- [ ] Verify build: `npm run build`

### During Development

- [ ] Follow code organization guidelines
- [ ] Use TypeScript with proper types
- [ ] Write tests for new functionality
- [ ] Update JSDoc comments
- [ ] Keep commits small and focused
- [ ] Run linting frequently: `npm run lint --fix`

### Before Submitting PR

- [ ] Run all checks: `npm run lint && npm test && npm run build`
- [ ] Verify in development: `npm run dev`
- [ ] Update CHANGELOG if needed
- [ ] Create comprehensive PR description
- [ ] Link related issues

### Code Review Checklist

**Reviewer should verify:**
- [ ] Code follows style guide
- [ ] No console.log left behind
- [ ] Proper error handling
- [ ] No security issues
- [ ] Types are correct
- [ ] Tests cover changes
- [ ] Performance acceptable
- [ ] Accessibility considerations
- [ ] Documentation updated

---

## Common Issues & Solutions

### Issue 1: ESLint "Unexpected any" Error

**Problem:** Type is `any` but I don't know the real type

**Solution:**
```typescript
// Option 1: Use unknown and type guard
function process(data: unknown): void {
  if (typeof data === 'object' && data !== null && 'property' in data) {
    // Now TypeScript knows data has 'property'
  }
}

// Option 2: Use proper interface
interface MyData { property: string; }
function process(data: MyData): void { }

// Option 3: Use type assertion carefully
function process(data: unknown): void {
  const myData = data as MyData; // Document why
}
```

### Issue 2: Build Time Exceeds 25 Seconds

**Problem:** Production build is slow

**Investigation:**
```bash
# Check bundle size
npm run build
du -sh dist/assets/*.js | sort -rh

# Analyze dependencies
npm ls --depth=0 --all

# Check for unused imports
npx unused-exports
```

**Solutions:**
- Remove unused dependencies
- Enable code splitting for large libraries
- Lazy load non-critical components
- Optimize images and assets

### Issue 3: Test Failures After Changes

**Problem:** Tests fail after code modification

**Debug:**
```bash
# Run failed test with details
npx jest tests/failing.test.ts --verbose

# Watch mode for debugging
npx jest tests/failing.test.ts --watch

# Debug in Chrome DevTools
node --inspect-brk ./node_modules/.bin/jest --runInBand
```

### Issue 4: TypeScript Strict Mode Errors

**Problem:** Enabling `strict: true` breaks compilation

**Approach:**
- Enable one flag at a time
- Start with `strict: false` with specific rules
- Gradually increase strictness
- Update type definitions as needed

**Current settings (intentionally relaxed for UI development):**
```json
{
  "compilerOptions": {
    "noImplicitAny": false,
    "strictNullChecks": false,
    "noUnusedParameters": false
  }
}
```

**To increase strictness:**
1. Enable `noImplicitAny: true`
2. Fix all implicit any errors
3. Enable `strictNullChecks: true`
4. Fix null/undefined handling
5. Proceed with other strict options

---

## Resources

### Documentation
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [React Documentation](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [shadcn-ui Components](https://ui.shadcn.com/)

### Tools
- [ESLint Rules](https://eslint.org/docs/rules/)
- [TypeScript ESLint](https://typescript-eslint.io/)
- [Jest Testing](https://jestjs.io/)
- [Playwright E2E](https://playwright.dev/)

### Project Specific
- [CLAUDE.md](./CLAUDE.md) - Project overview
- [CLEANUP_SUMMARY.md](./CLEANUP_SUMMARY.md) - Recent improvements
- [.eslintrc.js](./.eslintrc.js) - Linting configuration
- [tsconfig.json](./tsconfig.json) - TypeScript configuration

### Community
- [GitHub Issues](https://github.com/alaweimm90-archieve/Attributa/issues)
- [Discussions](https://github.com/alaweimm90-archieve/Attributa/discussions)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-19 | Initial comprehensive guide, post-cleanup |

---

## Questions & Feedback

For questions about maintaining code quality:
1. Check the relevant section above
2. Review recent commits for examples
3. Run linting and tests to verify
4. Open an issue with specific question

---

**Last Updated:** 2025-11-19
**Maintained by:** Development Team
**Status:** Current
