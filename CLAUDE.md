# Claude Code Instructions

## Repository Overview

Meta-governance repository with DevOps templates, governance tools, and AI coding documentation.

## Protected Files Policy

**CRITICAL: Before modifying any file listed below, you MUST:**

1. Read `.metaHub/policies/protected-files.yaml` first
2. Confirm with user before making changes
3. For README.md specifically: Only modify if user says "update README" or "edit README"

### Strict Protection (Never modify without explicit request)

- `README.md` - Personal profile README with custom animations
- `LICENSE`
- `CODEOWNERS`
- `.github/workflows/*.yml`
- `.metaHub/policies/*.yaml`

### Conditional (Can modify if task requires it)

- `package.json` - Only for adding/removing dependencies
- `tsconfig.json` - Only for TypeScript config changes
- `.gitignore` - Only when adding new ignore patterns

### Forbidden (Never modify)

- `.env*` files
- `*.key`, `*.pem` files
- `**/secrets/**`

## AI Orchestration Context

**Before starting any task, check these files for context:**

1. `.ai/context.yaml` - Current AI configuration and routing
2. `docs/CODEMAP.md` - Interactive system architecture diagrams
3. `docs/ARCHITECTURE.md` - Detailed system design
4. `.ai/task-history.json` - Similar past tasks and patterns

**Task Tracking:**

```bash
# Start tracking a task
npm run ai:start feature auth,api "Add OAuth authentication"

# Get context for a task type
npm run ai:context feature auth

# Complete and log a task
npm run ai:complete true "file1.ts,file2.ts" 150 20 5 "Notes"

# View metrics
npm run ai:metrics
```

**Routing Preferences:**

- Complex features → Claude Code
- Refactoring → Kilo Code
- Quick fixes → Copilot
- Code review → Claude Code

## Code Style

- TypeScript with ES modules (no CommonJS require())
- Prettier for formatting
- ESLint for linting
- Run `npm run lint` before committing

## Testing

- Vitest for unit tests
- Run `npm test` before committing

## Commit Conventions

- Use conventional commits: `feat:`, `fix:`, `docs:`, `chore:`
- Pre-commit hooks will run automatically
