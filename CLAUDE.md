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
