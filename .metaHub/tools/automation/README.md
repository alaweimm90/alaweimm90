# 🤖 Automation Infrastructure

**Welcome to the GitHub Monorepo Automation Hub!**

This directory contains all automation scripts, git hooks, task configurations, and external tool integrations for the monorepo.

---

## 📁 Directory Structure

```
.automation/
├── scripts/           # Shell automation scripts
├── hooks/             # Git hook templates
├── tasks/             # Task runner configurations
├── integrations/      # External tool integrations
└── README.md         # This file
```

---

## 🚀 Quick Start

### 1. Initial Setup

Run the dev setup script to configure your environment:

```bash
bash .automation/scripts/dev-setup.sh
```

This will:

- ✅ Check prerequisites (Node.js, pnpm, git)
- ✅ Install dependencies
- ✅ Set up git hooks
- ✅ Configure git settings
- ✅ Run health check
- ✅ Optionally install external tools

### 2. Install Git Hooks

If you skipped the setup script, manually install hooks:

```bash
# Copy hooks to .husky
cp .automation/hooks/* .husky/

# Make them executable (Unix/Mac)
chmod +x .husky/*
```

### 3. Verify Installation

Run the health check:

```bash
bash .automation/scripts/health-check.sh
```

---

## 📜 Available Scripts

### Core Scripts

#### `dev-setup.sh`

**Complete development environment setup**

```bash
bash .automation/scripts/dev-setup.sh
```

Sets up everything you need to start developing.

#### `health-check.sh`

**Comprehensive repository health diagnostic**

```bash
bash .automation/scripts/health-check.sh
```

Checks:

- Repository structure
- Dependencies
- Git configuration
- Environment variables
- TypeScript configuration
- Build system
- Testing infrastructure
- Linting setup
- Security
- Documentation
- CI/CD
- Automation

#### `security-scan.sh`

**Local security scanning**

```bash
bash .automation/scripts/security-scan.sh
```

Scans for:

- Secrets and credentials
- Hardcoded passwords
- Dependency vulnerabilities
- Unsafe code patterns
- Security misconfigurations
- Docker security issues
- File permissions

#### `fix-all.sh`

**Auto-fix linting, formatting, and common issues**

```bash
bash .automation/scripts/fix-all.sh
```

Automatically fixes:

- Prettier formatting
- ESLint issues
- Import organization
- package.json sorting
- Line endings
- File permissions

#### `run-tests.sh`

**Intelligent test runner**

```bash
# Run tests for changed files
bash .automation/scripts/run-tests.sh --changed

# Run all tests
bash .automation/scripts/run-tests.sh --all

# Watch mode
bash .automation/scripts/run-tests.sh --watch

# With coverage
bash .automation/scripts/run-tests.sh --coverage
```

---

## 🔗 Git Hooks

Git hooks enforce quality and security gates automatically.

### Pre-Commit Hook

**Runs before every commit**

Checks:

1. 🔍 Secret scanning (Gitleaks if available)
2. 📦 File size limits (max 5MB)
3. 📝 TypeScript type checking
4. 🎨 Linting (with lint-staged)
5. 🚫 Banned patterns (console.log, any types, debugger)
6. 🛡️ Security configurations
7. 🧪 Test file presence
8. 📦 Dependency sync (package.json + lockfile)

**Skip hook:**

```bash
SKIP_HOOKS=1 git commit -m "message"
```

### Pre-Push Hook

**Runs before pushing to remote**

Checks:

1. 🧪 Full test suite
2. 🔨 Build process
3. 🔒 Security scan
4. 🔀 Merge conflicts

**Skip hook:**

```bash
SKIP_HOOKS=1 git push
```

### Commit-Msg Hook

**Validates commit message format**

Enforces [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Valid types:**

- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Formatting
- `refactor` - Code refactoring
- `perf` - Performance
- `test` - Tests
- `build` - Build system
- `ci` - CI/CD
- `chore` - Maintenance

**Examples:**

```bash
git commit -m "feat(auth): add JWT authentication"
git commit -m "fix(api): resolve rate limiting bypass"
git commit -m "docs(readme): update installation instructions"
```

### Post-Checkout Hook

**Runs after checking out a branch**

- Checks if dependencies changed
- Prompts to run `pnpm install` if needed
- Notifies about .env.example changes
- Suggests cleaning build artifacts

---

## 🔧 Task Runner Configuration

### Enhanced Turbo Configuration

Location: `.automation/tasks/turbo-enhanced.json`

**Usage:**

```bash
# Copy to root
cp .automation/tasks/turbo-enhanced.json ./turbo.json

# Or merge with existing configuration
```

**Available tasks:**

- `build` - Build projects
- `test` - Run tests
- `test:unit` - Unit tests
- `test:integration` - Integration tests
- `test:e2e` - E2E tests
- `test:coverage` - Coverage report
- `lint` - Lint code
- `lint:fix` - Auto-fix linting
- `type-check` - TypeScript type checking
- `security:scan` - Security scanning
- `security:audit` - Dependency audit
- `format` - Format code
- `format:check` - Check formatting
- `clean` - Clean build artifacts
- `dev` - Development mode
- `deploy` - Deploy (runs all checks)

**Run tasks:**

```bash
# Single task
pnpm turbo run build

# Multiple tasks
pnpm turbo run lint type-check test

# Affected packages only
pnpm turbo run test --filter='...[HEAD^1]'

# Specific package
pnpm turbo run build --filter=my-package
```

---

## 🔌 External Tool Integrations

### PiecesOS

**Context preservation and code snippet management**

**Setup:**

```bash
bash .automation/integrations/pieces-os/setup.sh
```

**Usage:**

```bash
# Save context
pieces save --context "working on auth feature"

# Load context
pieces load --context "auth feature"

# Search snippets
pieces search "jwt implementation"
```

**Configuration:** `.pieces/config.json`

### Augment CLI

**AI-assisted development workflow**

**Setup:**

```bash
bash .automation/integrations/augment-cli/setup.sh
```

**Usage:**

```bash
# Review PR
augment review pr --number 123

# Generate docs
augment docs generate --path ./src

# Team analytics
augment analytics --team
```

**Configuration:** `.augment/config.yml`

### MCP Registry

**Model Context Protocol integrations**

**Location:** `.automation/integrations/mcp/registry.json`

**Categories:**

- **Developer MCPs** - Code analysis, testing, documentation, refactoring, security
- **Researcher MCPs** - Paper search, data analysis, citations, LaTeX
- **Scientist MCPs** - Simulations, plotting, data processing, equations

**Browse registry:**

```bash
cat .automation/integrations/mcp/registry.json | jq '.registry.mcps'
```

---

## 🎯 Common Workflows

### Starting Development

```bash
# 1. Fresh clone
git clone <repo-url>
cd <repo-name>

# 2. Run setup
bash .automation/scripts/dev-setup.sh

# 3. Start developing
pnpm dev
```

### Before Committing

```bash
# 1. Run auto-fix
bash .automation/scripts/fix-all.sh

# 2. Run tests
bash .automation/scripts/run-tests.sh --changed

# 3. Security scan (optional)
bash .automation/scripts/security-scan.sh

# 4. Commit (hooks will run automatically)
git add .
git commit -m "feat(component): add new feature"
```

### Before Pushing

```bash
# 1. Run health check
bash .automation/scripts/health-check.sh

# 2. Push (pre-push hook will run)
git push
```

### Daily Maintenance

```bash
# Update dependencies
pnpm update

# Audit security
pnpm audit

# Run full test suite
bash .automation/scripts/run-tests.sh --all

# Check repository health
bash .automation/scripts/health-check.sh
```

---

## 🔍 Troubleshooting

### Hook Failures

**Problem:** Pre-commit hook is blocking your commit

**Solutions:**

```bash
# Fix issues automatically
bash .automation/scripts/fix-all.sh

# Skip hooks temporarily (NOT recommended)
SKIP_HOOKS=1 git commit -m "message"

# Check what's failing
bash .automation/scripts/health-check.sh
```

### Test Failures

**Problem:** Tests failing in pre-push hook

**Solutions:**

```bash
# Run tests locally with details
pnpm test

# Run specific test
pnpm test path/to/test.spec.ts

# Clear cache
rm -rf node_modules/.cache
pnpm test
```

### Build Failures

**Problem:** Build failing

**Solutions:**

```bash
# Clean and rebuild
pnpm clean
pnpm build

# Check TypeScript errors
pnpm type-check

# Update dependencies
pnpm install
```

### Security Scan Issues

**Problem:** Security scan detecting issues

**Solutions:**

```bash
# View detailed report
bash .automation/scripts/security-scan.sh

# Fix dependency vulnerabilities
pnpm audit fix

# Check for secrets
docker run --rm -v $(pwd):/scan zricethezav/gitleaks:latest detect --source /scan
```

---

## 📚 Additional Resources

### Documentation

- [Master Plan](../WORKFLOW_AUTOMATION_MASTER_PLAN.md) - Complete automation strategy
- [Security Report](../FINAL_SECURITY_REMEDIATION_REPORT.md) - Security improvements
- [Contributing](../CONTRIBUTING.md) - Contribution guidelines

### External Tools

- [PiecesOS](https://pieces.app/) - Context management
- [Augment CLI](https://www.augmentcode.com/) - AI development assistant
- [Gitleaks](https://github.com/gitleaks/gitleaks) - Secret scanning
- [Turbo](https://turbo.build/) - Build system

### Best Practices

- Always run health check after major changes
- Keep dependencies up to date
- Review security scan reports regularly
- Use conventional commits
- Write tests for new features

---

## 🤝 Support

**Issues?** Check the troubleshooting section above or:

- Review logs in your terminal
- Run health check: `bash .automation/scripts/health-check.sh`
- Contact: meshal@berkeley.edu

---

## 📝 Maintenance

**Last Updated:** November 22, 2025
**Maintainer:** GitHub Monorepo Team
**Version:** 1.0.0

**Update Schedule:**

- Scripts: As needed
- Hooks: Quarterly review
- Integrations: When tools update
- Documentation: With each major change
