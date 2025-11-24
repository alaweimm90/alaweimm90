#!/bin/bash
# Monorepo Health Check Script
# Comprehensive diagnostic tool for repository health

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🏥 GitHub Monorepo Health Check${NC}"
echo -e "${BLUE}===============================${NC}"
echo ""

ERRORS=0
WARNINGS=0
CHECKS_PASSED=0

# Function to increment counters
pass() {
    echo -e "${GREEN}✅ $1${NC}"
    ((CHECKS_PASSED++))
}

fail() {
    echo -e "${RED}❌ $1${NC}"
    ((ERRORS++))
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 1. Repository Structure
echo -e "${BLUE}📁 Check 1/13: Repository Structure${NC}"
if [ -f "package.json" ]; then
    pass "Root package.json exists"
else
    fail "Root package.json missing"
fi

if [ -f "turbo.json" ]; then
    pass "Turbo configuration exists"
else
    warn "Turbo configuration missing"
fi

if [ -f "pnpm-workspace.yaml" ] || [ -f "pnpm-workspace.yml" ]; then
    pass "PNPM workspace configured"
else
    warn "PNPM workspace not configured"
fi

# 2. Dependencies
echo ""
echo -e "${BLUE}📦 Check 2/13: Dependencies${NC}"
if [ -d "node_modules" ]; then
    pass "node_modules directory exists"

    # Check if dependencies are up to date
    if pnpm outdated --depth 0 2>&1 | grep -q "WARN"; then
        warn "Some dependencies are outdated"
        info "Run 'pnpm outdated' for details"
    else
        pass "Dependencies are up to date"
    fi
else
    fail "node_modules missing - run 'pnpm install'"
fi

# Check for lockfile
if [ -f "pnpm-lock.yaml" ]; then
    pass "Lockfile exists"
else
    fail "pnpm-lock.yaml missing"
fi

# 3. Git Configuration
echo ""
echo -e "${BLUE}🔧 Check 3/13: Git Configuration${NC}"
if [ -d ".git" ]; then
    pass "Git repository initialized"

    # Check hooks
    if [ -d ".husky" ]; then
        pass "Git hooks directory exists"

        if [ -f ".husky/pre-commit" ]; then
            pass "Pre-commit hook configured"
        else
            warn "Pre-commit hook missing"
        fi
    else
        warn "Husky not configured"
    fi

    # Check for uncommitted changes
    if [ -n "$(git status --porcelain)" ]; then
        warn "Uncommitted changes present"
    else
        pass "Working directory clean"
    fi
else
    fail "Not a git repository"
fi

# 4. Environment Configuration
echo ""
echo -e "${BLUE}🔐 Check 4/13: Environment Configuration${NC}"
if [ -f ".env" ]; then
    pass ".env file exists"

    # Check for common required variables
    REQUIRED_VARS=("NODE_ENV")
    for var in "${REQUIRED_VARS[@]}"; do
        if grep -q "^${var}=" .env; then
            pass "$var is set"
        else
            warn "$var not found in .env"
        fi
    done
else
    warn ".env file missing (may be optional)"
fi

if [ -f ".env.example" ]; then
    pass ".env.example exists for reference"
fi

# Check .gitignore for sensitive files
if [ -f ".gitignore" ]; then
    if grep -q ".env" .gitignore; then
        pass ".env is gitignored"
    else
        fail ".env not in .gitignore - SECURITY RISK!"
    fi
fi

# 5. TypeScript Configuration
echo ""
echo -e "${BLUE}📝 Check 5/13: TypeScript Configuration${NC}"
if [ -f "tsconfig.json" ]; then
    pass "Root tsconfig.json exists"

    # Check for strict mode
    if grep -q '"strict": true' tsconfig.json; then
        pass "Strict mode enabled"
    else
        warn "Strict mode not enabled"
    fi
else
    warn "No tsconfig.json found"
fi

# 6. Build System
echo ""
echo -e "${BLUE}🔨 Check 6/13: Build System${NC}"
if pnpm run build --dry-run 2>&1 | grep -q "build"; then
    pass "Build script configured"
else
    warn "No build script found"
fi

# Check if builds are cached
if [ -d "node_modules/.cache/turbo" ]; then
    pass "Turbo cache directory exists"
fi

# 7. Testing
echo ""
echo -e "${BLUE}🧪 Check 7/13: Testing Infrastructure${NC}"
if pnpm run test --dry-run 2>&1 | grep -q "test"; then
    pass "Test script configured"
else
    warn "No test script found"
fi

# Check for test configuration
if [ -f "jest.config.js" ] || [ -f "vitest.config.ts" ]; then
    pass "Test configuration exists"
else
    warn "No test configuration found"
fi

# Check for test files
TEST_FILES=$(find . -name "*.test.ts" -o -name "*.test.tsx" -o -name "*.spec.ts" 2>/dev/null | wc -l)
if [ "$TEST_FILES" -gt 0 ]; then
    pass "Found $TEST_FILES test files"
else
    warn "No test files found"
fi

# 8. Linting
echo ""
echo -e "${BLUE}🎨 Check 8/13: Code Quality${NC}"
if [ -f ".eslintrc.json" ] || [ -f ".eslintrc.js" ] || [ -f "eslint.config.js" ]; then
    pass "ESLint configuration exists"
else
    warn "No ESLint configuration found"
fi

if [ -f ".prettierrc" ] || [ -f ".prettierrc.json" ]; then
    pass "Prettier configuration exists"
else
    warn "No Prettier configuration found"
fi

# 9. Security
echo ""
echo -e "${BLUE}🔒 Check 9/13: Security${NC}"

# Check for known vulnerabilities
info "Checking for known vulnerabilities..."
if pnpm audit --audit-level=high 2>&1 | grep -q "found 0 vulnerabilities"; then
    pass "No high/critical vulnerabilities found"
else
    warn "Vulnerabilities detected - run 'pnpm audit' for details"
fi

# Check for hardcoded secrets patterns
SECRETS_PATTERN="(password|secret|key|token|api_key).*=.*['\"][^'\"]{8,}"
if grep -r -i -E "$SECRETS_PATTERN" --exclude-dir=node_modules --exclude-dir=.git --exclude="*.md" . 2>/dev/null | grep -v ".env.example" | grep -q .; then
    fail "Potential hardcoded secrets found!"
    info "Run security scan for details"
else
    pass "No obvious hardcoded secrets detected"
fi

# Check for sensitive files in git
if [ -d ".git" ]; then
    SENSITIVE_FILES=(".env" "*.key" "*.pem" "credentials.json")
    for pattern in "${SENSITIVE_FILES[@]}"; do
        if git ls-files | grep -q "$pattern"; then
            fail "Sensitive file pattern '$pattern' found in git history"
        fi
    done
fi

# 10. Documentation
echo ""
echo -e "${BLUE}📚 Check 10/13: Documentation${NC}"
REQUIRED_DOCS=("README.md" "CONTRIBUTING.md" "LICENSE")
for doc in "${REQUIRED_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        pass "$doc exists"
    else
        warn "$doc missing"
    fi
done

# 11. CI/CD
echo ""
echo -e "${BLUE}🚀 Check 11/13: CI/CD Configuration${NC}"
if [ -d ".github/workflows" ]; then
    WORKFLOW_COUNT=$(find .github/workflows -name "*.yml" -o -name "*.yaml" 2>/dev/null | wc -l)
    pass "Found $WORKFLOW_COUNT GitHub Actions workflows"

    # Check for essential workflows
    if [ -f ".github/workflows/ci.yml" ] || [ -f ".github/workflows/test.yml" ]; then
        pass "CI workflow configured"
    else
        warn "No CI workflow found"
    fi
else
    warn "No GitHub Actions workflows found"
fi

# 12. Automation
echo ""
echo -e "${BLUE}🤖 Check 12/13: Automation Infrastructure${NC}"
if [ -d ".automation" ]; then
    pass "Automation directory exists"

    if [ -d ".automation/scripts" ]; then
        SCRIPT_COUNT=$(find .automation/scripts -name "*.sh" 2>/dev/null | wc -l)
        pass "Found $SCRIPT_COUNT automation scripts"
    fi

    if [ -f ".automation/README.md" ]; then
        pass "Automation documentation exists"
    else
        warn "Automation documentation missing"
    fi
else
    warn "Automation infrastructure not set up"
fi

# 13. Governance Manifest
echo ""
echo -e "${BLUE}📑 Check 13/13: Governance Mapping${NC}"
if [ -f "scripts/doc_audit.js" ]; then
    if DOC_AUDIT_OUTPUT=$(node scripts/doc_audit.js --quiet 2>&1); then
        pass "Governance manifest satisfied"
    else
        fail "Governance manifest missing required docs"
        echo "$DOC_AUDIT_OUTPUT"
    fi
else
    warn "Governance audit script missing"
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📊 Health Check Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Checks Passed: $CHECKS_PASSED${NC}"
echo -e "${YELLOW}⚠️  Warnings: $WARNINGS${NC}"
echo -e "${RED}❌ Errors: $ERRORS${NC}"
echo ""

# Overall health score
TOTAL_CHECKS=$((CHECKS_PASSED + WARNINGS + ERRORS))
if [ "$TOTAL_CHECKS" -gt 0 ]; then
    HEALTH_SCORE=$((CHECKS_PASSED * 100 / TOTAL_CHECKS))
    echo -e "${BLUE}Overall Health Score: $HEALTH_SCORE%${NC}"
    echo ""

    if [ "$HEALTH_SCORE" -ge 90 ]; then
        echo -e "${GREEN}🎉 Excellent! Your repository is in great shape!${NC}"
    elif [ "$HEALTH_SCORE" -ge 70 ]; then
        echo -e "${YELLOW}👍 Good! Consider addressing warnings for optimal health.${NC}"
    elif [ "$HEALTH_SCORE" -ge 50 ]; then
        echo -e "${YELLOW}⚠️  Fair. Please address errors and warnings.${NC}"
    else
        echo -e "${RED}🚨 Poor. Immediate attention required!${NC}"
    fi
fi

echo ""

# Exit with error if critical issues found
if [ "$ERRORS" -gt 0 ]; then
    echo -e "${RED}❌ Critical issues detected. Please resolve errors above.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ No critical issues detected.${NC}"
    exit 0
fi

# Blockchain health check
echo "🔗 Checking blockchain integration..."
if [ -d "automation/blockchain" ]; then
    cd automation/blockchain && node scripts/health-check.js && cd ../..
fi
