#!/bin/bash
# Local Security Scanning Script
# Comprehensive security checks before committing

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

trap 'echo -e "${RED}[x] Security scan aborted early. Review the errors above for context.${NC}" >&2' ERR

echo -e "${BLUE}=> GitHub Monorepo Security Scan${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

ISSUES_FOUND=0

report_issue() {
    echo -e "${RED}[err] $1${NC}"
    ((ISSUES_FOUND++)) || true
}

report_warning() {
    echo -e "${YELLOW}[warn] $1${NC}"
}

report_success() {
    echo -e "${GREEN}[ok] $1${NC}"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

TMP_DIR=${TMPDIR:-/tmp}
AUDIT_LOG="$TMP_DIR/security-scan-audit.log"
GITLEAKS_LOG="$TMP_DIR/security-scan-gitleaks.log"

# 1. SECRET SCANNING
echo -e "${BLUE}=> Scan 1/8: Secret Detection${NC}"

if command_exists gitleaks; then
    echo "Running Gitleaks..."
    if gitleaks detect --source . --redact --no-banner >"$GITLEAKS_LOG" 2>&1; then
        report_success "No secrets detected by Gitleaks"
    else
        report_issue "Secrets detected by Gitleaks (see $GITLEAKS_LOG)"
        tail -n 20 "$GITLEAKS_LOG" 2>/dev/null || true
        echo "Run 'gitleaks detect --source .' for details"
    fi
else
    report_warning "Gitleaks not installed - install from https://github.com/gitleaks/gitleaks"
    if [ "${SECURITY_SCAN_FALLBACK:-0}" = "1" ]; then
        echo "Running pattern-based secret detection..."

        SECRET_PATTERNS=(
            "password\s*=\s*['\"][^'\"]{8,}"
            "api[_-]?key\s*=\s*['\"][^'\"]{8,}"
            "secret\s*=\s*['\"][^'\"]{8,}"
            "token\s*=\s*['\"][^'\"]{8,}"
            "AKIA[0-9A-Z]{16}"
            "sk_live_[0-9a-zA-Z]{24}"
            "ghp_[0-9a-zA-Z]{36}"
        )

        for pattern in "${SECRET_PATTERNS[@]}"; do
            if grep -r -i -E "$pattern" --exclude-dir=node_modules --exclude-dir=.git --exclude="*.md" --exclude-dir=.automation . 2>/dev/null | grep -v ".env.example" >/dev/null; then
                report_issue "Potential secret pattern found: $pattern"
            fi
        done
    else
        report_warning "Skipping fallback pattern scan (set SECURITY_SCAN_FALLBACK=1 to enable)"
    fi
fi

# 2. HARDCODED CREDENTIALS
echo ""
echo -e "${BLUE}=> Scan 2/8: Hardcoded Credentials${NC}"

CREDENTIAL_FILES=$(grep -r -i -E "(password|secret|key|token).*=.*['\"][^'\"]{8,}" \
    --include="*.ts" --include="*.tsx" --include="*.js" --include="*.jsx" --include="*.py" \
    --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=dist --exclude-dir=build \
    . 2>/dev/null | wc -l)

if [ "$CREDENTIAL_FILES" -eq 0 ]; then
    report_success "No obvious hardcoded credentials found"
else
    report_issue "Found $CREDENTIAL_FILES potential hardcoded credentials"
fi

# 3. DEPENDENCY VULNERABILITIES
echo ""
echo -e "${BLUE}=> Scan 3/8: Dependency Vulnerabilities${NC}"

if command_exists pnpm; then
    echo "Scanning with pnpm audit..."
    if pnpm audit --audit-level=high >"$AUDIT_LOG" 2>&1; then
        report_success "No high/critical vulnerabilities in dependencies"
    else
        report_issue "Vulnerabilities found in dependencies"
        tail -n 20 "$AUDIT_LOG" 2>/dev/null || true
        echo "Run 'pnpm audit' for full report"
    fi
else
    report_warning "pnpm is not installed - skipping dependency audit"
fi

# 4. UNSAFE CODE PATTERNS
echo ""
echo -e "${BLUE}=> Scan 4/8: Unsafe Code Patterns${NC}"

EVAL_COUNT=$(grep -r "eval(" --include="*.ts" --include="*.js" --exclude-dir=node_modules . 2>/dev/null | wc -l)
if [ "$EVAL_COUNT" -gt 0 ]; then
    report_issue "Found $EVAL_COUNT instances of eval() - potential code injection risk"
else
    report_success "No eval() calls found"
fi

INNERHTML_COUNT=$(grep -r "innerHTML" --include="*.ts" --include="*.tsx" --include="*.js" --include="*.jsx" --exclude-dir=node_modules . 2>/dev/null | wc -l)
if [ "$INNERHTML_COUNT" -gt 0 ]; then
    report_warning "Found $INNERHTML_COUNT instances of innerHTML - verify XSS protection"
fi

SQL_CONCAT=$(grep -r -E "SELECT.*\+|INSERT.*\+|UPDATE.*\+|DELETE.*\+" --include="*.ts" --include="*.js" --include="*.py" --exclude-dir=node_modules . 2>/dev/null | wc -l)
if [ "$SQL_CONCAT" -gt 0 ]; then
    report_issue "Found $SQL_CONCAT potential SQL injection vulnerabilities"
else
    report_success "No obvious SQL injection patterns found"
fi

# 5. SECURITY HEADERS
echo ""
echo -e "${BLUE}=> Scan 5/8: Security Headers Configuration${NC}"

HEADER_FILES=$( (find . -path "./node_modules" -prune -o \( -name "*middleware*" -o -name "*server*" \) -print 2>/dev/null | head -5) || true )

if [ -n "$HEADER_FILES" ]; then
    HEADERS_FOUND=false
    while IFS= read -r file; do
        if grep -q -i "x-frame-options\|content-security-policy\|strict-transport-security" "$file" 2>/dev/null; then
            HEADERS_FOUND=true
            break
        fi
    done <<<"$HEADER_FILES"

    if [ "$HEADERS_FOUND" = true ]; then
        report_success "Security headers configuration found"
    else
        report_warning "No security headers configuration detected"
    fi
fi

# 6. ENVIRONMENT FILE SECURITY
echo ""
echo -e "${BLUE}=> Scan 6/8: Environment File Security${NC}"

if [ -f ".gitignore" ]; then
    if grep -q "^\.env$" .gitignore; then
        report_success ".env is properly gitignored"
    else
        report_issue ".env is not gitignored - CRITICAL SECURITY RISK!"
    fi
fi

if [ -d ".git" ] && command_exists git; then
    if git log --all --full-history --name-only -- .env 2>/dev/null | grep -q ".env"; then
        report_issue ".env file found in git history!"
        echo "  Consider using git-filter-repo to remove it"
    else
        report_success ".env not found in git history"
    fi
fi

# 7. DOCKER SECURITY
echo ""
echo -e "${BLUE}=> Scan 7/8: Docker Security${NC}"

DOCKERFILES=$( (find . -path "./node_modules" -prune -o -name "Dockerfile*" -print 2>/dev/null) || true )

if [ -n "$DOCKERFILES" ]; then
    while IFS= read -r dockerfile; do
        if [ -z "$dockerfile" ]; then
            continue
        fi

        if ! grep -q "USER" "$dockerfile"; then
            report_warning "$dockerfile does not specify USER - may run as root"
        fi

        if grep -q "FROM.*:latest" "$dockerfile"; then
            report_warning "$dockerfile uses :latest tag - pin specific versions"
        fi

        if grep -q -i -E "ARG.*(PASSWORD|SECRET|KEY|TOKEN)" "$dockerfile"; then
            report_issue "$dockerfile may contain secrets in build args"
        fi
    done <<<"$DOCKERFILES"

    report_success "Docker security scan complete"
else
    echo "No Dockerfiles found"
fi

# 8. FILE PERMISSIONS
echo ""
echo -e "${BLUE}=> Scan 8/8: File Permissions${NC}"

if [ "$(uname)" != "Windows_NT" ]; then
    PERMISSIVE_FILES=$(find . -type f -perm -002 ! -path "./node_modules/*" ! -path "./.git/*" 2>/dev/null | wc -l)
    if [ "$PERMISSIVE_FILES" -gt 0 ]; then
        report_warning "Found $PERMISSIVE_FILES world-writable files"
    else
        report_success "No overly permissive files found"
    fi
else
    report_warning "File permission scan skipped on Windows"
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}=> Security Scan Summary${NC}"
echo -e "${BLUE}========================================${NC}"

if [ "$ISSUES_FOUND" -eq 0 ]; then
    echo -e "${GREEN}[ok] No security issues detected${NC}"
    echo -e "${GREEN}Repository passes all automated security checks.${NC}"
    exit 0
else
    echo -e "${RED}[err] Found $ISSUES_FOUND security issues${NC}"
    echo -e "${YELLOW}Please address the issues above before committing.${NC}"
    echo ""
    echo -e "${BLUE}Recommended actions:${NC}"
    echo "  1. Review and fix all reported issues"
    echo "  2. Run security scan again to verify fixes"
    echo "  3. Cross-check against SECURITY.md and remediation report"
    echo "  4. Run pnpm audit / gitleaks directly for verbose context"
    exit 1
fi
