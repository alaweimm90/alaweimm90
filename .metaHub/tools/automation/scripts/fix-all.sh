#!/bin/bash
# Auto-fix Script
# Automatically fixes linting, formatting, and common issues

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 Auto-Fix: Linting, Formatting & Common Issues${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

FIXES_APPLIED=0

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

apply_fix() {
    echo -e "${GREEN}✅ $1${NC}"
    ((FIXES_APPLIED++))
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

require_command() {
    local cmd="$1"
    local help="$2"
    if ! command_exists "$cmd"; then
        echo -e "${RED}❌ Missing required tool: ${cmd}${NC}"
        [ -n "$help" ] && info "$help"
        exit 1
    fi
}

require_command pnpm "Install pnpm (or enable via corepack) before running fix-all."

TMP_DIR=${TMPDIR:-/tmp}
PRETTIER_LOG="$TMP_DIR/fix-all-prettier.log"
IMPORTS_LOG="$TMP_DIR/fix-all-organize-imports.log"
UNUSED_LOG="$TMP_DIR/fix-all-unused-imports.log"

# 1. PRETTIER FORMATTING
echo -e "${BLUE}📝 Step 1/7: Running Prettier...${NC}"
if pnpm exec prettier --loglevel warn --write "**/*.{ts,tsx,js,jsx,json,md,yml,yaml}" >"$PRETTIER_LOG" 2>&1; then
    if [ -s "$PRETTIER_LOG" ]; then
        apply_fix "Code formatted with Prettier"
        cat "$PRETTIER_LOG"
    else
        info "Prettier ran with no reported changes"
    fi
else
    warn "Prettier reported errors (see $PRETTIER_LOG)"
    cat "$PRETTIER_LOG"
fi

# 2. ESLINT AUTO-FIX
echo ""
echo -e "${BLUE}🎨 Step 2/7: Running ESLint auto-fix...${NC}"
if pnpm exec eslint . --fix --ext .ts,.tsx,.js,.jsx 2>/dev/null; then
    apply_fix "ESLint auto-fixes applied"
else
    echo "ESLint completed (some issues may require manual fix)"
fi

# 3. ORGANIZE IMPORTS
echo ""
echo -e "${BLUE}📦 Step 3/7: Organizing imports...${NC}"
if pnpm exec organize-imports-cli --version >/dev/null 2>&1; then
    pnpm exec organize-imports-cli "**/*.ts" "**/*.tsx" >"$IMPORTS_LOG" 2>&1 || true
    apply_fix "Imports organized"
else
    warn "organize-imports-cli not installed (skip or install via 'pnpm dlx organize-imports-cli')"
fi

# 4. REMOVE UNUSED IMPORTS
echo ""
echo -e "${BLUE}🧹 Step 4/7: Detecting unused imports...${NC}"
if pnpm exec ts-unused-exports --help >/dev/null 2>&1; then
    if [ -f tsconfig.json ]; then
        pnpm exec ts-unused-exports tsconfig.json >"$UNUSED_LOG" 2>&1 || true
        cat "$UNUSED_LOG"
        apply_fix "ts-unused-exports completed (review log for manual removals)"
    else
        warn "tsconfig.json not found - skipping unused import detection"
    fi
else
    warn "ts-unused-exports not installed - skipping destructive heuristics"
fi

# 5. FIX PACKAGE.JSON
echo ""
echo -e "${BLUE}📦 Step 5/7: Fixing package.json files...${NC}"
find . -name "package.json" | grep -v node_modules | while read pkg; do
    # Sort package.json
    if command -v jq >/dev/null 2>&1; then
        jq -S . "$pkg" > "${pkg}.tmp" && mv "${pkg}.tmp" "$pkg"
        apply_fix "Sorted $pkg"
    fi
done

# 6. FIX LINE ENDINGS
echo ""
echo -e "${BLUE}↩️  Step 6/7: Normalizing line endings...${NC}"
if command_exists python; then
    python - <<'PY'
import pathlib

patterns = ("*.ts", "*.tsx", "*.js", "*.jsx")
for pattern in patterns:
    for path in pathlib.Path(".").rglob(pattern):
        if "node_modules" in path.parts:
            continue
        data = path.read_bytes()
        normalized = data.replace(b"\r\n", b"\n")
        if normalized != data:
            path.write_bytes(normalized)
PY
    apply_fix "Line endings normalized to LF"
else
    warn "Python not available - skipping line ending normalization"
fi

# 7. FIX PERMISSIONS
echo ""
echo -e "${BLUE}🔐 Step 7/7: Fixing file permissions...${NC}"
if [ "$(uname)" != "Windows_NT" ]; then
    # Make scripts executable
    find . -name "*.sh" | grep -v node_modules | while read script; do
        chmod +x "$script"
    done
    apply_fix "Script permissions fixed"

    # Remove world-writable permissions
    find . -type f -perm -002 ! -path "./node_modules/*" -exec chmod o-w {} \;
    apply_fix "Removed world-writable permissions"
else
    echo "Skipping permission fixes on Windows"
fi

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Auto-fix Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Applied $FIXES_APPLIED automatic fixes${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Review changes with 'git diff'"
echo "  2. Run tests with 'pnpm test'"
echo "  3. Run health check with '.automation/scripts/health-check.sh'"
echo "  4. Commit changes if everything looks good"
echo ""
