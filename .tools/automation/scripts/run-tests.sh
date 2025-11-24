#!/bin/bash
# Intelligent Test Runner
# Runs tests based on changed files and context

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TEST_EXIT_CODE=0
MAX_TARGETED_FILES=${MAX_TARGETED_FILES:-200}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

require_command() {
    local cmd="$1"
    local help="$2"
    if ! command_exists "$cmd"; then
        echo -e "${RED}Missing required tool: ${cmd}${NC}"
        [ -n "$help" ] && echo -e "${YELLOW}$help${NC}"
        exit 1
    fi
}

require_command pnpm "Install pnpm (or enable corepack) before running tests."

run_tests() {
    set +e
    "$@"
    local status=$?
    set -e
    if [ $status -ne 0 ]; then
        TEST_EXIT_CODE=$status
    fi
    return $status
}

resolve_base_branch() {
    local candidates=()
    if [ -n "${BASE_BRANCH:-}" ]; then
        candidates+=("$BASE_BRANCH")
    fi
    if [ -n "${GITHUB_BASE_REF:-}" ]; then
        candidates+=("origin/${GITHUB_BASE_REF}" "${GITHUB_BASE_REF}")
    fi
    candidates+=("origin/main" "main" "origin/master" "master")

    for candidate in "${candidates[@]}"; do
        if git rev-parse --verify "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return
        fi
    done

    if git rev-parse --verify HEAD^ >/dev/null 2>&1; then
        echo "HEAD^"
    else
        echo "HEAD"
    fi
}

get_base_commit() {
    if [ ! -d ".git" ]; then
        echo ""
        return
    fi

    local base_branch
    base_branch=$(resolve_base_branch)
    git merge-base "$base_branch" HEAD 2>/dev/null || echo ""
}

# Default mode
MODE="changed"  # changed, all, watch, coverage

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            MODE="all"
            shift
            ;;
        --watch)
            MODE="watch"
            shift
            ;;
        --coverage)
            MODE="coverage"
            shift
            ;;
        --changed)
            MODE="changed"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🧪 Intelligent Test Runner${NC}"
echo -e "${BLUE}Mode: ${MODE}${NC}"
echo -e "${BLUE}========================${NC}"
echo ""

# Function to get changed files
get_changed_files() {
    if [ ! -d ".git" ]; then
        echo ""
        return
    fi

    local base_branch
    base_branch=$(resolve_base_branch)
    if git diff --name-only "$base_branch"...HEAD 2>/dev/null; then
        return
    fi

    git diff --name-only HEAD
}

# Function to find affected packages
find_affected_packages() {
    local changed_files="$1"

    # Extract package directories from changed files
    echo "$changed_files" | while read file; do
        # Extract package path (assumes packages are in organizations/, metaHub/, etc.)
        if [[ "$file" =~ ^(organizations|metaHub|apps|packages)/([^/]+) ]]; then
            echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
        fi
    done | sort -u
}

case $MODE in
    "all")
        echo -e "${BLUE}Running all tests...${NC}"
        run_tests pnpm test
        ;;

    "watch")
        echo -e "${BLUE}Running tests in watch mode...${NC}"
        run_tests pnpm test -- --watch
        ;;

    "coverage")
        echo -e "${BLUE}Running tests with coverage...${NC}"
        run_tests pnpm test -- --coverage
        echo ""
        echo -e "${BLUE}Coverage report generated${NC}"
        echo "View detailed report: open coverage/index.html"
        ;;

    "changed")
        echo -e "${BLUE}Detecting changed files...${NC}"
        CHANGED_FILES=$(get_changed_files)

        if [ -z "$CHANGED_FILES" ]; then
            echo -e "${YELLOW}No changes detected. Running all tests.${NC}"
            run_tests pnpm test
        else
            CHANGED_COUNT=$(echo "$CHANGED_FILES" | sed '/^\s*$/d' | wc -l | tr -d '[:space:]')
            CHANGED_COUNT=${CHANGED_COUNT:-0}
            echo -e "${GREEN}Found changes in:${NC}"
            echo "$CHANGED_FILES" | head -10
            [ "$CHANGED_COUNT" -gt 10 ] && echo "... and more ($CHANGED_COUNT files total)"
            echo ""

            if [ "$CHANGED_COUNT" -gt "$MAX_TARGETED_FILES" ]; then
                echo -e "${YELLOW}Change set exceeds ${MAX_TARGETED_FILES} files. Running full test suite.${NC}"
                run_tests pnpm test
            else
                # Find affected packages
                AFFECTED_PACKAGES=$(find_affected_packages "$CHANGED_FILES")
                if [ -n "$AFFECTED_PACKAGES" ]; then
                    echo -e "${BLUE}Affected packages:${NC}"
                    echo "$AFFECTED_PACKAGES"
                    echo ""
                fi

                BASE_COMMIT=$(get_base_commit)
                if [ -z "$BASE_COMMIT" ]; then
                    echo -e "${YELLOW}Unable to determine base commit. Running full test suite.${NC}"
                    run_tests pnpm test
                elif pnpm exec turbo --version >/dev/null 2>&1; then
                    echo -e "${BLUE}Running targeted tests with Turbo (base: $BASE_COMMIT)...${NC}"
                    run_tests pnpm exec turbo run test --filter="...[${BASE_COMMIT}]"
                else
                    echo -e "${YELLOW}Turbo not available. Running pnpm test instead.${NC}"
                    run_tests pnpm test
                fi
            fi
        fi
        ;;
esac

# Check test results
if [ "$TEST_EXIT_CODE" -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Some tests failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting tips:${NC}"
    echo "  1. Review test output above for specific failures"
    echo "  2. Run failed tests individually for more details"
    echo "  3. Check if dependencies are up to date: pnpm install"
    echo "  4. Clear cache: rm -rf node_modules/.cache"
    exit "$TEST_EXIT_CODE"
fi
