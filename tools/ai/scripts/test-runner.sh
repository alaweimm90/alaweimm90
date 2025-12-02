#!/bin/bash
# Enhanced Test Runner v2.3
# AI Tools - Multi-framework test execution with TDD support

LOGS_DIR="$HOME/.ai_tools/logs"
mkdir -p "$LOGS_DIR"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

detect_framework() {
    [ -f "package.json" ] && grep -q "jest\|mocha\|vitest" package.json && echo "node"
    [ -f "pytest.ini" ] || [ -f "pyproject.toml" ] && echo "pytest"
    [ -f "Cargo.toml" ] && echo "cargo"
    [ -f "go.mod" ] && echo "go"
}

run_tests() {
    local framework=$(detect_framework)
    local start=$(date +%s)
    echo -e "${CYAN}🧪 Running tests...${NC}"
    
    case "$framework" in
        node) npm test 2>&1 ;;
        pytest) pytest -v 2>&1 ;;
        cargo) cargo test 2>&1 ;;
        go) go test ./... 2>&1 ;;
        *) echo "No test framework detected"; return 1 ;;
    esac
    
    local exit_code=$?
    local duration=$(($(date +%s) - start))
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Tests passed${NC} (${duration}s)"
    else
        echo -e "${RED}❌ Tests failed${NC} (${duration}s)"
    fi
    
    echo "[$(date)] $framework: exit=$exit_code duration=${duration}s" >> "$LOGS_DIR/tests.log"
    return $exit_code
}

watch_tests() {
    echo -e "${CYAN}👀 Watching for changes...${NC}"
    local framework=$(detect_framework)
    case "$framework" in
        node) npm run test:watch 2>/dev/null || npx jest --watch ;;
        pytest) ptw . ;;
        cargo) cargo watch -x test ;;
        *) echo "Watch not supported" ;;
    esac
}

tdd_cycle() {
    local prompt="$1"
    echo -e "${YELLOW}🔄 TDD Cycle${NC}"
    echo "1. Write failing test"
    echo "2. Run: ai-route 'Write test: $prompt'"
    echo "3. Verify: ai-test run"
    echo "4. Implement: ai-route 'Implement: $prompt'"
    echo "5. Refactor: ai-route 'Refactor: $prompt'"
}

case "$1" in
    run) run_tests ;;
    watch) watch_tests ;;
    tdd) tdd_cycle "$2" ;;
    *) echo "Usage: ai-test <run|watch|tdd>" ;;
esac
