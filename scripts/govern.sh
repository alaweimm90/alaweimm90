#!/usr/bin/env bash
#
# govern.sh - Local Governance Pre-commit Hook
#
# This script runs governance checks locally before commits.
# Install via pre-commit or as a direct git hook.
#
# Usage:
#   ./govern.sh [options]
#
# Options:
#   --check     Run checks only (no fixes)
#   --fix       Auto-fix issues where possible
#   --verbose   Show detailed output
#   --schema    Validate schema only
#   --docker    Check Dockerfiles only
#   --help      Show this help message
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Find central repo path
find_central_path() {
    local current="$PWD"
    while [[ "$current" != "/" ]]; do
        if [[ -d "$current/.metaHub" ]]; then
            echo "$current"
            return 0
        fi
        current="$(dirname "$current")"
    done

    # Fallback to GOLDEN_PATH_ROOT
    if [[ -n "$GOLDEN_PATH_ROOT" && -d "$GOLDEN_PATH_ROOT/.metaHub" ]]; then
        echo "$GOLDEN_PATH_ROOT"
        return 0
    fi

    return 1
}

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

# Show help
show_help() {
    cat << 'EOF'
govern.sh - Local Governance Pre-commit Hook

Usage: govern.sh [options]

Options:
  --check     Run checks only (default, no modifications)
  --fix       Auto-fix issues where possible
  --verbose   Show detailed output
  --schema    Validate .meta/repo.yaml schema only
  --docker    Check Dockerfiles only
  --all       Run all checks (default)
  --help      Show this help message

Examples:
  # Run all checks
  ./govern.sh

  # Check schema only
  ./govern.sh --schema

  # Verbose output
  ./govern.sh --verbose

  # Auto-fix where possible
  ./govern.sh --fix

Environment:
  GOLDEN_PATH_ROOT    Path to central governance repo
  GOVERN_STRICT       Set to 'true' for strict mode (fail on warnings)
EOF
}

# Check if Python is available
check_python() {
    if command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        print_msg "$RED" "[ERROR] Python not found. Please install Python 3.8+."
        exit 1
    fi
}

# Validate .meta/repo.yaml schema
check_schema() {
    local verbose=$1
    local repo_path="${2:-.}"

    print_msg "$BLUE" "[CHECK] Validating .meta/repo.yaml schema..."

    local meta_file="$repo_path/.meta/repo.yaml"
    if [[ ! -f "$meta_file" ]]; then
        print_msg "$YELLOW" "[WARN] No .meta/repo.yaml found at $repo_path"
        return 1
    fi

    local central_path
    central_path=$(find_central_path) || {
        print_msg "$YELLOW" "[WARN] Could not find central governance repo"
        return 1
    }

    local schema_file="$central_path/.metaHub/schemas/repo-schema.json"
    if [[ ! -f "$schema_file" ]]; then
        print_msg "$YELLOW" "[WARN] Schema file not found: $schema_file"
        return 1
    fi

    local python_cmd
    python_cmd=$(check_python)

    # Validate using Python
    $python_cmd -c "
import json
import yaml
import jsonschema
import sys

try:
    with open('$schema_file') as f:
        schema = json.load(f)
    with open('$meta_file') as f:
        metadata = yaml.safe_load(f)

    jsonschema.validate(metadata, schema)
    print('[OK] Schema validation passed')
    sys.exit(0)
except jsonschema.ValidationError as e:
    print(f'[ERROR] Schema validation failed: {e.message}')
    sys.exit(1)
except Exception as e:
    print(f'[ERROR] {e}')
    sys.exit(1)
" 2>&1

    return $?
}

# Check Dockerfiles
check_docker() {
    local verbose=$1
    local repo_path="${2:-.}"

    print_msg "$BLUE" "[CHECK] Validating Dockerfiles..."

    local dockerfiles
    dockerfiles=$(find "$repo_path" -name "Dockerfile*" -type f 2>/dev/null)

    if [[ -z "$dockerfiles" ]]; then
        print_msg "$GREEN" "[OK] No Dockerfiles found"
        return 0
    fi

    local errors=0
    while IFS= read -r dockerfile; do
        [[ -z "$dockerfile" ]] && continue

        print_msg "$BLUE" "  Checking: $dockerfile"

        # Check for USER directive (non-root)
        if ! grep -qE '^USER\s+(?!root)' "$dockerfile"; then
            print_msg "$YELLOW" "    [WARN] Missing non-root USER directive"
            ((errors++)) || true
        fi

        # Check for HEALTHCHECK
        if ! grep -q '^HEALTHCHECK' "$dockerfile"; then
            print_msg "$YELLOW" "    [WARN] Missing HEALTHCHECK directive"
        fi

        # Check for :latest tag
        if grep -qE '^FROM\s+\S+:latest' "$dockerfile"; then
            print_msg "$RED" "    [ERROR] Using :latest tag in FROM"
            ((errors++)) || true
        fi

        # Check for secrets in ENV
        if grep -qiE '^ENV\s+\S*(PASSWORD|SECRET|TOKEN|API_KEY)' "$dockerfile"; then
            print_msg "$RED" "    [ERROR] Possible secrets in ENV directive"
            ((errors++)) || true
        fi

    done <<< "$dockerfiles"

    if [[ $errors -gt 0 ]]; then
        print_msg "$RED" "[FAIL] Docker checks: $errors issues found"
        return 1
    fi

    print_msg "$GREEN" "[OK] Docker checks passed"
    return 0
}

# Check required files
check_required_files() {
    local verbose=$1
    local repo_path="${2:-.}"

    print_msg "$BLUE" "[CHECK] Validating required files..."

    local required_files=(
        "README.md"
        ".meta/repo.yaml"
    )

    local missing=0
    for file in "${required_files[@]}"; do
        if [[ ! -f "$repo_path/$file" ]]; then
            print_msg "$YELLOW" "  [MISSING] $file"
            ((missing++)) || true
        fi
    done

    if [[ $missing -gt 0 ]]; then
        print_msg "$YELLOW" "[WARN] $missing required files missing"
        return 1
    fi

    print_msg "$GREEN" "[OK] All required files present"
    return 0
}

# Run full enforcement
run_enforcement() {
    local verbose=$1
    local repo_path="${2:-.}"

    print_msg "$BLUE" "[CHECK] Running full enforcement..."

    local central_path
    central_path=$(find_central_path) || {
        print_msg "$YELLOW" "[WARN] Could not find central governance repo"
        return 1
    }

    local python_cmd
    python_cmd=$(check_python)

    local enforce_script="$central_path/.metaHub/scripts/enforce.py"
    if [[ ! -f "$enforce_script" ]]; then
        print_msg "$YELLOW" "[WARN] enforce.py not found at $enforce_script"
        return 1
    fi

    local args=("$repo_path")
    [[ "$verbose" == "true" ]] && args+=("--report" "text")

    $python_cmd "$enforce_script" "${args[@]}"
    return $?
}

# Main function
main() {
    local mode="check"
    local verbose="false"
    local check_schema_only="false"
    local check_docker_only="false"
    local repo_path="."

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check)
                mode="check"
                shift
                ;;
            --fix)
                mode="fix"
                shift
                ;;
            --verbose|-v)
                verbose="true"
                shift
                ;;
            --schema)
                check_schema_only="true"
                shift
                ;;
            --docker)
                check_docker_only="true"
                shift
                ;;
            --all)
                check_schema_only="false"
                check_docker_only="false"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                if [[ -d "$1" ]]; then
                    repo_path="$1"
                fi
                shift
                ;;
        esac
    done

    print_msg "$BLUE" "========================================"
    print_msg "$BLUE" "   Golden Path Governance Check"
    print_msg "$BLUE" "========================================"
    echo ""

    local exit_code=0

    # Run selected checks
    if [[ "$check_schema_only" == "true" ]]; then
        check_schema "$verbose" "$repo_path" || exit_code=1
    elif [[ "$check_docker_only" == "true" ]]; then
        check_docker "$verbose" "$repo_path" || exit_code=1
    else
        # Run all checks
        check_required_files "$verbose" "$repo_path" || exit_code=1
        check_schema "$verbose" "$repo_path" || exit_code=1
        check_docker "$verbose" "$repo_path" || exit_code=1

        # Run full enforcement if available
        run_enforcement "$verbose" "$repo_path" || exit_code=1
    fi

    echo ""
    if [[ $exit_code -eq 0 ]]; then
        print_msg "$GREEN" "========================================"
        print_msg "$GREEN" "   All governance checks passed!"
        print_msg "$GREEN" "========================================"
    else
        print_msg "$RED" "========================================"
        print_msg "$RED" "   Governance checks failed!"
        print_msg "$RED" "   Please fix issues before committing."
        print_msg "$RED" "========================================"
    fi

    exit $exit_code
}

# Run main
main "$@"
