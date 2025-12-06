#!/bin/bash

# HELIOS Code Formatter
# Version: 0.1.0
# Purpose: Format Python code with black and isort

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================"
echo "HELIOS Code Formatter"
echo "================================"
echo ""

# Check if tools are installed
if ! command -v black &> /dev/null; then
    echo -e "${RED}✗ black not found. Run: pip install -e .[dev]${NC}"
    exit 1
fi

if ! command -v isort &> /dev/null; then
    echo -e "${RED}✗ isort not found. Run: pip install -e .[dev]${NC}"
    exit 1
fi

# Parse arguments
PATHS="${@:-helios/}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            DRY_RUN=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

PATHS="${PATHS:-helios/}"

echo -e "${BLUE}Formatting${NC}: $PATHS"
echo ""

# Format with isort (import sorting)
echo -e "${BLUE}[1/2]${NC} Sorting imports with isort..."
if [ "$DRY_RUN" = true ]; then
    isort --check-only --diff "$PATHS"
    echo -e "${YELLOW}ℹ Dry run: no changes made${NC}"
else
    isort "$PATHS"
    echo -e "${GREEN}✓ Imports sorted${NC}"
fi
echo ""

# Format with black (code formatting)
echo -e "${BLUE}[2/2]${NC} Formatting code with black..."
if [ "$DRY_RUN" = true ]; then
    black --check --diff "$PATHS"
    echo -e "${YELLOW}ℹ Dry run: no changes made${NC}"
else
    black "$PATHS"
    echo -e "${GREEN}✓ Code formatted${NC}"
fi

echo ""
echo "================================"
echo -e "${GREEN}✓ Formatting complete!${NC}"
echo "================================"
echo ""
echo "Settings (from pyproject.toml):"
echo "  - Line length: 100"
echo "  - Target Python: 3.8+"
echo ""
echo "Usage:"
echo "  bash helios/scripts/format.sh              # Format all"
echo "  bash helios/scripts/format.sh helios/core  # Format specific path"
echo "  bash helios/scripts/format.sh --check      # Dry run (check only)"
echo ""
