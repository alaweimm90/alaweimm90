#!/bin/bash

# HELIOS Code Linter
# Version: 0.1.0
# Purpose: Lint code with flake8 and type check with mypy

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================"
echo "HELIOS Code Linter"
echo "================================"
echo ""

# Check if tools are installed
if ! command -v flake8 &> /dev/null; then
    echo -e "${RED}✗ flake8 not found. Run: pip install -e .[dev]${NC}"
    exit 1
fi

if ! command -v mypy &> /dev/null; then
    echo -e "${RED}✗ mypy not found. Run: pip install -e .[dev]${NC}"
    exit 1
fi

# Parse arguments
PATHS="${@:-helios/}"

echo -e "${BLUE}Linting${NC}: $PATHS"
echo ""

# Run flake8 (style violations)
echo -e "${BLUE}[1/2]${NC} Checking with flake8..."
FLAKE8_EXIT=0
flake8 "$PATHS" \
    --max-line-length=100 \
    --extend-ignore=E203,W503 \
    --count \
    --statistics \
    || FLAKE8_EXIT=$?

if [ $FLAKE8_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ No style issues${NC}"
else
    echo -e "${YELLOW}⚠ Style issues found${NC}"
fi

echo ""

# Run mypy (type checking)
echo -e "${BLUE}[2/2]${NC} Type checking with mypy..."
MYPY_EXIT=0
mypy "$PATHS" \
    --ignore-missing-imports \
    --disallow-untyped-defs=false \
    --warn-return-any \
    --warn-unused-configs \
    || MYPY_EXIT=$?

if [ $MYPY_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ No type errors${NC}"
else
    echo -e "${YELLOW}⚠ Type issues found${NC}"
fi

echo ""
echo "================================"

TOTAL_EXIT=$((FLAKE8_EXIT + MYPY_EXIT))
if [ $TOTAL_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Linting passed!${NC}"
else
    echo -e "${YELLOW}⚠ Linting issues found (see above)${NC}"
fi

echo "================================"
echo ""
echo "Linting tools:"
echo "  - flake8: PEP8 style violations"
echo "  - mypy: Type checking"
echo ""
echo "Settings (from pyproject.toml):"
echo "  - Max line length: 100"
echo "  - Python version: 3.8+"
echo "  - Type checking: Partial (warn_return_any=true)"
echo ""
echo "Usage:"
echo "  bash helios/scripts/lint.sh              # Lint all"
echo "  bash helios/scripts/lint.sh helios/core  # Lint specific path"
echo ""

exit $TOTAL_EXIT
