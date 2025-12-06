#!/bin/bash

# HELIOS Test Runner
# Version: 0.1.0
# Purpose: Run test suite with coverage reporting

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================"
echo "HELIOS Test Suite"
echo "================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}✗ pytest not found. Run: pip install -e .[dev]${NC}"
    exit 1
fi

# Parse arguments
VERBOSE=""
COVERAGE_HTML=false
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-vv"
            shift
            ;;
        --html)
            COVERAGE_HTML=true
            shift
            ;;
        --unit)
            SPECIFIC_TEST="helios/tests/unit/"
            shift
            ;;
        --integration)
            SPECIFIC_TEST="helios/tests/integration/"
            shift
            ;;
        *)
            SPECIFIC_TEST="$1"
            shift
            ;;
    esac
done

# Set test path
TEST_PATH="${SPECIFIC_TEST:-helios/tests/}"

echo -e "${BLUE}Running tests${NC}: $TEST_PATH"
echo ""

# Run pytest with coverage
pytest \
    $VERBOSE \
    --cov=helios \
    --cov-report=term-missing \
    --cov-report=html \
    --tb=short \
    "$TEST_PATH" \
    2>&1 | tee test_results.txt

# Capture exit code
EXIT_CODE=$?

echo ""
echo "================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Tests failed!${NC}"
fi

echo "================================"

# Show coverage report location
if [ "$COVERAGE_HTML" = true ] || grep -q "htmlcov" <(ls -d htmlcov 2>/dev/null); then
    echo ""
    echo "Coverage HTML report:"
    echo "  Open: htmlcov/index.html"
fi

echo ""
echo "Test summary:"
tail -5 test_results.txt

exit $EXIT_CODE
