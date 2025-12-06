#!/bin/bash
# Secret Scanner
# Runs Gitleaks to detect hardcoded secrets and credentials

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
GITLEAKS_CONFIG="${SCRIPT_DIR}/../configs/gitleaks.toml"
OUTPUT_DIR="${SCRIPT_DIR}/../../../outputs/security"
REPORT_FILE="${OUTPUT_DIR}/secret-scan-report.json"
SARIF_FILE="${OUTPUT_DIR}/secret-scan-report.sarif"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Secret Scanner - Gitleaks${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check if Gitleaks is installed
if ! command -v gitleaks &> /dev/null; then
    echo -e "${RED}Error: Gitleaks is not installed${NC}"
    echo "Install with: brew install gitleaks (macOS)"
    echo "Or download from: https://github.com/gitleaks/gitleaks/releases"
    exit 1
fi

echo -e "${YELLOW}Scanning directory:${NC} ${PROJECT_ROOT}"
echo -e "${YELLOW}Configuration:${NC} ${GITLEAKS_CONFIG}"
echo ""

# Run Gitleaks detect (for current state)
echo -e "${GREEN}Running secret detection scan...${NC}"
gitleaks detect \
    --source "${PROJECT_ROOT}" \
    --config "${GITLEAKS_CONFIG}" \
    --report-format json \
    --report-path "${REPORT_FILE}" \
    --no-git \
    --verbose || GITLEAKS_EXIT=$?

# Also generate SARIF format
echo -e "${GREEN}Generating SARIF report...${NC}"
gitleaks detect \
    --source "${PROJECT_ROOT}" \
    --config "${GITLEAKS_CONFIG}" \
    --report-format sarif \
    --report-path "${SARIF_FILE}" \
    --no-git || true

# Parse results
if [ -f "${REPORT_FILE}" ]; then
    LEAK_COUNT=$(jq '. | length' "${REPORT_FILE}" 2>/dev/null || echo "0")

    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Secret Scan Results${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""

    if [ "${LEAK_COUNT}" -eq 0 ]; then
        echo -e "${GREEN}No secrets detected!${NC}"
        echo ""
        echo -e "Report saved to: ${REPORT_FILE}"
        echo -e "SARIF report saved to: ${SARIF_FILE}"
        exit 0
    else
        echo -e "${RED}Secrets detected: ${LEAK_COUNT}${NC}"
        echo ""
        echo -e "${RED}Leaked Secrets:${NC}"
        jq -r '.[] | "  - \(.RuleID): \(.Description) (\(.File):\(.StartLine))"' "${REPORT_FILE}"
        echo ""
        echo -e "Full report saved to: ${REPORT_FILE}"
        echo -e "SARIF report saved to: ${SARIF_FILE}"
        echo ""
        echo -e "${RED}Secret scan found ${LEAK_COUNT} leaked secrets${NC}"
        exit 1
    fi
else
    if [ "${LEAK_COUNT:-0}" -eq 0 ]; then
        echo -e "${GREEN}No secrets detected!${NC}"
        exit 0
    else
        echo -e "${RED}Error: Report file not generated${NC}"
        exit 1
    fi
fi
