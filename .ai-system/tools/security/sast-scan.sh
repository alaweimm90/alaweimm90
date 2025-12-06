#!/bin/bash
# SAST (Static Application Security Testing) Scanner
# Runs Semgrep for code security analysis

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SEMGREP_CONFIG="${SCRIPT_DIR}/../configs/semgrep.yml"
OUTPUT_DIR="${SCRIPT_DIR}/../../../outputs/security"
REPORT_FILE="${OUTPUT_DIR}/sast-report.json"
SARIF_FILE="${OUTPUT_DIR}/sast-report.sarif"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}SAST Scanner - Semgrep${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check if Semgrep is installed
if ! command -v semgrep &> /dev/null; then
    echo -e "${RED}Error: Semgrep is not installed${NC}"
    echo "Install with: pip install semgrep"
    echo "Or: brew install semgrep (macOS)"
    exit 1
fi

echo -e "${YELLOW}Scanning directory:${NC} ${PROJECT_ROOT}"
echo -e "${YELLOW}Configuration:${NC} ${SEMGREP_CONFIG}"
echo ""

# Run Semgrep with custom config
echo -e "${GREEN}Running SAST scan with custom rules...${NC}"
semgrep scan \
    --config "${SEMGREP_CONFIG}" \
    --config "p/security-audit" \
    --config "p/owasp-top-ten" \
    --config "p/ci" \
    --json \
    --output "${REPORT_FILE}" \
    "${PROJECT_ROOT}" || true

# Also generate SARIF format for GitHub Security
echo -e "${GREEN}Generating SARIF report for GitHub Security...${NC}"
semgrep scan \
    --config "${SEMGREP_CONFIG}" \
    --config "p/security-audit" \
    --sarif \
    --output "${SARIF_FILE}" \
    "${PROJECT_ROOT}" || true

# Parse results
if [ -f "${REPORT_FILE}" ]; then
    TOTAL_FINDINGS=$(jq '.results | length' "${REPORT_FILE}")
    ERROR_COUNT=$(jq '[.results[] | select(.extra.severity == "ERROR")] | length' "${REPORT_FILE}")
    WARNING_COUNT=$(jq '[.results[] | select(.extra.severity == "WARNING")] | length' "${REPORT_FILE}")
    INFO_COUNT=$(jq '[.results[] | select(.extra.severity == "INFO")] | length' "${REPORT_FILE}")

    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}SAST Scan Results${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo -e "Total Findings: ${TOTAL_FINDINGS}"
    echo -e "${RED}Errors: ${ERROR_COUNT}${NC}"
    echo -e "${YELLOW}Warnings: ${WARNING_COUNT}${NC}"
    echo -e "Info: ${INFO_COUNT}"
    echo ""
    echo -e "Report saved to: ${REPORT_FILE}"
    echo -e "SARIF report saved to: ${SARIF_FILE}"
    echo ""

    # Display critical findings
    if [ "${ERROR_COUNT}" -gt 0 ]; then
        echo -e "${RED}Critical Findings:${NC}"
        jq -r '.results[] | select(.extra.severity == "ERROR") | "  - \(.check_id): \(.extra.message) (\(.path):\(.start.line))"' "${REPORT_FILE}"
        echo ""
    fi

    # Exit with error if critical findings found
    if [ "${ERROR_COUNT}" -gt 0 ]; then
        echo -e "${RED}SAST scan found ${ERROR_COUNT} critical security issues${NC}"
        exit 1
    else
        echo -e "${GREEN}SAST scan completed successfully${NC}"
        exit 0
    fi
else
    echo -e "${RED}Error: Report file not generated${NC}"
    exit 1
fi
