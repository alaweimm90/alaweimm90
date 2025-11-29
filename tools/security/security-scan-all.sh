#!/bin/bash
# Comprehensive Security Scanner
# Runs all security scans: SAST, Secret Scanning, and SCA

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../../../outputs/security"

# Tracking
TOTAL_SCANS=0
PASSED_SCANS=0
FAILED_SCANS=0

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  SuperTool Security Scanner Suite     ║${NC}"
echo -e "${BLUE}║  Comprehensive Security Analysis       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Running all security scans...${NC}"
echo ""

# Function to run scan
run_scan() {
    local scan_name="$1"
    local scan_script="$2"

    TOTAL_SCANS=$((TOTAL_SCANS + 1))

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Running: ${scan_name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if bash "${scan_script}"; then
        echo -e "${GREEN}✓ ${scan_name} PASSED${NC}"
        PASSED_SCANS=$((PASSED_SCANS + 1))
        return 0
    else
        echo -e "${RED}✗ ${scan_name} FAILED${NC}"
        FAILED_SCANS=$((FAILED_SCANS + 1))
        return 1
    fi
}

# Run all scans
echo -e "${GREEN}Starting security scans...${NC}"
echo ""

# 1. Secret Scanning (run first, most critical)
run_scan "Secret Scanning (Gitleaks)" "${SCRIPT_DIR}/secret-scan.sh" || true
echo ""

# 2. SAST Scanning
run_scan "SAST Scanning (Semgrep)" "${SCRIPT_DIR}/sast-scan.sh" || true
echo ""

# 3. Dependency Scanning
run_scan "Dependency Scanning (Trivy)" "${SCRIPT_DIR}/dependency-scan.sh" || true
echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Security Scan Summary            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "Total Scans: ${TOTAL_SCANS}"
echo -e "${GREEN}Passed: ${PASSED_SCANS}${NC}"
echo -e "${RED}Failed: ${FAILED_SCANS}${NC}"
echo ""

# Calculate success rate
if [ "${TOTAL_SCANS}" -gt 0 ]; then
    SUCCESS_RATE=$((PASSED_SCANS * 100 / TOTAL_SCANS))
    echo -e "Success Rate: ${SUCCESS_RATE}%"
    echo ""
fi

# Generate summary report
SUMMARY_FILE="${OUTPUT_DIR}/security-summary.json"
cat > "${SUMMARY_FILE}" <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "total_scans": ${TOTAL_SCANS},
  "passed_scans": ${PASSED_SCANS},
  "failed_scans": ${FAILED_SCANS},
  "success_rate": ${SUCCESS_RATE:-0},
  "scans": {
    "secret_scanning": "$([ -f "${OUTPUT_DIR}/secret-scan-report.json" ] && echo "completed" || echo "failed")",
    "sast_scanning": "$([ -f "${OUTPUT_DIR}/sast-report.json" ] && echo "completed" || echo "failed")",
    "dependency_scanning": "$([ -f "${OUTPUT_DIR}/dependency-scan-report.json" ] && echo "completed" || echo "failed")"
  },
  "reports": {
    "secret_scan": "${OUTPUT_DIR}/secret-scan-report.json",
    "sast_scan": "${OUTPUT_DIR}/sast-report.json",
    "dependency_scan": "${OUTPUT_DIR}/dependency-scan-report.json",
    "sbom": "${OUTPUT_DIR}/sbom.json"
  }
}
EOF

echo -e "Summary report: ${SUMMARY_FILE}"
echo ""

# Exit with error if any scan failed
if [ "${FAILED_SCANS}" -gt 0 ]; then
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}Security scans failed!${NC}"
    echo -e "${RED}Please review the reports in: ${OUTPUT_DIR}${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
else
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}All security scans passed!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 0
fi
