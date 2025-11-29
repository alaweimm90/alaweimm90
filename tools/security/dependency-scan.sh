#!/bin/bash
# Dependency Scanner (SCA - Software Composition Analysis)
# Runs Trivy for dependency vulnerability scanning

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRIVY_CONFIG="${SCRIPT_DIR}/../configs/trivy.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/../../../outputs/security"
REPORT_FILE="${OUTPUT_DIR}/dependency-scan-report.json"
SARIF_FILE="${OUTPUT_DIR}/dependency-scan-report.sarif"
SBOM_FILE="${OUTPUT_DIR}/sbom.json"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Dependency Scanner - Trivy${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check if Trivy is installed
if ! command -v trivy &> /dev/null; then
    echo -e "${RED}Error: Trivy is not installed${NC}"
    echo "Install with: brew install trivy (macOS)"
    echo "Or: wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -"
    exit 1
fi

echo -e "${YELLOW}Scanning directory:${NC} ${PROJECT_ROOT}"
echo -e "${YELLOW}Configuration:${NC} ${TRIVY_CONFIG}"
echo ""

# Update Trivy database
echo -e "${GREEN}Updating vulnerability database...${NC}"
trivy image --download-db-only

# Scan filesystem for dependencies
echo -e "${GREEN}Running dependency vulnerability scan...${NC}"
trivy fs \
    --config "${TRIVY_CONFIG}" \
    --security-checks vuln \
    --severity CRITICAL,HIGH,MEDIUM \
    --format json \
    --output "${REPORT_FILE}" \
    "${PROJECT_ROOT}"

# Generate SARIF report
echo -e "${GREEN}Generating SARIF report...${NC}"
trivy fs \
    --config "${TRIVY_CONFIG}" \
    --security-checks vuln \
    --severity CRITICAL,HIGH,MEDIUM \
    --format sarif \
    --output "${SARIF_FILE}" \
    "${PROJECT_ROOT}"

# Generate SBOM (Software Bill of Materials)
echo -e "${GREEN}Generating SBOM...${NC}"
trivy fs \
    --format cyclonedx \
    --output "${SBOM_FILE}" \
    "${PROJECT_ROOT}"

# Parse results
if [ -f "${REPORT_FILE}" ]; then
    # Count vulnerabilities
    CRITICAL_COUNT=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "${REPORT_FILE}")
    HIGH_COUNT=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "${REPORT_FILE}")
    MEDIUM_COUNT=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "MEDIUM")] | length' "${REPORT_FILE}")
    TOTAL_COUNT=$((CRITICAL_COUNT + HIGH_COUNT + MEDIUM_COUNT))

    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Dependency Scan Results${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo -e "Total Vulnerabilities: ${TOTAL_COUNT}"
    echo -e "${RED}Critical: ${CRITICAL_COUNT}${NC}"
    echo -e "${YELLOW}High: ${HIGH_COUNT}${NC}"
    echo -e "Medium: ${MEDIUM_COUNT}"
    echo ""
    echo -e "Report saved to: ${REPORT_FILE}"
    echo -e "SARIF report saved to: ${SARIF_FILE}"
    echo -e "SBOM saved to: ${SBOM_FILE}"
    echo ""

    # Display critical vulnerabilities
    if [ "${CRITICAL_COUNT}" -gt 0 ]; then
        echo -e "${RED}Critical Vulnerabilities:${NC}"
        jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL") | "  - \(.VulnerabilityID): \(.PkgName) \(.InstalledVersion) -> \(.FixedVersion // "no fix")"' "${REPORT_FILE}" | head -10
        echo ""
    fi

    # Display high vulnerabilities
    if [ "${HIGH_COUNT}" -gt 0 ] && [ "${CRITICAL_COUNT}" -eq 0 ]; then
        echo -e "${YELLOW}High Severity Vulnerabilities:${NC}"
        jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH") | "  - \(.VulnerabilityID): \(.PkgName) \(.InstalledVersion) -> \(.FixedVersion // "no fix")"' "${REPORT_FILE}" | head -10
        echo ""
    fi

    # Exit with error if critical vulnerabilities found
    if [ "${CRITICAL_COUNT}" -gt 0 ]; then
        echo -e "${RED}Dependency scan found ${CRITICAL_COUNT} critical vulnerabilities${NC}"
        exit 1
    elif [ "${HIGH_COUNT}" -gt 5 ]; then
        echo -e "${YELLOW}Dependency scan found ${HIGH_COUNT} high severity vulnerabilities${NC}"
        exit 1
    else
        echo -e "${GREEN}Dependency scan completed successfully${NC}"
        exit 0
    fi
else
    echo -e "${RED}Error: Report file not generated${NC}"
    exit 1
fi
