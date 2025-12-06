#!/bin/bash
# SuperTool Trivy Security Scanner
# Scans Docker images, filesystem, and dependencies for vulnerabilities

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="${1:-supertool:latest}"
SEVERITY="${2:-CRITICAL,HIGH}"
OUTPUT_DIR="/mnt/c/Users/mesha/Desktop/Projects/SuperTool/outputs/security"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   SuperTool Security Scanner (Trivy)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if Trivy is installed
if ! command -v trivy &> /dev/null; then
    echo -e "${YELLOW}⚠ Trivy not installed. Installing...${NC}"

    # Detect OS and install
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux installation
        wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
        echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install trivy -y
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS installation
        brew install aquasecurity/trivy/trivy
    else
        echo -e "${RED}✗ Unsupported OS. Please install Trivy manually: https://aquasecurity.github.io/trivy/${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Trivy version: $(trivy --version)${NC}"
echo ""

# Update vulnerability database
echo -e "${BLUE}ℹ Updating vulnerability database...${NC}"
trivy image --download-db-only

# Function to scan Docker image
scan_image() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}   Scanning Docker Image: $IMAGE_NAME${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Scan image for vulnerabilities
    trivy image \
        --severity "$SEVERITY" \
        --format table \
        --output "$OUTPUT_DIR/image-scan.txt" \
        "$IMAGE_NAME"

    # Generate JSON report
    trivy image \
        --severity "$SEVERITY" \
        --format json \
        --output "$OUTPUT_DIR/image-scan.json" \
        "$IMAGE_NAME"

    # Generate SARIF for GitHub
    trivy image \
        --severity "$SEVERITY" \
        --format sarif \
        --output "$OUTPUT_DIR/image-scan.sarif" \
        "$IMAGE_NAME"

    echo -e "${GREEN}✓ Image scan complete${NC}"
    echo -e "  Reports saved to: $OUTPUT_DIR/"
}

# Function to scan filesystem
scan_filesystem() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}   Scanning Filesystem${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    trivy fs \
        --severity "$SEVERITY" \
        --format table \
        --output "$OUTPUT_DIR/filesystem-scan.txt" \
        /mnt/c/Users/mesha/Desktop/Projects/SuperTool/cli

    trivy fs \
        --severity "$SEVERITY" \
        --format json \
        --output "$OUTPUT_DIR/filesystem-scan.json" \
        /mnt/c/Users/mesha/Desktop/Projects/SuperTool/cli

    echo -e "${GREEN}✓ Filesystem scan complete${NC}"
}

# Function to scan configuration files
scan_config() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}   Scanning Configuration Files${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    trivy config \
        --severity "$SEVERITY" \
        --format table \
        --output "$OUTPUT_DIR/config-scan.txt" \
        /mnt/c/Users/mesha/Desktop/Projects/SuperTool/devops

    echo -e "${GREEN}✓ Configuration scan complete${NC}"
}

# Function to generate SBOM
generate_sbom() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}   Generating SBOM (Software Bill of Materials)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Generate CycloneDX SBOM
    trivy image \
        --format cyclonedx \
        --output "$OUTPUT_DIR/sbom-cyclonedx.json" \
        "$IMAGE_NAME"

    # Generate SPDX SBOM
    trivy image \
        --format spdx-json \
        --output "$OUTPUT_DIR/sbom-spdx.json" \
        "$IMAGE_NAME"

    echo -e "${GREEN}✓ SBOM generated${NC}"
    echo -e "  CycloneDX: $OUTPUT_DIR/sbom-cyclonedx.json"
    echo -e "  SPDX: $OUTPUT_DIR/sbom-spdx.json"
}

# Run all scans
echo -e "${BLUE}ℹ Starting comprehensive security scan...${NC}"
echo -e "${BLUE}ℹ Severity levels: $SEVERITY${NC}"

scan_image
scan_filesystem
scan_config
generate_sbom

# Summary
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   Scan Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}✓ All scans completed successfully${NC}"
echo -e "${BLUE}ℹ Reports saved to: $OUTPUT_DIR/${NC}"
echo ""
ls -lh "$OUTPUT_DIR/"
echo ""

# Check for critical vulnerabilities
CRITICAL_COUNT=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL") | .VulnerabilityID' "$OUTPUT_DIR/image-scan.json" 2>/dev/null | wc -l || echo "0")
HIGH_COUNT=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity=="HIGH") | .VulnerabilityID' "$OUTPUT_DIR/image-scan.json" 2>/dev/null | wc -l || echo "0")

echo -e "${BLUE}Vulnerabilities Found:${NC}"
echo -e "  ${RED}Critical: $CRITICAL_COUNT${NC}"
echo -e "  ${YELLOW}High: $HIGH_COUNT${NC}"
echo ""

if [ "$CRITICAL_COUNT" -gt 0 ]; then
    echo -e "${RED}✗ CRITICAL vulnerabilities found! Review and fix before deployment.${NC}"
    exit 1
else
    echo -e "${GREEN}✓ No CRITICAL vulnerabilities found${NC}"
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
