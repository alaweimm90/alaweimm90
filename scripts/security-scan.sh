#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔍 Running security scans...${NC}"

# Function to run a security check
run_scan() {
    local name=$1
    local cmd=$2
    
    echo -e "\n${YELLOW}Running ${name}...${NC}"
    if eval "$cmd"; then
        echo -e "${GREEN}✓ ${name} passed${NC}"
        return 0
    else
        echo -e "${RED}✗ ${name} found issues${NC}"
        return 1
    fi
}

# Run security checks
EXIT_CODE=0

# Check for secrets in git history
run_scan "GitLeaks (secrets in git history)" "gitleaks detect --source . --verbose --redact" || EXIT_CODE=1

# Check for vulnerable dependencies
run_scan "Trivy (vulnerability scanner)" "trivy fs --severity HIGH,CRITICAL ." || EXIT_CODE=1

# Check for known vulnerabilities in Node.js dependencies
run_scan "npm audit" "npm audit --production" || EXIT_CODE=1

# Check for outdated dependencies
run_scan "Outdated dependencies" "npm outdated" || true # Non-fatal

# Check for misconfigurations
run_scan "TruffleHog (secrets in files)" "trufflehog filesystem . --json" || EXIT_CODE=1

# Check for security headers in web applications
if [ -d "web" ]; then
    run_scan "Security headers check" "npx check-headers $(find web -name '*.html' | head -n 1)" || true
fi

# Exit with appropriate status
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}✅ All security scans passed!${NC}"
else
    echo -e "\n${RED}❌ Some security checks failed. Please review the issues above.${NC}"
fi

exit $EXIT_CODE
