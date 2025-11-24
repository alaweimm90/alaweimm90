#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Setting up development environment...${NC}"

# Check for required tools
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${YELLOW}⚠️  $1 is not installed. Installing...${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ $1 is installed${NC}"
    return 0
}

# Install Node.js dependencies
echo -e "\n${YELLOW}📦 Installing Node.js dependencies...${NC}
pnpm install

# Install pre-commit hooks
echo -e "\n${YELLOW}🔧 Setting up pre-commit hooks...${NC}
pre-commit install --install-hooks

# Install development tools
if ! check_command "gitleaks"; then
    brew install gitleaks
fi

if ! check_command "trivy"; then
    brew install aquasecurity/trivy/trivy
fi

if ! check_command "hadolint"; then
    brew install hadolint
fi

# Verify setup
echo -e "\n${YELLOW}✅ Development environment setup complete!${NC}
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Run 'pnpm dev' to start the development server"
echo "2. Run 'pnpm test' to run tests"
echo "3. Run 'pnpm lint' to check code style"

# Make the script executable
chmod +x scripts/setup-dev.sh
