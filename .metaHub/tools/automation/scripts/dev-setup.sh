#!/bin/bash
# Development Environment Setup Script
# Creates a complete development environment for the GitHub monorepo

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

trap 'echo -e "${RED}[x] Setup aborted. Review the errors above for details.${NC}" >&2' ERR

echo -e "${BLUE}=> GitHub Monorepo - Development Environment Setup${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print success
success() {
    echo -e "${GREEN}[ok] $1${NC}"
}

# Function to print error
error() {
    echo -e "${RED}[err] $1${NC}"
}

# Function to print warning
warning() {
    echo -e "${YELLOW}[warn] $1${NC}"
}

# Function to print info
info() {
    echo -e "${BLUE}[info] $1${NC}"
}

# Determine if we can prompt the user
IS_INTERACTIVE=1
if [ ! -t 0 ] || [ "${FORCE_NON_INTERACTIVE:-0}" = "1" ]; then
    IS_INTERACTIVE=0
fi

confirm() {
    local prompt="$1"
    if [ "$IS_INTERACTIVE" -ne 1 ]; then
        info "$prompt Skipped (non-interactive mode)."
        return 1
    fi

    printf "%b" "${YELLOW}${prompt} [y/N]: ${NC}"
    read -r response
    [[ "$response" =~ ^[Yy]$ ]]
}

ensure_command() {
    local cmd="$1"
    local install_help="$2"
    if ! command_exists "$cmd"; then
        error "$cmd is not installed."
        [ -n "$install_help" ] && info "$install_help"
        exit 1
    fi
}

# Step 1: Check prerequisites
echo -e "${BLUE}=> Step 1/10: Checking prerequisites...${NC}"

ensure_command "node" "Install Node.js from https://nodejs.org/ or via nvm."
success "Node.js $(node --version) installed"

if ! command_exists pnpm; then
    if command_exists corepack; then
        info "pnpm not detected. Enabling via corepack..."
        corepack enable pnpm >/dev/null 2>&1 || true
    fi
fi

if ! command_exists pnpm; then
    ensure_command "npm" "Install npm or enable corepack to manage pnpm."
    warning "Installing pnpm globally via npm (requires appropriate permissions)."
    npm install -g pnpm
fi
success "pnpm $(pnpm --version) installed"

ensure_command "git" "Install git from https://git-scm.com/."
success "git $(git --version | cut -d' ' -f3) installed"

# Step 2: Check Node version
echo ""
echo -e "${BLUE}=> Step 2/10: Verifying Node.js version...${NC}"
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    error "Node.js version 18 or higher required (found v${NODE_VERSION})"
    exit 1
fi
success "Node.js version check passed"

# Step 3: Install dependencies
echo ""
echo -e "${BLUE}=> Step 3/10: Installing dependencies...${NC}"
info "This may take a few minutes..."

if pnpm install; then
    success "Dependencies installed"
else
    error "Failed to install dependencies"
    exit 1
fi

# Step 4: Set up git hooks
echo ""
echo -e "${BLUE}=> Step 4/10: Setting up git hooks...${NC}"

# Check if we're in a git repository
if [ ! -d .git ]; then
    warning "Not in a git repository - skipping git hooks setup"
else
    mkdir -p .husky
    if ! pnpm dlx husky install .husky >/dev/null 2>&1; then
        pnpm exec husky install
    fi

    # Copy hooks from automation directory
    if [ -d .automation/hooks ]; then
        cp .automation/hooks/* .husky/ 2>/dev/null || true
        success "Git hooks configured"
    else
        warning "No hooks found in .automation/hooks"
    fi
fi

# Step 5: Configure git
echo ""
echo -e "${BLUE}=> Step 5/10: Configuring git settings...${NC}"

if [ -d .git ]; then
    git config core.hooksPath .husky

    # Set up commit template if it exists
    if [ -f .gitmessage ]; then
        git config commit.template .gitmessage
        success "Git commit template configured"
    fi

    # Configure pull strategy
    git config pull.rebase true

    # Configure default branch
    git config init.defaultBranch main

    success "Git configured"
else
    warning "Not in a git repository - skipping git configuration"
fi

# Step 6: Set up environment variables
echo ""
echo -e "${BLUE}=> Step 6/10: Setting up environment variables...${NC}"

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        success "Created .env from .env.example"
        warning "Please update .env with your configuration"
    else
        warning "No .env.example found - you may need to create .env manually"
    fi
else
    success ".env already exists"
fi

# Step 7: Install recommended tools
echo ""
echo -e "${BLUE}=> Step 7/10: Installing recommended development tools...${NC}"

# Check for recommended global tools
RECOMMENDED_TOOLS=("eslint" "prettier" "typescript")
for tool in "${RECOMMENDED_TOOLS[@]}"; do
    if ! command_exists "$tool"; then
        if command_exists npm; then
            info "Installing $tool globally..."
            npm install -g "$tool"
        else
            warning "npm not available to install $tool globally. Skipping."
        fi
    fi
done

success "Recommended tools checked"

# Step 8: External tools (optional)
echo ""
echo -e "${BLUE}=> Step 8/10: External tool integrations (optional)${NC}"

# PiecesOS
if [ -d .automation/integrations/pieces-os ]; then
    if confirm "Would you like to install PiecesOS?"; then
        if [ -f .automation/integrations/pieces-os/setup.sh ]; then
            bash .automation/integrations/pieces-os/setup.sh
        else
            warning "PiecesOS setup script not found"
        fi
    fi
fi

# Augment CLI
if [ -d .automation/integrations/augment-cli ]; then
    if confirm "Would you like to install Augment CLI?"; then
        if command_exists npm; then
            npm install -g @augment/cli 2>/dev/null || warning "Augment CLI not available or requires authentication"
            if [ -f .automation/integrations/augment-cli/setup.sh ]; then
                bash .automation/integrations/augment-cli/setup.sh
            fi
        else
            warning "npm not available - skipping Augment CLI installation"
        fi
    fi
fi

# Step 9: Run health check
echo ""
echo -e "${BLUE}=> Step 9/10: Running health check...${NC}"

if [ -f .automation/scripts/health-check.sh ]; then
    bash .automation/scripts/health-check.sh || warning "Health check reported issues (see above)."
else
    warning "Health check script not found - skipping"
fi

# Step 10: Build initial projects
echo ""
echo -e "${BLUE}=> Step 10/10: Running initial build (optional)...${NC}"

if confirm "Would you like to build all projects now? This may take a while."; then
    info "Building all projects..."
    if pnpm build; then
        success "Build completed successfully"
    else
        warning "Build failed - you may need to fix errors"
    fi
fi

# Final summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}[done] Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}=> Next Steps:${NC}"
echo "  1. Review and update .env with your configuration"
echo "  2. Read CONTRIBUTING.md for contribution guidelines"
echo "  3. Read .automation/README.md for automation tools"
echo "  4. Run 'pnpm dev' to start development servers"
echo ""
echo -e "${BLUE}=> Useful Commands:${NC}"
echo "  pnpm dev              - Start development servers"
echo "  pnpm build            - Build all projects"
echo "  pnpm test             - Run all tests"
echo "  pnpm lint             - Lint all code"
echo "  pnpm lint:fix         - Auto-fix linting issues"
echo "  pnpm format           - Format all code"
echo "  .automation/scripts/health-check.sh   - Run health diagnostics"
echo "  .automation/scripts/security-scan.sh  - Run security scan"
echo ""
echo -e "${BLUE}=> Documentation:${NC}"
echo "  - Master Plan: WORKFLOW_AUTOMATION_MASTER_PLAN.md"
echo "  - Security: FINAL_SECURITY_REMEDIATION_REPORT.md"
echo "  - Quick Start: .automation/README.md"
echo ""
echo -e "${GREEN}Happy coding!${NC}"
