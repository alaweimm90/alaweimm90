#!/bin/bash

#
# REPZ One-Command Deployment Script
# Deploys REPZ to production with full validation
#
# Usage: ./scripts/deploy-one-command.sh
#

set -euo pipefail

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_step() {
    echo -e "\n${PURPLE}▶${NC} $1"
}

# Banner
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🚀 REPZ PRODUCTION DEPLOYMENT                          ║
║                                                           ║
║   Deploying fitness coaching platform to production      ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# Step 1: Pre-flight checks
log_step "Step 1: Pre-flight Checks"

log_info "Checking Node.js version..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    log_success "Node.js $NODE_VERSION installed"
else
    log_error "Node.js not found. Please install Node.js 18+"
    exit 1
fi

log_info "Checking npm..."
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    log_success "npm $NPM_VERSION installed"
else
    log_error "npm not found"
    exit 1
fi

log_info "Checking if we're in the right directory..."
if [[ ! -f "package.json" ]]; then
    log_error "package.json not found. Please run from REPZ/platform directory"
    exit 1
fi
log_success "In correct directory"

log_info "Checking environment variables..."
if [[ ! -f ".env.production" ]]; then
    log_warning ".env.production not found"
    log_info "Please create .env.production from .env.production.READY template"
    log_info "See DEPLOY_380_CHECKLIST.md for required variables"
    exit 1
fi
log_success ".env.production found"

# Step 2: Dependency check
log_step "Step 2: Checking Dependencies"

log_info "Installing/updating dependencies..."
npm install --silent
log_success "Dependencies ready"

# Step 3: Type checking
log_step "Step 3: TypeScript Validation"

log_info "Running type check..."
if npm run type-check; then
    log_success "TypeScript validation passed"
else
    log_error "TypeScript errors found. Fix before deploying."
    exit 1
fi

# Step 4: Build
log_step "Step 4: Production Build"

log_info "Building production bundle..."
if npm run build:production; then
    log_success "Build successful"

    # Show build stats
    if [[ -d "dist" ]]; then
        DIST_SIZE=$(du -sh dist | cut -f1)
        log_info "Build size: $DIST_SIZE"
    fi
else
    log_error "Build failed"
    exit 1
fi

# Step 5: Vercel deployment check
log_step "Step 5: Deployment Preparation"

log_info "Checking for Vercel CLI..."
if command -v vercel &> /dev/null; then
    log_success "Vercel CLI found"
else
    log_warning "Vercel CLI not found. Installing..."
    npm install -g vercel
    log_success "Vercel CLI installed"
fi

# Step 6: Deploy
log_step "Step 6: Deploying to Vercel"

log_info "Starting deployment..."
log_warning "This will deploy to PRODUCTION. Press Ctrl+C to cancel, or Enter to continue..."
read -r

log_info "Deploying..."
if vercel --prod; then
    log_success "Deployment successful!"
else
    log_error "Deployment failed"
    exit 1
fi

# Step 7: Post-deployment checks
log_step "Step 7: Post-Deployment Validation"

log_info "Deployment URL will be shown above"
log_info "Next steps:"
echo ""
echo "  1. ✅ Verify deployment at your Vercel URL"
echo "  2. ✅ Test intake form: /intake"
echo "  3. ✅ Test payment flow with Stripe test card"
echo "  4. ✅ Check admin dashboard: /admin/non-portal-clients"
echo "  5. ✅ Monitor Sentry for errors"
echo "  6. ✅ Check SendGrid for email delivery"
echo ""

log_success "REPZ deployment complete! 🎉"

# Optional: Open deployment
log_info "Open deployment in browser? (y/n)"
read -r OPEN_BROWSER

if [[ "$OPEN_BROWSER" == "y" ]] || [[ "$OPEN_BROWSER" == "Y" ]]; then
    log_info "Opening Vercel dashboard..."
    vercel --prod --open
fi

echo ""
log_step "Next: Complete DEPLOY_380_CHECKLIST.md Phase 7 (Testing)"
echo ""

exit 0
