#!/bin/bash

# REPZ Platform - Production Deployment Script
# This script automates the deployment process to Vercel

set -e  # Exit on error

echo "🚀 REPZ Platform - Production Deployment"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${RED}❌ Vercel CLI not found${NC}"
    echo "Install it with: npm install -g vercel"
    exit 1
fi

echo -e "${GREEN}✅ Vercel CLI found${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ package.json not found. Are you in the platform directory?${NC}"
    exit 1
fi

echo -e "${GREEN}✅ In correct directory${NC}"
echo ""

# Run type checking
echo "📝 Running TypeScript type check..."
npm run type-check
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Type check passed${NC}"
else
    echo -e "${RED}❌ Type check failed${NC}"
    exit 1
fi
echo ""

# Build production version
echo "🏗️  Building production version..."
npm run build:production
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi
echo ""

# Check if environment variables are set
echo "🔍 Checking environment variables..."
if [ ! -f ".env.production" ]; then
    echo -e "${YELLOW}⚠️  .env.production not found${NC}"
    echo "Copy .env.production.template and fill in your values"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✅ Environment file found${NC}"
fi
echo ""

# Deploy to Vercel
echo "🚢 Deploying to Vercel production..."
echo "This may take a few minutes..."
echo ""

vercel --prod

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Deployment successful!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Check deployment at your Vercel URL"
    echo "2. Run database migrations if needed"
    echo "3. Test all integrations"
    echo "4. Monitor errors in Sentry"
    echo ""
    echo "View logs: vercel logs --prod"
else
    echo -e "${RED}❌ Deployment failed${NC}"
    echo "Check logs with: vercel logs"
    exit 1
fi
