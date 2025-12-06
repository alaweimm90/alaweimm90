#!/bin/bash

# REPZ Platform - Health Check Script
# Validates that all services are properly configured and running

set -e

echo "🏥 REPZ Platform - Health Check"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0
SUCCESS=0

check_service() {
    local service_name=$1
    local check_command=$2
    local is_required=${3:-true}

    echo -n "Checking $service_name... "

    if eval "$check_command" &> /dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
        ((SUCCESS++))
        return 0
    else
        if [ "$is_required" = true ]; then
            echo -e "${RED}❌ FAILED${NC}"
            ((ERRORS++))
        else
            echo -e "${YELLOW}⚠️  OPTIONAL${NC}"
            ((WARNINGS++))
        fi
        return 1
    fi
}

check_env_var() {
    local var_name=$1
    local is_required=${2:-true}

    if [ -n "${!var_name}" ]; then
        echo -e "${GREEN}✅${NC} $var_name is set"
        ((SUCCESS++))
        return 0
    else
        if [ "$is_required" = true ]; then
            echo -e "${RED}❌${NC} $var_name is missing"
            ((ERRORS++))
        else
            echo -e "${YELLOW}⚠️${NC} $var_name is missing (optional)"
            ((WARNINGS++))
        fi
        return 1
    fi
}

check_url() {
    local url=$1
    local service_name=$2

    echo -n "Testing $service_name... "

    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "^[23]"; then
        echo -e "${GREEN}✅ Responding${NC}"
        ((SUCCESS++))
        return 0
    else
        echo -e "${RED}❌ Not responding${NC}"
        ((ERRORS++))
        return 1
    fi
}

echo "📦 Checking Dependencies"
echo "========================"
check_service "Node.js" "command -v node"
check_service "npm" "command -v npm"
check_service "Git" "command -v git"
check_service "Vercel CLI" "command -v vercel" false
check_service "Supabase CLI" "command -v supabase" false
echo ""

echo "📁 Checking Project Structure"
echo "============================="
check_service "package.json" "test -f package.json"
check_service "vercel.json" "test -f vercel.json"
check_service "tsconfig.json" "test -f tsconfig.json"
check_service "vite.config.ts" "test -f vite.config.ts"
check_service "src directory" "test -d src"
check_service "supabase directory" "test -d supabase"
echo ""

echo "🔐 Checking Environment Variables"
echo "=================================="

# Load environment if it exists
if [ -f ".env.production" ]; then
    source .env.production
    echo -e "${GREEN}✅ .env.production loaded${NC}"
elif [ -f ".env.production.local" ]; then
    source .env.production.local
    echo -e "${GREEN}✅ .env.production.local loaded${NC}"
else
    echo -e "${YELLOW}⚠️  No production environment file found${NC}"
fi

# Required variables
check_env_var "VITE_SUPABASE_URL"
check_env_var "VITE_SUPABASE_ANON_KEY"
check_env_var "VITE_STRIPE_PUBLIC_KEY"

# Optional but recommended
check_env_var "VITE_OPENAI_API_KEY" false
check_env_var "VITE_SENTRY_DSN" false
check_env_var "SENDGRID_API_KEY" false
check_env_var "VITE_WHOOP_CLIENT_ID" false
check_env_var "VITE_GOOGLE_CLIENT_ID" false
check_env_var "VITE_STRAVA_CLIENT_ID" false
echo ""

echo "🏗️  Checking Build"
echo "=================="
echo "Running type check..."
if npm run type-check &> /dev/null; then
    echo -e "${GREEN}✅ TypeScript types valid${NC}"
    ((SUCCESS++))
else
    echo -e "${RED}❌ TypeScript errors found${NC}"
    ((ERRORS++))
fi

echo "Testing production build..."
if npm run build:production &> /dev/null; then
    echo -e "${GREEN}✅ Production build successful${NC}"
    ((SUCCESS++))
else
    echo -e "${RED}❌ Production build failed${NC}"
    ((ERRORS++))
fi
echo ""

echo "🗄️  Checking Database"
echo "====================="
if [ -n "$VITE_SUPABASE_URL" ]; then
    check_url "$VITE_SUPABASE_URL/rest/v1/" "Supabase API"
else
    echo -e "${YELLOW}⚠️  Supabase URL not configured${NC}"
fi
echo ""

echo "📊 Summary"
echo "=========="
echo -e "${GREEN}✅ Successful checks: $SUCCESS${NC}"
echo -e "${YELLOW}⚠️  Warnings: $WARNINGS${NC}"
echo -e "${RED}❌ Errors: $ERRORS${NC}"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}🎉 All critical checks passed!${NC}"
    echo -e "${GREEN}Your platform is ready for deployment!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some critical checks failed.${NC}"
    echo "Please fix the errors above before deploying."
    exit 1
fi
