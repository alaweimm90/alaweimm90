#!/bin/bash

# REPZ Platform - Cost Calculator
# Interactive tool to calculate your deployment costs

echo "💰 REPZ Platform - Cost Calculator"
echo "===================================="
echo ""
echo "This tool helps you calculate costs for deploying REPZ."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

total_monthly=0
total_onetime=0
services_selected=()

ask_service() {
    local service_name=$1
    local monthly_cost=$2
    local description=$3
    local is_required=${4:-false}

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$service_name${NC}"
    echo "$description"
    echo -e "Cost: ${YELLOW}\$$monthly_cost/month${NC}"

    if [ "$is_required" = true ]; then
        echo -e "${GREEN}✅ REQUIRED${NC}"
        total_monthly=$(echo "$total_monthly + $monthly_cost" | bc)
        services_selected+=("$service_name: \$$monthly_cost/mo")
        return 0
    fi

    read -p "Include this service? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        total_monthly=$(echo "$total_monthly + $monthly_cost" | bc)
        services_selected+=("$service_name: \$$monthly_cost/mo")
        echo -e "${GREEN}✅ Added${NC}"
    else
        echo -e "${YELLOW}⊘ Skipped${NC}"
    fi
}

ask_onetime() {
    local service_name=$1
    local cost=$2
    local description=$3

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$service_name${NC}"
    echo "$description"
    echo -e "Cost: ${YELLOW}\$$cost (one-time)${NC}"

    read -p "Include this service? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        total_onetime=$(echo "$total_onetime + $cost" | bc)
        services_selected+=("$service_name: \$$cost (one-time)")
        echo -e "${GREEN}✅ Added${NC}"
    else
        echo -e "${YELLOW}⊘ Skipped${NC}"
    fi
}

echo "Let's calculate your costs!"
echo ""

# Required services
echo -e "${GREEN}REQUIRED SERVICES${NC}"
echo "These are essential for the platform to work:"

ask_service "Vercel Pro" 20 "Hosting, custom domains, analytics, deployment" true
ask_service "Supabase Pro" 25 "Database, 8GB storage, daily backups, auth" true
ask_service "OpenAI API" 20 "AI features (usage-based, ~\$20 estimated)" true

# Recommended services
echo ""
echo ""
echo -e "${YELLOW}RECOMMENDED SERVICES${NC}"
echo "These greatly enhance the platform:"

ask_service "Sentry" 26 "Error tracking, performance monitoring"
ask_service "SendGrid" 90 "Email service, 100K emails/month"
ask_service "Cloudflare Pro" 20 "CDN, DDoS protection, WAF"

# Optional integrations
echo ""
echo ""
echo -e "${BLUE}OPTIONAL INTEGRATIONS${NC}"
echo "Add these for enhanced functionality:"

ask_service "Whoop API" 50 "Biometric data integration"
ask_onetime "Twilio Credits" 50 "SMS notifications (pay-as-you-go)"

# One-time costs
echo ""
echo ""
echo -e "${BLUE}ONE-TIME COSTS${NC}"

ask_onetime "Apple Developer" 99 "Required for iOS app with HealthKit (\$99/year)"

# Calculate totals
echo ""
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}COST SUMMARY${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "Selected Services:"
for service in "${services_selected[@]}"; do
    echo "  • $service"
done
echo ""

echo -e "Monthly Cost:     ${YELLOW}\$$total_monthly${NC}"
echo -e "One-time Cost:    ${YELLOW}\$$total_onetime${NC}"
echo ""

# Calculate for different periods
month_1=$(echo "$total_monthly + $total_onetime" | bc)
month_3=$(echo "$total_monthly * 3 + $total_onetime" | bc)
month_6=$(echo "$total_monthly * 6 + $total_onetime" | bc)
month_12=$(echo "$total_monthly * 12 + $total_onetime" | bc)

echo "Total Costs:"
echo -e "  Month 1:        ${YELLOW}\$$month_1${NC}"
echo -e "  3 Months:       ${YELLOW}\$$month_3${NC}"
echo -e "  6 Months:       ${YELLOW}\$$month_6${NC}"
echo -e "  12 Months:      ${YELLOW}\$$month_12${NC}"
echo ""

# Budget check
budget=800
if (( $(echo "$month_3 <= $budget" | bc -l) )); then
    remaining=$(echo "$budget - $month_3" | bc)
    echo -e "${GREEN}✅ Under budget! Remaining: \$$remaining${NC}"
else
    overage=$(echo "$month_3 - $budget" | bc)
    echo -e "${YELLOW}⚠️  Over budget by: \$$overage${NC}"
    echo ""
    echo "Consider:"
    echo "  • Skipping Whoop API (saves \$150)"
    echo "  • Reducing SendGrid to 1 month (saves \$180)"
    echo "  • Skipping Twilio credits (saves \$50)"
fi

echo ""
echo "💡 Tip: Google Calendar and Strava are FREE!"
echo ""
