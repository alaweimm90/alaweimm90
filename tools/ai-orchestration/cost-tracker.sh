#!/bin/bash
# Enhanced Cost Tracker v2.3
# AI Tools - Comprehensive cost monitoring with budgets

METRICS_DIR="$HOME/.ai_tools/metrics"
mkdir -p "$METRICS_DIR"

COST_FILE="$METRICS_DIR/costs.json"
BUDGET_FILE="$METRICS_DIR/budget.json"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

init_costs() {
    [ -f "$COST_FILE" ] || echo '{"total":0,"daily":{},"by_tool":{},"by_model":{}}' > "$COST_FILE"
    [ -f "$BUDGET_FILE" ] || echo '{"daily_limit":10,"monthly_limit":100,"alerts":true}' > "$BUDGET_FILE"
}

record_cost() {
    local tool=$1 amount=$2 model=${3:-"default"}
    init_costs
    local today=$(date +%Y-%m-%d)
    local tmp=$(mktemp)
    jq --arg tool "$tool" --argjson amt "$amount" --arg model "$model" --arg day "$today" '
        .total += $amt |
        .daily[$day] = ((.daily[$day] // 0) + $amt) |
        .by_tool[$tool] = ((.by_tool[$tool] // 0) + $amt) |
        .by_model[$model] = ((.by_model[$model] // 0) + $amt)
    ' "$COST_FILE" > "$tmp" && mv "$tmp" "$COST_FILE"
    
    # Budget check
    local daily=$(jq -r ".daily[\"$today\"] // 0" "$COST_FILE")
    local limit=$(jq -r '.daily_limit' "$BUDGET_FILE")
    if (( $(echo "$daily > $limit * 0.8" | bc -l) )); then
        echo -e "${YELLOW}⚠️  Daily spend: \$$daily / \$$limit ($(echo "$daily * 100 / $limit" | bc)%)${NC}"
    fi
    echo -e "${GREEN}✓${NC} Recorded: \$$amount for $tool"
}

show_report() {
    init_costs
    local today=$(date +%Y-%m-%d)
    echo ""
    echo -e "${CYAN}💰 Cost Report${NC}"
    echo "==============="
    echo ""
    echo "Today:    \$$(jq -r ".daily[\"$today\"] // 0" "$COST_FILE")"
    echo "Total:    \$$(jq -r '.total' "$COST_FILE")"
    echo ""
    echo "By Tool:"
    jq -r '.by_tool | to_entries | sort_by(-.value)[] | "  \(.key): $\(.value)"' "$COST_FILE"
    echo ""
    echo "By Model:"
    jq -r '.by_model | to_entries | sort_by(-.value)[] | "  \(.key): $\(.value)"' "$COST_FILE"
}

set_budget() {
    local daily=$1 monthly=${2:-100}
    jq --argjson d "$daily" --argjson m "$monthly" '.daily_limit=$d | .monthly_limit=$m' "$BUDGET_FILE" > /tmp/b.json && mv /tmp/b.json "$BUDGET_FILE"
    echo "✓ Budget set: \$$daily/day, \$$monthly/month"
}

case "$1" in
    record) record_cost "$2" "$3" "$4" ;;
    report) show_report ;;
    budget) set_budget "$2" "$3" ;;
    reset) rm -f "$COST_FILE" && init_costs && echo "✓ Reset" ;;
    *) echo "Usage: ai-cost <record|report|budget|reset>" ;;
esac
