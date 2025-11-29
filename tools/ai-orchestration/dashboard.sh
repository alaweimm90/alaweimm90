#!/bin/bash
# Enhanced Dashboard v2.3
# AI Tools - Real-time monitoring with auto-refresh

METRICS_DIR="$HOME/.ai_tools/metrics"
LOGS_DIR="$HOME/.ai_tools/logs"
LEARNING_DIR="$HOME/.ai_tools/learning"
CONFIG_DIR="$HOME/.ai_tools/config"

mkdir -p "$METRICS_DIR" "$LOGS_DIR"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; MAGENTA='\033[0;35m'
BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

# Box drawing
H_LINE="─"; V_LINE="│"; TL="┌"; TR="┐"; BL="└"; BR="┘"
T_DOWN="┬"; T_UP="┴"; T_LEFT="┤"; T_RIGHT="├"; CROSS="┼"

# Metrics file
METRICS_FILE="$METRICS_DIR/current.json"

# Initialize metrics
init_metrics() {
    [ -f "$METRICS_FILE" ] || cat > "$METRICS_FILE" << 'EOF'
{
  "tasks": {"total": 0, "success": 0, "failed": 0},
  "tools": {},
  "costs": {"total": 0.00, "by_tool": {}},
  "time": {"total_seconds": 0, "by_tool": {}},
  "last_updated": ""
}
EOF
}

# Draw box
draw_box() {
    local title=$1 width=${2:-40}
    local title_len=${#title}
    local pad_left=$(( (width - title_len - 2) / 2 ))
    local pad_right=$(( width - title_len - 2 - pad_left ))
    
    echo -e "${CYAN}${TL}$(printf "${H_LINE}%.0s" $(seq 1 $pad_left)) ${BOLD}$title${NC}${CYAN} $(printf "${H_LINE}%.0s" $(seq 1 $pad_right))${TR}${NC}"
}

close_box() {
    local width=${1:-40}
    echo -e "${CYAN}${BL}$(printf "${H_LINE}%.0s" $(seq 1 $width))${BR}${NC}"
}

# Get metric value
get_metric() {
    local path=$1
    jq -r "$path // 0" "$METRICS_FILE" 2>/dev/null || echo "0"
}

# Header
show_header() {
    clear
    local now=$(date '+%Y-%m-%d %H:%M:%S')
    echo ""
    echo -e "${BOLD}${MAGENTA}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${MAGENTA}║${NC}           ${BOLD}🤖 AI Tools Dashboard v2.3${NC}                           ${BOLD}${MAGENTA}║${NC}"
    echo -e "${BOLD}${MAGENTA}║${NC}           ${DIM}$now${NC}                            ${BOLD}${MAGENTA}║${NC}"
    echo -e "${BOLD}${MAGENTA}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Task Summary Panel
show_task_summary() {
    draw_box "📊 Task Summary" 35
    
    local total=$(get_metric '.tasks.total')
    local success=$(get_metric '.tasks.success')
    local failed=$(get_metric '.tasks.failed')
    local rate=0
    [ "$total" -gt 0 ] && rate=$(echo "scale=1; $success * 100 / $total" | bc 2>/dev/null || echo "0")
    
    printf "${CYAN}${V_LINE}${NC}  Total Tasks:     ${BOLD}%-15s${NC}${CYAN}${V_LINE}${NC}\n" "$total"
    printf "${CYAN}${V_LINE}${NC}  ${GREEN}✓ Success:${NC}       ${GREEN}%-15s${NC}${CYAN}${V_LINE}${NC}\n" "$success"
    printf "${CYAN}${V_LINE}${NC}  ${RED}✗ Failed:${NC}        ${RED}%-15s${NC}${CYAN}${V_LINE}${NC}\n" "$failed"
    printf "${CYAN}${V_LINE}${NC}  Success Rate:    ${BOLD}%-14s${NC}${CYAN}${V_LINE}${NC}\n" "${rate}%"
    
    close_box 35
}

# Cost Tracker Panel
show_cost_panel() {
    draw_box "💰 Cost Tracking" 35
    
    local total_cost=$(get_metric '.costs.total')
    local today=$(date +%Y-%m-%d)
    local daily_cost=$(jq -r ".costs.daily[\"$today\"] // 0" "$METRICS_FILE" 2>/dev/null || echo "0")
    
    printf "${CYAN}${V_LINE}${NC}  Today:           ${YELLOW}\$%-13s${NC}${CYAN}${V_LINE}${NC}\n" "$daily_cost"
    printf "${CYAN}${V_LINE}${NC}  All-time:        ${BOLD}\$%-13s${NC}${CYAN}${V_LINE}${NC}\n" "$total_cost"
    
    # Top spending tools
    printf "${CYAN}${V_LINE}${NC}  ${DIM}─── By Tool ───${NC}              ${CYAN}${V_LINE}${NC}\n"
    
    jq -r '.costs.by_tool | to_entries | sort_by(-.value) | .[0:3][] | "\(.key)|\(.value)"' "$METRICS_FILE" 2>/dev/null | \
    while IFS='|' read -r tool cost; do
        [ -n "$tool" ] && printf "${CYAN}${V_LINE}${NC}    %-14s ${DIM}\$%-10s${NC}${CYAN}${V_LINE}${NC}\n" "$tool" "$cost"
    done
    
    close_box 35
}

# Tool Usage Panel
show_tool_usage() {
    draw_box "🔧 Tool Usage" 35
    
    jq -r '.tools | to_entries | sort_by(-.value) | .[0:5][] | "\(.key)|\(.value)"' "$METRICS_FILE" 2>/dev/null | \
    while IFS='|' read -r tool count; do
        local bar_len=$((count / 2))
        [ $bar_len -gt 20 ] && bar_len=20
        local bar=$(printf "█%.0s" $(seq 1 $bar_len 2>/dev/null) 2>/dev/null)
        printf "${CYAN}${V_LINE}${NC}  %-10s ${GREEN}%s${NC} %s${CYAN}  ${V_LINE}${NC}\n" "$tool" "$bar" "$count"
    done
    
    [ -z "$(jq -r '.tools | keys[]' "$METRICS_FILE" 2>/dev/null)" ] && \
        printf "${CYAN}${V_LINE}${NC}  ${DIM}No tool usage yet${NC}          ${CYAN}${V_LINE}${NC}\n"
    
    close_box 35
}

# Learning Stats Panel
show_learning_stats() {
    draw_box "🧠 ML Learning" 35
    
    local model="$LEARNING_DIR/routing_model.json"
    if [ -f "$model" ]; then
        local task_count=$(jq -r '.task_history | length' "$model" 2>/dev/null || echo "0")
        local last_train=$(jq -r '.last_training' "$model" 2>/dev/null || echo "never")
        
        printf "${CYAN}${V_LINE}${NC}  Training samples: ${BOLD}%-12s${NC}${CYAN}${V_LINE}${NC}\n" "$task_count"
        printf "${CYAN}${V_LINE}${NC}  Last trained:     ${DIM}%-12s${NC}${CYAN}${V_LINE}${NC}\n" "${last_train:0:10}"
        
        # Show category priors
        printf "${CYAN}${V_LINE}${NC}  ${DIM}─── Priors ───${NC}               ${CYAN}${V_LINE}${NC}\n"
        jq -r '.priors | to_entries[] | "\(.key)|\(.value)"' "$model" 2>/dev/null | head -4 | \
        while IFS='|' read -r cat prob; do
            [ -n "$cat" ] && printf "${CYAN}${V_LINE}${NC}    %-12s ${MAGENTA}%.2f${NC}         ${CYAN}${V_LINE}${NC}\n" "$cat" "$prob"
        done
    else
        printf "${CYAN}${V_LINE}${NC}  ${DIM}No learning data yet${NC}        ${CYAN}${V_LINE}${NC}\n"
    fi
    
    close_box 35
}

# Recent Activity Panel
show_recent_activity() {
    draw_box "📜 Recent Activity" 72
    
    local log="$LOGS_DIR/tasks.log"
    if [ -f "$log" ]; then
        tail -5 "$log" | while IFS= read -r line; do
            local truncated="${line:0:68}"
            printf "${CYAN}${V_LINE}${NC} %-69s ${CYAN}${V_LINE}${NC}\n" "$truncated"
        done
    else
        printf "${CYAN}${V_LINE}${NC} ${DIM}No recent activity${NC}                                                   ${CYAN}${V_LINE}${NC}\n"
    fi
    
    close_box 72
}

# System Health Panel
show_system_health() {
    draw_box "❤️  System Health" 35
    
    # Check key components
    local mcp_status="${GREEN}●${NC} running"
    pgrep -f "mcp-server" > /dev/null 2>&1 || mcp_status="${RED}○${NC} stopped"
    
    local scripts_count=$(ls ~/.ai_tools/scripts/*.sh 2>/dev/null | wc -l)
    local plugins_count=$(ls ~/.ai_tools/plugins/*.yaml 2>/dev/null | wc -l)
    
    printf "${CYAN}${V_LINE}${NC}  MCP Servers:      $mcp_status      ${CYAN}${V_LINE}${NC}\n"
    printf "${CYAN}${V_LINE}${NC}  Scripts:          ${BOLD}%-14s${NC}${CYAN}${V_LINE}${NC}\n" "$scripts_count active"
    printf "${CYAN}${V_LINE}${NC}  Plugins:          ${BOLD}%-14s${NC}${CYAN}${V_LINE}${NC}\n" "$plugins_count loaded"
    
    # Disk usage for ai_tools
    local disk=$(du -sh ~/.ai_tools 2>/dev/null | cut -f1)
    printf "${CYAN}${V_LINE}${NC}  Storage:          ${DIM}%-14s${NC}${CYAN}${V_LINE}${NC}\n" "$disk"
    
    close_box 35
}

# Quick Commands Panel
show_quick_commands() {
    echo ""
    echo -e "${DIM}Quick Commands:${NC}"
    echo -e "  ${CYAN}ai-route${NC} <task>     Route task to best AI"
    echo -e "  ${CYAN}ai-parallel${NC} run     Run parallel tasks"
    echo -e "  ${CYAN}ai-cost${NC} report      View cost breakdown"
    echo -e "  ${CYAN}ai-test${NC} run         Run test suite"
    echo -e "  ${CYAN}ai-undo${NC} list        View checkpoints"
    echo ""
}

# Main dashboard view
show_dashboard() {
    init_metrics
    show_header
    
    # Top row - side by side panels
    echo -e "${BOLD}┌─────────────────────────────────────┬─────────────────────────────────────┐${NC}"
    
    # Use paste to show panels side by side
    paste <(
        show_task_summary
    ) <(
        show_cost_panel
    ) 2>/dev/null | sed 's/\t/  /'
    
    echo ""
    
    paste <(
        show_tool_usage
    ) <(
        show_learning_stats
    ) 2>/dev/null | sed 's/\t/  /'
    
    echo ""
    
    paste <(
        show_system_health
    ) 2>/dev/null
    
    echo ""
    show_recent_activity
    show_quick_commands
}

# Live mode with auto-refresh
live_mode() {
    local interval=${1:-5}
    
    while true; do
        show_dashboard
        echo -e "${DIM}Refreshing in ${interval}s... (Ctrl+C to exit)${NC}"
        sleep "$interval"
    done
}

# Record a task
record_task() {
    local tool=$1 status=$2 duration=${3:-0} cost=${4:-0.00}
    
    init_metrics
    
    local tmp=$(mktemp)
    local today=$(date +%Y-%m-%d)
    
    jq --arg tool "$tool" --arg status "$status" --argjson dur "$duration" --argjson cost "$cost" --arg today "$today" '
        .tasks.total += 1 |
        (if $status == "success" then .tasks.success += 1 else .tasks.failed += 1 end) |
        .tools[$tool] = ((.tools[$tool] // 0) + 1) |
        .costs.total = ((.costs.total // 0) + $cost) |
        .costs.by_tool[$tool] = ((.costs.by_tool[$tool] // 0) + $cost) |
        .costs.daily[$today] = ((.costs.daily[$today] // 0) + $cost) |
        .time.total_seconds = ((.time.total_seconds // 0) + $dur) |
        .time.by_tool[$tool] = ((.time.by_tool[$tool] // 0) + $dur) |
        .last_updated = (now | todate)
    ' "$METRICS_FILE" > "$tmp" && mv "$tmp" "$METRICS_FILE"
    
    # Also log it
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $tool: $status (${duration}s, \$$cost)" >> "$LOGS_DIR/tasks.log"
}

# Reset metrics
reset_metrics() {
    rm -f "$METRICS_FILE"
    init_metrics
    echo "✅ Metrics reset"
}

show_help() {
    echo "AI Tools Dashboard v2.3"
    echo ""
    echo "Usage: ai-dashboard <command> [options]"
    echo ""
    echo "Commands:"
    echo "  show              Show dashboard (default)"
    echo "  live [interval]   Live mode with auto-refresh"
    echo "  record <tool> <status> [duration] [cost]"
    echo "                    Record a task execution"
    echo "  reset             Reset all metrics"
    echo ""
    echo "Examples:"
    echo "  ai-dashboard"
    echo "  ai-dashboard live 10"
    echo "  ai-dashboard record cline success 45 0.02"
}

case "$1" in
    show|"") show_dashboard ;;
    live) live_mode "${2:-5}" ;;
    record) record_task "$2" "$3" "$4" "$5" ;;
    reset) reset_metrics ;;
    -h|--help) show_help ;;
    *) echo "Unknown: $1"; show_help ;;
esac
